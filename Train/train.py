"""
train.py — main training entry point.

Usage:
    python train.py

    # override any config field:
    python train.py --batch_size 16 --epochs 50 --lr 5e-5
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from config  import get_config
from dataset import build_dataloaders
from model   import LUMINAModel, count_parameters
from utils   import greedy_ctc_decode, compute_cer, compute_wer


# ──────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    """Linear warmup followed by cosine decay to 0."""
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def parse_overrides():
    """Allow any config field to be overridden from CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",  type=int,   default=None)
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--output_dir",  type=str,   default=None)
    parser.add_argument("--resume",      type=str,   default=None,
                        help="Path to checkpoint to resume from")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN / VALIDATE
# ──────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, scaler, ctc_loss, device, cfg, epoch):
    model.train()
    running_loss, running_n = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", ncols=100)

    for step, batch in enumerate(pbar):
        videos        = batch["videos"].to(device,        non_blocking=True)
        labels        = batch["labels"].to(device,        non_blocking=True)
        input_lengths = batch["input_lengths"].to(device, non_blocking=True)
        label_lengths = batch["label_lengths"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=cfg.use_amp):
            log_probs = model(videos)                         # [T, B, V]
            # CTC expects FP32 log-probs — cast back if AMP is on
            loss = ctc_loss(log_probs.float(), labels, input_lengths, label_lengths)

        # Skip bad batches (T < L etc.) — zero_infinity makes CTC return inf, not NaN
        if torch.isinf(loss) or torch.isnan(loss):
            continue

        scaler.scale(loss).backward()
        # Unscale before clipping so the grad-norm threshold is in real units
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * videos.size(0)
        running_n    += videos.size(0)

        if (step + 1) % cfg.log_every == 0:
            pbar.set_postfix(loss=f"{running_loss/running_n:.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

    return running_loss / max(running_n, 1)


@torch.no_grad()
def validate(model, loader, ctc_loss, device, vocab, cfg, tag="val"):
    model.eval()
    idx_to_char = {v: k for k, v in vocab.items()}

    total_loss, total_n = 0.0, 0
    all_refs, all_hyps = [], []

    for batch in tqdm(loader, desc=f"[{tag}]", ncols=100):
        videos        = batch["videos"].to(device)
        labels        = batch["labels"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        label_lengths = batch["label_lengths"].to(device)

        with autocast('cuda', enabled=cfg.use_amp):
            log_probs = model(videos)

        loss = ctc_loss(log_probs.float(), labels, input_lengths, label_lengths)
        if not (torch.isinf(loss) or torch.isnan(loss)):
            total_loss += loss.item() * videos.size(0)
            total_n    += videos.size(0)

        hyps = greedy_ctc_decode(log_probs, idx_to_char, blank=cfg.ctc_blank)
        all_hyps.extend(hyps)
        all_refs.extend(batch["texts"])

    cer = compute_cer(all_refs, all_hyps)
    wer = compute_wer(all_refs, all_hyps)
    mean_loss = total_loss / max(total_n, 1)
    return mean_loss, cer, wer, all_refs, all_hyps


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_overrides()
    cfg  = get_config()
    for k, v in vars(args).items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)

    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("[warning] mamba_ssm requires CUDA — training on CPU will fail.")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, vocab = build_dataloaders(cfg)
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = LUMINAModel(vocab_size=vocab_size, cfg=cfg).to(device)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params/1e6:.2f}M")

    # ── Loss / optimizer / scheduler / AMP scaler ────────────────────────────
    ctc_loss  = nn.CTCLoss(blank=cfg.ctc_blank, reduction="mean", zero_infinity=True)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs
    scheduler   = build_scheduler(optimizer, total_steps, cfg.warmup_ratio)
    scaler      = GradScaler('cuda', enabled=cfg.use_amp)

    # ── Resume (optional) ────────────────────────────────────────────────────
    start_epoch, best_cer = 1, float("inf")
    epochs_without_improvement = 0
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_cer    = ckpt.get("best_cer", float("inf"))
        epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # ── Training loop ────────────────────────────────────────────────────────
    history = []
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, ctc_loss, device, cfg, epoch
        )
        val_loss, val_cer, val_wer, refs, hyps = validate(
            model, val_loader, ctc_loss, device, vocab, cfg, tag="val"
        )
        dt = time.time() - t0

        log_line = (f"Epoch {epoch:3d} | "
                    f"train_loss {train_loss:.4f} | "
                    f"val_loss {val_loss:.4f} | "
                    f"val_CER {val_cer:.4f} | "
                    f"val_WER {val_wer:.4f} | "
                    f"time {dt:.1f}s")
        print(log_line)
        # Show 3 decoded samples for sanity check
        for r, h in list(zip(refs, hyps))[:3]:
            print(f"   ref: {r}")
            print(f"   hyp: {h}")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_cer": val_cer, "val_wer": val_wer, "time_s": dt,
        })
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # ── Checkpointing ────────────────────────────────────────────────────
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_cer": best_cer,
            "epochs_without_improvement": epochs_without_improvement,
            "vocab": vocab,
        }
        if epoch % cfg.save_every_epochs == 0:
            torch.save(ckpt, output_dir / f"ckpt_epoch{epoch}.pt")
        torch.save(ckpt, output_dir / "latest.pt")

        if val_cer < best_cer - cfg.early_stopping_min_delta:
            best_cer = val_cer
            epochs_without_improvement = 0
            ckpt["best_cer"] = best_cer
            torch.save(ckpt, output_dir / "best.pt")
            print(f"  ↳ new best CER: {best_cer:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  ↳ no improvement ({epochs_without_improvement}/{cfg.early_stopping_patience})")
            if epochs_without_improvement >= cfg.early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch}. Best CER: {best_cer:.4f}")
                break

    # ── Final test evaluation with the best checkpoint ───────────────────────
    print("\nLoading best checkpoint for final test evaluation...")
    best_ckpt = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model"])
    test_loss, test_cer, test_wer, _, _ = validate(
        model, test_loader, ctc_loss, device, vocab, cfg, tag="test"
    )
    print(f"FINAL TEST — loss {test_loss:.4f} | CER {test_cer:.4f} | WER {test_wer:.4f}")


if __name__ == "__main__":
    main()