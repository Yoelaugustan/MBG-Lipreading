"""
test.py - standalone evaluation script with richer metrics.

Usage examples:
    python test.py
    python test.py --checkpoint runs/lumina_sequential/best.pt --variant sequential
    python test.py --split val --save_json runs/lumina_sequential/test_metrics.json
    python test.py --save_csv runs/lumina_sequential/test_predictions.csv
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from config import get_config
from dataset import build_dataloaders
from model import LUMINAModel, count_parameters
from utils import compute_cer, compute_wer, greedy_ctc_decode, levenshtein


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint with richer metrics.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint. Defaults to <output_dir>/best.pt.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Override config.output_dir.")
    parser.add_argument("--variant", type=str, default=None, help="Override config.variant.")
    parser.add_argument("--val_speakers", type=str, default=None, help="Override config.val_speakers.")
    parser.add_argument("--test_speakers", type=str, default=None, help="Override config.test_speakers.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override config.batch_size.")
    parser.add_argument("--num_workers", type=int, default=None, help="Override config.num_workers.")
    parser.add_argument(
        "--save_json",
        type=str,
        nargs="?",
        const="",
        default=None,
        help="Optional path to save aggregate metrics as JSON.",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        nargs="?",
        const="",
        default=None,
        help="Optional path to save per-sample predictions and errors as CSV.",
    )
    parser.add_argument(
        "--show_worst",
        type=int,
        default=10,
        help="How many worst samples (by CER) to print.",
    )
    parser.add_argument(
        "--heatmap_top_k",
        type=int,
        default=25,
        help="How many characters to include on each axis of the confusion heatmap.",
    )
    parser.add_argument(
        "--save_heatmap",
        type=str,
        nargs="?",
        const="",
        default=None,
        help="Optional path for the seaborn confusion heatmap. Defaults to <output_dir>/confusion_heatmap.png.",
    )
    parser.add_argument("--beam_width", type=int, default=0, help="Run simple beam search when >0")
    parser.add_argument("--beam_topk", type=int, default=5, help="Top-k per time-step expansion in beam search")
    parser.add_argument("--lm_n", type=int, default=3, help="n for n-gram LM (word-level)")
    parser.add_argument("--lm_alpha", type=float, default=0.8, help="LM weight for rescoring")
    return parser.parse_args()


def normalize_text(s: str) -> str:
    return " ".join(s.lower().strip().split())


def resolve_output_path(raw_path: str | None, output_dir: str | Path, default_filename: str) -> Path | None:
    if raw_path is None:
        return None
    if raw_path == "":
        return Path(output_dir) / default_filename
    return Path(raw_path)


def build_char_confusion(refs: list[str], hyps: list[str]):
    """Return a nested dict-style confusion table and substitution counts."""
    confusion = defaultdict(Counter)
    substitution_pairs = Counter()

    for ref, hyp in zip(refs, hyps):
        ref_chars = list(ref)
        hyp_chars = list(hyp)

        dp = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]
        for i in range(len(ref_chars) + 1):
            dp[i][0] = i
        for j in range(len(hyp_chars) + 1):
            dp[0][j] = j

        for i in range(1, len(ref_chars) + 1):
            for j in range(1, len(hyp_chars) + 1):
                cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )

        i, j = len(ref_chars), len(hyp_chars)
        edits = []
        while i > 0 or j > 0:
            if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] and ref_chars[i - 1] == hyp_chars[j - 1]:
                edits.append(("match", ref_chars[i - 1], hyp_chars[j - 1]))
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                edits.append(("sub", ref_chars[i - 1], hyp_chars[j - 1]))
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                edits.append(("del", ref_chars[i - 1], ""))
                i -= 1
            else:
                edits.append(("ins", "", hyp_chars[j - 1]))
                j -= 1

        edits.reverse()
        for op, ref_ch, hyp_ch in edits:
            if op == "sub":
                confusion[ref_ch][hyp_ch] += 1
                substitution_pairs[(ref_ch, hyp_ch)] += 1

    return confusion, substitution_pairs


class NGramLM:
    def __init__(self, texts: list[str], n: int = 3):
        self.n = n
        self.counts = {}
        self.context_counts = {}
        self.vocab = set()
        for t in texts:
            words = [w for w in normalize_text(t).split() if w]
            for w in words:
                self.vocab.add(w)
            padded = ["<s>"] * (n - 1) + words + ["</s>"]
            for i in range(len(padded) - n + 1):
                ngram = tuple(padded[i:i + n])
                ctx = ngram[:-1]
                self.counts[ngram] = self.counts.get(ngram, 0) + 1
                self.context_counts[ctx] = self.context_counts.get(ctx, 0) + 1

        self.V = max(1, len(self.vocab))

    def log_prob(self, words: list[str]) -> float:
        # returns log probability (natural log) of the whole sentence
        padded = ["<s>"] * (self.n - 1) + words + ["</s>"]
        total = 0.0
        for i in range(self.n - 1, len(padded)):
            ctx = tuple(padded[i - self.n + 1:i])
            word = padded[i]
            ngram = ctx + (word,)
            num = self.counts.get(ngram, 0) + 1  # add-one smoothing
            den = self.context_counts.get(ctx, 0) + self.V
            total += np.log(num / den)
        return total


def beam_search_simple(log_probs_t: np.ndarray, idx_to_char: dict, beam_width: int = 8, topk: int = 5, lm: NGramLM | None = None, lm_alpha: float = 0.0):
    # log_probs_t: [T, V] numpy array (log probs)
    T, V = log_probs_t.shape
    beams = [([], 0.0)]  # (seq_indices, score)

    for t in range(T):
        row = log_probs_t[t]
        top_indices = np.argsort(row)[-topk:][::-1]
        new_beams = {}
        for seq, sc in beams:
            for c in top_indices:
                new_seq = seq + [int(c)]
                new_sc = sc + float(row[c])
                key = tuple(new_seq)
                if key not in new_beams or new_beams[key] < new_sc:
                    new_beams[key] = new_sc

        # keep top beam_width
        beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        beams = [(list(k), v) for k, v in beams]

    # collapse and aggregate
    final_map = {}
    for seq, sc in beams:
        # collapse repeats and remove blanks (assume blank index is 0)
        chars = []
        prev = None
        for idx in seq:
            if idx == 0:
                prev = idx
                continue
            if idx == prev:
                prev = idx
                continue
            ch = idx_to_char.get(idx, "")
            if ch in ("<blank>", "<unk>"):
                prev = idx
                continue
            chars.append(ch)
            prev = idx

        hyp = "".join(chars)
        if hyp in final_map:
            final_map[hyp] = np.logaddexp(final_map[hyp], sc)
        else:
            final_map[hyp] = sc

    # apply LM rescoring if available
    rescored = {}
    for hyp, sc in final_map.items():
        lm_score = 0.0
        if lm is not None and lm_alpha > 0.0:
            words = [w for w in normalize_text(hyp).split() if w]
            lm_score = lm.log_prob(words)
        rescored[hyp] = sc + lm_alpha * lm_score

    if not rescored:
        return ""
    best = max(rescored.items(), key=lambda x: x[1])[0]
    return best


def print_confusion_summary(confusion, substitution_pairs, top_k: int = 25):
    print("\n=== Top Character Substitutions ===")
    for idx, ((ref_ch, hyp_ch), count) in enumerate(substitution_pairs.most_common(top_k), start=1):
        ref_label = repr(ref_ch)
        hyp_label = repr(hyp_ch)
        print(f"[{idx:02d}] {ref_label} -> {hyp_label} : {count}")

    print("\n=== Confusion Matrix (top reference chars by total substitutions) ===")
    ranked_refs = sorted(confusion.keys(), key=lambda ch: sum(confusion[ch].values()), reverse=True)
    for ref_ch in ranked_refs[:top_k]:
        total = sum(confusion[ref_ch].values())
        top_confusions = ", ".join(
            f"{repr(hyp_ch)}:{count}" for hyp_ch, count in confusion[ref_ch].most_common(5)
        )
        print(f"{repr(ref_ch)} -> total {total} | {top_confusions}")


def save_confusion_heatmap(confusion, out_path: Path, top_k: int = 25):
    ref_totals = Counter({ref_ch: sum(counter.values()) for ref_ch, counter in confusion.items()})
    hyp_totals = Counter()
    for ref_ch, counter in confusion.items():
        for hyp_ch, count in counter.items():
            hyp_totals[hyp_ch] += count

    top_refs = [ch for ch, _ in ref_totals.most_common(top_k)]
    top_hyps = [ch for ch, _ in hyp_totals.most_common(top_k)]
    axis_chars = list(dict.fromkeys(top_refs + top_hyps))

    if not axis_chars:
        raise RuntimeError("No substitutions were found, so a confusion heatmap cannot be built.")

    matrix = pd.DataFrame(0, index=axis_chars, columns=axis_chars, dtype=float)
    for ref_ch in axis_chars:
        for hyp_ch, count in confusion.get(ref_ch, {}).items():
            if hyp_ch in matrix.columns:
                matrix.loc[ref_ch, hyp_ch] = count

    row_sums = matrix.sum(axis=1).replace(0, 1)
    matrix = matrix.div(row_sums, axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(10, 0.45 * len(axis_chars)), max(8, 0.4 * len(axis_chars))))
    sns.heatmap(
        matrix,
        cmap="mako",
        linewidths=0.3,
        linecolor="#222222",
        cbar_kws={"label": "Row-normalized substitution rate"},
    )
    plt.title(f"Character Confusion Heatmap (top {len(axis_chars)} chars)")
    plt.xlabel("Hypothesis character")
    plt.ylabel("Reference character")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion heatmap: {out_path}")


@torch.no_grad()
def evaluate(model, loader, ctc_loss, device, vocab, use_amp: bool):
    idx_to_char = {v: k for k, v in vocab.items()}
    model.eval()

    total_loss = 0.0
    total_n = 0
    refs = []
    hyps = []
    sample_rows = []

    for batch in tqdm(loader, desc="[evaluate]", ncols=100):
        videos = batch["videos"].to(device)
        labels = batch["labels"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        label_lengths = batch["label_lengths"].to(device)

        with autocast("cuda", enabled=use_amp and device.type == "cuda"):
            log_probs = model(videos)

        loss = ctc_loss(log_probs.float(), labels, input_lengths, label_lengths)
        if not (torch.isinf(loss) or torch.isnan(loss)):
            total_loss += loss.item() * videos.size(0)
            total_n += videos.size(0)

        batch_hyps = greedy_ctc_decode(log_probs, idx_to_char, blank=0)
        batch_refs = batch["texts"]
        refs.extend(batch_refs)
        hyps.extend(batch_hyps)

        for ref, hyp in zip(batch_refs, batch_hyps):
            char_ed = levenshtein(list(ref), list(hyp))
            char_len = max(len(ref), 1)
            word_ref = ref.split()
            word_hyp = hyp.split()
            word_ed = levenshtein(word_ref, word_hyp)
            word_len = max(len(word_ref), 1)

            sample_rows.append(
                {
                    "ref": ref,
                    "hyp": hyp,
                    "ref_norm": normalize_text(ref),
                    "hyp_norm": normalize_text(hyp),
                    "exact_match": int(normalize_text(ref) == normalize_text(hyp)),
                    "char_ed": char_ed,
                    "char_len": char_len,
                    "char_er": char_ed / char_len,
                    "char_acc": 1.0 - (char_ed / char_len),
                    "word_ed": word_ed,
                    "word_len": word_len,
                    "word_er": word_ed / word_len,
                    "word_acc": 1.0 - (word_ed / word_len),
                }
            )

    df = pd.DataFrame(sample_rows)
    if df.empty:
        raise RuntimeError("No samples were evaluated. Check data split and manifest paths.")

    cer = compute_cer(refs, hyps)
    wer = compute_wer(refs, hyps)
    mean_loss = total_loss / max(total_n, 1)

    metrics = {
        "num_samples": int(len(df)),
        "loss": float(mean_loss),
        "cer": float(cer),
        "wer": float(wer),
        "char_acc_micro": float(1.0 - cer),
        "word_acc_micro": float(1.0 - wer),
        "sentence_acc": float(df["exact_match"].mean()),
        "char_er_mean": float(df["char_er"].mean()),
        "char_er_median": float(df["char_er"].median()),
        "word_er_mean": float(df["word_er"].mean()),
        "word_er_median": float(df["word_er"].median()),
    }

    return metrics, df


def print_report(metrics: dict, df: pd.DataFrame, show_worst: int):
    print("\n=== Evaluation Summary ===")
    print(f"Samples          : {metrics['num_samples']}")
    print(f"CTC Loss         : {metrics['loss']:.4f}")
    print(f"CER              : {metrics['cer']:.4f}")
    print(f"WER              : {metrics['wer']:.4f}")
    print(f"Sentence Acc     : {metrics['sentence_acc']:.4f}")
    print(f"Char Acc (micro) : {metrics['char_acc_micro']:.4f}")
    print(f"Word Acc (micro) : {metrics['word_acc_micro']:.4f}")
    print(f"Char ER mean/med : {metrics['char_er_mean']:.4f} / {metrics['char_er_median']:.4f}")
    print(f"Word ER mean/med : {metrics['word_er_mean']:.4f} / {metrics['word_er_median']:.4f}")

    print("\n=== Worst Samples (by CER) ===")
    worst = df.sort_values("char_er", ascending=False).head(max(show_worst, 0))
    for i, row in enumerate(worst.itertuples(index=False), start=1):
        print(
            f"[{i:02d}] CER={row.char_er:.3f} | WER={row.word_er:.3f} | "
            f"ref='{row.ref}' | hyp='{row.hyp}'"
        )


def main():
    args = parse_args()
    cfg = get_config()

    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.variant is not None:
        cfg.variant = args.variant
    if args.val_speakers is not None:
        cfg.val_speakers = tuple(args.val_speakers.split(","))
    if args.test_speakers is not None:
        cfg.test_speakers = tuple(args.test_speakers.split(","))
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, vocab = build_dataloaders(cfg)
    if args.split == "train":
        loader = train_loader
    elif args.split == "val":
        loader = val_loader
    else:
        loader = test_loader

    model = LUMINAModel(vocab_size=len(vocab), cfg=cfg).to(device)
    print(f"Model parameters: {count_parameters(model)/1e6:.2f}M")

    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path(cfg.output_dir) / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint: {ckpt_path}")

    ctc_loss = nn.CTCLoss(blank=cfg.ctc_blank, reduction="mean", zero_infinity=True)
    metrics, sample_df = evaluate(model, loader, ctc_loss, device, vocab, use_amp=cfg.use_amp)
    confusion, substitution_pairs = build_char_confusion(sample_df["ref"].tolist(), sample_df["hyp"].tolist())

    metrics.update(
        {
            "split": args.split,
            "checkpoint": str(ckpt_path),
            "variant": cfg.variant,
            "batch_size": int(cfg.batch_size),
        }
    )

    print_report(metrics, sample_df, args.show_worst)
    print_confusion_summary(confusion, substitution_pairs)

    heatmap_path = resolve_output_path(args.save_heatmap, cfg.output_dir, "confusion_heatmap.png")
    if heatmap_path is not None:
        save_confusion_heatmap(confusion, heatmap_path, top_k=args.heatmap_top_k)

    beam_results = None
    # Optional: run simple beam search decoding and LM rescoring
    if args.beam_width > 0:
        print(f"\nRunning beam search (width={args.beam_width}, topk={args.beam_topk})...")

        # Build LM from training texts
        manifest = pd.read_csv(cfg.manifest_csv)
        train_texts = manifest[~manifest["speaker_id"].isin(set(cfg.val_speakers) | set(cfg.test_speakers))]["text"].tolist()
        lm = NGramLM(train_texts, n=args.lm_n) if args.lm_n and args.lm_alpha > 0 else None

        idx_to_char = {v: k for k, v in vocab.items()}

        beam_hyps = []
        # run model over loader again to grab log_probs per batch
        model.eval()
        for batch in tqdm(loader, desc="[beam decode]", ncols=100):
            videos = batch["videos"].to(device)
            with autocast("cuda", enabled=cfg.use_amp and device.type == "cuda"):
                log_probs = model(videos)  # [T, B, V]
                log_probs = log_probs.detach().cpu().numpy()  # [T, B, V]
            T, B, V = log_probs.shape
            for b in range(B):
                lp = log_probs[:, b, :]
                best = beam_search_simple(lp, idx_to_char, beam_width=args.beam_width, topk=args.beam_topk, lm=lm, lm_alpha=args.lm_alpha)
                beam_hyps.append(best)

        # Compute metrics and show delta
        refs = sample_df["ref"].tolist()
        greedy_hyps = sample_df["hyp"].tolist()
        beam_cer = compute_cer(refs, beam_hyps)
        beam_wer = compute_wer(refs, beam_hyps)
        print(f"\nBeam+LM CER: {beam_cer:.4f} | WER: {beam_wer:.4f}")
        print(f"Delta CER (beam - greedy): {beam_cer - metrics['cer']:+.4f}")
        print(f"Delta WER (beam - greedy): {beam_wer - metrics['wer']:+.4f}")

        beam_results = {
            "beam_metrics": {
                "cer": float(beam_cer),
                "wer": float(beam_wer),
                "delta_cer": float(beam_cer - metrics["cer"]),
                "delta_wer": float(beam_wer - metrics["wer"]),
            },
            "beam_predictions": [
                {
                    "ref": ref,
                    "greedy": greedy,
                    "beam": beam,
                }
                for ref, greedy, beam in zip(refs, greedy_hyps, beam_hyps)
            ],
        }

        # save beam predictions
        out_beam_csv = Path(cfg.output_dir) / "beam_predictions.csv"
        pd.DataFrame({"ref": refs, "greedy": greedy_hyps, "beam": beam_hyps}).to_csv(out_beam_csv, index=False)
        print(f"Saved beam predictions: {out_beam_csv}")

    if beam_results is not None:
        metrics.update(beam_results)

    out_json = resolve_output_path(args.save_json, cfg.output_dir, "test_metrics.json")
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics JSON: {out_json}")

    out_csv = resolve_output_path(args.save_csv, cfg.output_dir, "test_predictions.csv")
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        sample_df.to_csv(out_csv, index=False)
        print(f"Saved per-sample CSV: {out_csv}")


if __name__ == "__main__":
    main()
