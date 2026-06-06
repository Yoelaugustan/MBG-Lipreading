"""
plot_history.py — plot training curves from history.json.

Produces SEPARATE images for loss and CER so each can be used as its own
figure in the paper. Supports two modes:
  - Single run: plot one fold/variant
  - Multi-fold overlay: plot all 5 folds for one variant on the same axes

Usage:
    # Single run (default)
    python plot_history.py
    python plot_history.py path/to/history.json

    # Multi-fold overlay for one variant
    python plot_history.py --variant sequential --folds 1 2 3 4 5

Outputs (in same folder as the history.json):
    Single mode:
        training_loss.png   — train and val loss
        training_cer.png    — train and val CER
    Multi-fold mode (saved to Train/runs/comparison/):
        <variant>_fold_overlay_train_loss.png
        <variant>_fold_overlay_val_loss.png
        <variant>_fold_overlay_train_cer.png
        <variant>_fold_overlay_val_cer.png
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


# Color palette for fold overlay (5 distinguishable colors)
FOLD_COLORS = {
    1: "#378ADD",   # blue
    2: "#1D9E75",   # green
    3: "#D85A30",   # orange
    4: "#7F77DD",   # purple
    5: "#C84F8A",   # pink
}


# ──────────────────────────────────────────────────────────────────────────────
# SINGLE RUN — two separate figures (loss and CER)
# ──────────────────────────────────────────────────────────────────────────────
def plot_single_run(history_path: str):
    with open(history_path) as f:
        history = json.load(f)

    if not history:
        print(f"{history_path} is empty")
        return

    epochs     = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"] for h in history]
    train_cer  = [h.get("train_cer", float("nan")) for h in history]
    val_cer    = [h["val_cer"] for h in history]

    out_dir = Path(history_path).parent

    # ── Loss figure ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, train_loss, label="Train", color="#378ADD", linewidth=2)
    ax.plot(epochs, val_loss,   label="Validation",   color="#D85A30", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CTC Loss")
    ax.set_title("Loss curves")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    loss_path = out_dir / "training_loss.png"
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {loss_path}")

    # ── CER figure ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, train_cer, label="Train", color="#378ADD", linewidth=2)
    ax.plot(epochs, val_cer,   label="Validation",   color="#D85A30", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Character Error Rate")
    ax.set_title("CER curves")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    cer_path = out_dir / "training_cer.png"
    plt.savefig(cer_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {cer_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MULTI-FOLD OVERLAY — one variant across all folds
# ──────────────────────────────────────────────────────────────────────────────
def plot_fold_overlay(variant: str, folds: list, runs_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load every fold's history once; reuse for all four figures.
    fold_data = {}
    for fold_num in folds:
        history_path = runs_dir / f"fold{fold_num}_{variant}" / "history.json"
        if not history_path.is_file():
            print(f"  [skip] fold{fold_num}_{variant}: history.json not found")
            continue
        with open(history_path) as f:
            history = json.load(f)
        if not history:
            continue
        fold_data[fold_num] = history

    if not fold_data:
        print("No history.json files found for the specified variant and folds")
        return

    def _overlay(metric_key: str, ylabel: str, split_label: str, fname: str, ylim=None):
        fig, ax = plt.subplots(figsize=(8, 5))
        for fold_num, history in fold_data.items():
            epochs = [h["epoch"] for h in history]
            values = [h.get(metric_key, float("nan")) for h in history]
            color  = FOLD_COLORS.get(fold_num, "gray")
            ax.plot(epochs, values, color=color, linewidth=2, label=f"Fold {fold_num}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{split_label} across 5 folds — {variant}")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        if ylim is not None:
            ax.set_ylim(*ylim)
        plt.tight_layout()
        out_path = out_dir / f"{variant}_fold_overlay_{fname}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    _overlay("train_loss", "CTC Loss",              "Train loss",      "train_loss")
    _overlay("val_loss",   "CTC Loss",              "Validation loss", "val_loss")
    _overlay("train_cer",  "Character Error Rate",  "Train CER",       "train_cer", ylim=(0, 1))
    _overlay("val_cer",    "Character Error Rate",  "Validation CER",  "val_cer",   ylim=(0, 1))


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("history_path", nargs="?",
                        default="Train/runs/fold1_sequential/history.json",
                        help="Path to history.json (single-run mode)")
    parser.add_argument("--variant", type=str, default=None,
                        help="Variant name for multi-fold overlay (e.g., 'sequential')")
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                        help="Folds to overlay (only used with --variant)")
    parser.add_argument("--runs_dir", type=str, default="Train/runs",
                        help="Parent directory containing fold{N}_{variant} subfolders")
    parser.add_argument("--out_dir", type=str, default="Train/runs/comparison",
                        help="Output directory for multi-fold plots")
    args = parser.parse_args()

    if args.variant is not None:
        plot_fold_overlay(
            variant=args.variant,
            folds=args.folds,
            runs_dir=Path(args.runs_dir),
            out_dir=Path(args.out_dir),
        )
    else:
        plot_single_run(args.history_path)


if __name__ == "__main__":
    main()