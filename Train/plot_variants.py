"""
plot_variants.py — compare training runs across architecture variants.

Reads history.json from each variant's run directory, produces:
  1. 2x2 grid of train/val loss + train/val CER curves
  2. Bar chart of best val CER and WER across variants
  3. comparison.csv with best metrics per variant

Usage:
    python plot_variants.py
    python plot_variants.py --runs_dir Train/runs
    python plot_variants.py --variants parallel bigru_only mamba_only sequential
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


VARIANT_STYLE = {
    "parallel":   {"color": "#7F77DD", "label": "Parallel"},
    "bigru_only": {"color": "#1D9E75", "label": "Bi-GRU only"},
    "mamba_only": {"color": "#D85A30", "label": "Mamba only"},
    "sequential": {"color": "#378ADD", "label": "Sequential (ours)"},
}


def load_history(history_path: Path) -> list:
    if not history_path.exists():
        return []
    with open(history_path) as f:
        return json.load(f)


def compute_best(history: list) -> dict:
    if not history:
        return {}
    return min(history, key=lambda h: h["val_cer"])


def plot_curves(all_histories: dict, output_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax_tl, ax_vl), (ax_tc, ax_vc) = axes

    for variant_name, history in all_histories.items():
        if not history:
            continue
        style = VARIANT_STYLE.get(variant_name, {"color": "gray", "label": variant_name})
        epochs = [h["epoch"] for h in history]
        ax_tl.plot(epochs, [h["train_loss"] for h in history], linewidth=2, **style)
        ax_vl.plot(epochs, [h["val_loss"]   for h in history], linewidth=2, **style)
        ax_tc.plot(epochs, [h.get("train_cer", float("nan")) for h in history], linewidth=2, **style)
        ax_vc.plot(epochs, [h["val_cer"] for h in history], linewidth=2, **style)

    for ax, title, ylabel in [
        (ax_tl, "Training loss",   "CTC loss"),
        (ax_vl, "Validation loss", "CTC loss"),
        (ax_tc, "Training CER",    "Character Error Rate"),
        (ax_vc, "Validation CER",  "Character Error Rate"),
    ]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)

    ax_tc.set_ylim(bottom=0)
    ax_vc.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  saved: {output_path}")
    plt.close(fig)


def plot_best_comparison(results: list, output_path: Path):
    if not results:
        return
    variants = [r["variant"] for r in results]
    labels   = [VARIANT_STYLE.get(v, {"label": v})["label"] for v in variants]
    colors   = [VARIANT_STYLE.get(v, {"color": "gray"})["color"] for v in variants]
    cers     = [r["val_cer"] for r in results]
    wers     = [r["val_wer"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, 0.7 * len(variants))))
    y_pos = list(range(len(variants)))

    ax1.barh(y_pos, cers, color=colors)
    ax1.set_yticks(y_pos); ax1.set_yticklabels(labels)
    ax1.set_xlabel("Best Validation CER")
    ax1.set_title("Character Error Rate (lower is better)")
    ax1.grid(axis="x", alpha=0.3)
    for i, v in enumerate(cers):
        ax1.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=9)
    ax1.invert_yaxis()

    ax2.barh(y_pos, wers, color=colors)
    ax2.set_yticks(y_pos); ax2.set_yticklabels(labels)
    ax2.set_xlabel("Best Validation WER")
    ax2.set_title("Word Error Rate (lower is better)")
    ax2.grid(axis="x", alpha=0.3)
    for i, v in enumerate(wers):
        ax2.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=9)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default="Train/runs")
    parser.add_argument("--variants", nargs="+",
                        default=["parallel", "bigru_only", "mamba_only", "sequential"])
    parser.add_argument("--out_dir",  type=str, default="Train/runs/comparison")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_histories = {}
    results       = []

    print(f"Looking for runs in {runs_dir.resolve()}")
    for variant in args.variants:
        run_path = runs_dir / f"lumina_{variant}"
        history  = load_history(run_path / "history.json")
        if not history:
            print(f"  [skip] {variant}: no history.json in {run_path}")
            continue
        all_histories[variant] = history
        best = compute_best(history)
        results.append({
            "variant":        variant,
            "best_epoch":     best["epoch"],
            "val_loss":       round(best["val_loss"], 4),
            "val_cer":        round(best["val_cer"],  4),
            "val_wer":        round(best["val_wer"],  4),
            "epochs_trained": len(history),
        })
        print(f"  [ok]   {variant}: best CER {best['val_cer']:.4f} at epoch {best['epoch']}")

    if not results:
        print("\nNo results found. Did you train any variants yet?")
        return

    results.sort(key=lambda r: r["val_cer"])

    print("\nGenerating plots...")
    plot_curves(all_histories, out_dir / "variants_curves.png")
    plot_best_comparison(results, out_dir / "variants_best.png")

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "comparison.csv", index=False)
    print(f"  saved: {out_dir / 'comparison.csv'}")

    print("\n" + "=" * 60)
    print("FINAL COMPARISON (ranked by val CER)")
    print("=" * 60)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

"""
plot_variants.py — compare training runs across architecture variants.

Reads history.json from each variant's run directory, produces:
  1. 2x2 grid of train/val loss + train/val CER curves
  2. Bar chart of best val CER and WER across variants
  3. comparison.csv with best metrics per variant

Usage:
    python plot_variants.py
    python plot_variants.py --runs_dir Train/runs
    python plot_variants.py --variants parallel bigru_only mamba_only sequential
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


VARIANT_STYLE = {
    "parallel":   {"color": "#7F77DD", "label": "Parallel"},
    "bigru_only": {"color": "#1D9E75", "label": "Bi-GRU only"},
    "mamba_only": {"color": "#D85A30", "label": "Mamba only"},
    "sequential": {"color": "#378ADD", "label": "Sequential (ours)"},
}


def load_history(history_path: Path) -> list:
    if not history_path.exists():
        return []
    with open(history_path) as f:
        return json.load(f)


def compute_best(history: list) -> dict:
    if not history:
        return {}
    return min(history, key=lambda h: h["val_cer"])


def plot_curves(all_histories: dict, output_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax_tl, ax_vl), (ax_tc, ax_vc) = axes

    for variant_name, history in all_histories.items():
        if not history:
            continue
        style = VARIANT_STYLE.get(variant_name, {"color": "gray", "label": variant_name})
        epochs = [h["epoch"] for h in history]
        ax_tl.plot(epochs, [h["train_loss"] for h in history], linewidth=2, **style)
        ax_vl.plot(epochs, [h["val_loss"]   for h in history], linewidth=2, **style)
        ax_tc.plot(epochs, [h.get("train_cer", float("nan")) for h in history], linewidth=2, **style)
        ax_vc.plot(epochs, [h["val_cer"] for h in history], linewidth=2, **style)

    for ax, title, ylabel in [
        (ax_tl, "Training loss",   "CTC loss"),
        (ax_vl, "Validation loss", "CTC loss"),
        (ax_tc, "Training CER",    "Character Error Rate"),
        (ax_vc, "Validation CER",  "Character Error Rate"),
    ]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)

    ax_tc.set_ylim(bottom=0)
    ax_vc.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  saved: {output_path}")
    plt.close(fig)


def plot_best_comparison(results: list, output_path: Path):
    if not results:
        return
    variants = [r["variant"] for r in results]
    labels   = [VARIANT_STYLE.get(v, {"label": v})["label"] for v in variants]
    colors   = [VARIANT_STYLE.get(v, {"color": "gray"})["color"] for v in variants]
    cers     = [r["val_cer"] for r in results]
    wers     = [r["val_wer"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, 0.7 * len(variants))))
    y_pos = list(range(len(variants)))

    ax1.barh(y_pos, cers, color=colors)
    ax1.set_yticks(y_pos); ax1.set_yticklabels(labels)
    ax1.set_xlabel("Best Validation CER")
    ax1.set_title("Character Error Rate (lower is better)")
    ax1.grid(axis="x", alpha=0.3)
    for i, v in enumerate(cers):
        ax1.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=9)
    ax1.invert_yaxis()

    ax2.barh(y_pos, wers, color=colors)
    ax2.set_yticks(y_pos); ax2.set_yticklabels(labels)
    ax2.set_xlabel("Best Validation WER")
    ax2.set_title("Word Error Rate (lower is better)")
    ax2.grid(axis="x", alpha=0.3)
    for i, v in enumerate(wers):
        ax2.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=9)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default="Train/runs")
    parser.add_argument("--variants", nargs="+",
                        default=["parallel", "bigru_only", "mamba_only", "sequential"])
    parser.add_argument("--out_dir",  type=str, default="Train/runs/comparison")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_histories = {}
    results       = []

    print(f"Looking for runs in {runs_dir.resolve()}")
    for variant in args.variants:
        run_path = runs_dir / f"lumina_{variant}"
        history  = load_history(run_path / "history.json")
        if not history:
            print(f"  [skip] {variant}: no history.json in {run_path}")
            continue
        all_histories[variant] = history
        best = compute_best(history)
        results.append({
            "variant":        variant,
            "best_epoch":     best["epoch"],
            "val_loss":       round(best["val_loss"], 4),
            "val_cer":        round(best["val_cer"],  4),
            "val_wer":        round(best["val_wer"],  4),
            "epochs_trained": len(history),
        })
        print(f"  [ok]   {variant}: best CER {best['val_cer']:.4f} at epoch {best['epoch']}")

    if not results:
        print("\nNo results found. Did you train any variants yet?")
        return

    results.sort(key=lambda r: r["val_cer"])

    print("\nGenerating plots...")
    plot_curves(all_histories, out_dir / "variants_curves.png")
    plot_best_comparison(results, out_dir / "variants_best.png")

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "comparison.csv", index=False)
    print(f"  saved: {out_dir / 'comparison.csv'}")

    print("\n" + "=" * 60)
    print("FINAL COMPARISON (ranked by val CER)")
    print("=" * 60)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
