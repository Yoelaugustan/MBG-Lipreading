"""
plot_history.py — plot training curves from history.json.

Usage:
    python plot_history.py                    # uses runs/lumina_exp1/history.json
    python plot_history.py path/to/history.json
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def plot(history_path: str):
    with open(history_path) as f:
        history = json.load(f)

    if not history:
        print("history.json is empty")
        return

    epochs     = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"] for h in history]
    train_cer  = [h.get("train_cer", float('nan')) for h in history]
    val_cer    = [h["val_cer"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ─── Loss curves ────────────────────────────────────────────────
    ax1.plot(epochs, train_loss, label="Train", color="#378ADD", linewidth=2)
    ax1.plot(epochs, val_loss,   label="Val",   color="#D85A30", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CTC Loss")
    ax1.set_title("Loss curves")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ─── CER curves ─────────────────────────────────────────────────
    ax2.plot(epochs, train_cer, label="Train CER", color="#378ADD", linewidth=2)
    ax2.plot(epochs, val_cer,   label="Val CER",   color="#D85A30", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Character Error Rate")
    ax2.set_title("CER curves")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()

    out_path = Path(history_path).parent / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {out_path}")
    plt.show()


if __name__ == "__main__":
    default_path = "Train/runs/lumina_bigru_only/history.json"
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    plot(path)

"""
plot_history.py — plot training curves from history.json.

Usage:
    python plot_history.py                    # uses runs/lumina_exp1/history.json
    python plot_history.py path/to/history.json
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def plot(history_path: str):
    with open(history_path) as f:
        history = json.load(f)

    if not history:
        print("history.json is empty")
        return

    epochs     = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"] for h in history]
    train_cer  = [h.get("train_cer", float('nan')) for h in history]
    val_cer    = [h["val_cer"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ─── Loss curves ────────────────────────────────────────────────
    ax1.plot(epochs, train_loss, label="Train", color="#378ADD", linewidth=2)
    ax1.plot(epochs, val_loss,   label="Val",   color="#D85A30", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CTC Loss")
    ax1.set_title("Loss curves")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ─── CER curves ─────────────────────────────────────────────────
    ax2.plot(epochs, train_cer, label="Train CER", color="#378ADD", linewidth=2)
    ax2.plot(epochs, val_cer,   label="Val CER",   color="#D85A30", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Character Error Rate")
    ax2.set_title("CER curves")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()

    out_path = Path(history_path).parent / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {out_path}")
    plt.show()


if __name__ == "__main__":
    default_path = "Train/runs/lumina_bigru_only/history.json"
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    plot(path)
