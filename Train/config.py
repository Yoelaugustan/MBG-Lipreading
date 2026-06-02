"""
config.py — all hyperparameters in one place.
Edit values here; everything else imports from this file.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ─── Paths ────────────────────────────────────────────────────────────────
    data_root    : str = "./LUMINA_preprocessed"
    manifest_csv : str = "./LUMINA_preprocessed/manifest.csv"
    vocab_json   : str = "./LUMINA_preprocessed/vocab.json"
    output_dir   : str = "runs/lumina_exp1"

    # ─── Data ─────────────────────────────────────────────────────────────────
    num_frames : int = 84
    roi_size   : int = 88

    # Speaker-independent split (speaker IDs held out from training)
    # Default: keep one female for val, one male for test (adjust as you like)
    val_speakers  : tuple = ("P12",)
    test_speakers : tuple = ("P08",)

    # Augmentation (train only)
    aug_random_crop    : bool  = True     # crop 80x80 random window from 88x88
    aug_crop_size      : int   = 80
    aug_horizontal_flip: bool  = True
    aug_time_mask_p    : float = 0.5      # probability of applying time mask
    aug_time_mask_max  : int   = 10       # max consecutive frames to mask

    # ─── Model ────────────────────────────────────────────────────────────────
    hidden_dim      : int = 512
    frontend_pretrained: bool = True       # ImageNet-pretrained ResNet-18
    input_channels  : int = 1              # 1 = grayscale, 3 = RGB

    # ─── Variant selection for ablation studies ──────────────────────────────
    # Main model: "sequential" (Mamba → Bi-GRU stacked).
    # Options: "parallel" | "bigru_only" | "mamba_only" | "sequential"
    variant : str = "sequential"

    mamba_d_state : int = 16
    mamba_d_conv  : int = 4
    mamba_expand  : int = 2

    gru_hidden : int = 256    # bidirectional → output = 2 * 256 = 512
    gru_layers : int = 2

    dropout : float = 0.2

    # ─── Training ─────────────────────────────────────────────────────────────
    batch_size  : int   = 32
    num_workers : int   = 4
    epochs      : int   = 100
    lr          : float = 1e-4
    weight_decay: float = 1e-4

    # Gradient stability
    grad_clip_norm : float = 5.0
    use_amp        : bool  = True

    # LR schedule: linear warmup → cosine decay
    warmup_ratio   : float = 0.05

    # CTC settings
    ctc_blank : int = 0          # must match <blank> index in vocab.json

    # Logging / checkpoints
    log_every          : int  = 50    # log training loss every N steps
    save_every_epochs  : int  = 5     # save a checkpoint every N epochs
    # Early stopping — halt training if val CER stops improving
    early_stopping_patience  : int   = 10      # stop after N epochs without improvement
    early_stopping_min_delta : float = 0.001   # minimum CER decrease to count as improvement
    seed               : int  = 42


def get_config() -> Config:
    cfg = Config()
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    return cfg