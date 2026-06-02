"""
Generate a torchinfo summary of the model architecture.

Usage:
    python Train/print_summary.py              # main sequential model
    python Train/print_summary.py parallel     # any other variant: parallel, bigru_only, mamba_only

Requires:
    pip install torchinfo
"""
import sys
import torch
from torchinfo import summary
from config import get_config
from model import LUMINAModel, VARIANTS_NEEDING_MAMBA

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "sequential"
VOCAB_SIZE = 27  # adjust if vocab.json has different size

cfg = get_config()
cfg.variant = VARIANT
cfg.frontend_pretrained = False  # skip downloading weights, summary only needs structure

# Mamba's causal_conv1d kernel is CUDA-only; variants without Mamba can use CPU.
needs_cuda = VARIANT in VARIANTS_NEEDING_MAMBA
if needs_cuda and not torch.cuda.is_available():
    sys.exit(f"[error] variant '{VARIANT}' uses Mamba and requires CUDA, but no GPU is available.")
device = "cuda" if needs_cuda else "cpu"

model = LUMINAModel(vocab_size=VOCAB_SIZE, cfg=cfg).to(device)

print(f"\n{'='*92}")
print(f"Model variant: {VARIANT}  (device: {device})")
print(f"{'='*92}\n")

summary(
    model,
    input_size=(1, cfg.num_frames, cfg.input_channels, cfg.aug_crop_size, cfg.aug_crop_size),
    col_names=("input_size", "output_size", "num_params"),
    depth=3,
    device=device,
    verbose=2,
)

"""
Generate a torchinfo summary of the model architecture.

Usage:
    python Train/print_summary.py              # main sequential model
    python Train/print_summary.py parallel     # any other variant: parallel, bigru_only, mamba_only

Requires:
    pip install torchinfo
"""
import sys
import torch
from torchinfo import summary
from config import get_config
from model import LUMINAModel, VARIANTS_NEEDING_MAMBA

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "sequential"
VOCAB_SIZE = 27  # adjust if vocab.json has different size

cfg = get_config()
cfg.variant = VARIANT
cfg.frontend_pretrained = False  # skip downloading weights, summary only needs structure

# Mamba's causal_conv1d kernel is CUDA-only; variants without Mamba can use CPU.
needs_cuda = VARIANT in VARIANTS_NEEDING_MAMBA
if needs_cuda and not torch.cuda.is_available():
    sys.exit(f"[error] variant '{VARIANT}' uses Mamba and requires CUDA, but no GPU is available.")
device = "cuda" if needs_cuda else "cpu"

model = LUMINAModel(vocab_size=VOCAB_SIZE, cfg=cfg).to(device)

print(f"\n{'='*92}")
print(f"Model variant: {VARIANT}  (device: {device})")
print(f"{'='*92}\n")

summary(
    model,
    input_size=(1, cfg.num_frames, cfg.input_channels, cfg.aug_crop_size, cfg.aug_crop_size),
    col_names=("input_size", "output_size", "num_params"),
    depth=3,
    device=device,
    verbose=2,
)
