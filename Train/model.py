"""
model.py — sentence-level lip reading model with variant support.

Architecture:
    Video [B, T, 1, 80, 80]   (grayscale, single channel)
        │
        ▼
    3D conv stem  (temporal context, 1→64 channels)
        │
        ▼
    2D ResNet-18 per frame  (ImageNet pretrained, layer1–layer4)   → [B, T, 512]
        │
        ▼
    Temporal backend (variant-specific)                              → [B, T, 512]
        │
        ▼
    CTC head (Linear → vocab_size) → log_softmax                     → [T, B, V]

Variants (selected via cfg.variant):
  • "sequential" : Mamba → Bi-GRU (stacked), projection to D  (main model)
  • "parallel"   : Mamba ‖ Bi-GRU, concat + projection         (ablation variant)
  • "bigru_only" : Bi-GRU only, projection to D
  • "mamba_only" : Mamba only, projection to D  (kept for param-count parity)

Design decisions:
  • Frontend is shared across all variants — only the temporal backend differs
  • Sequential composition was selected as the main model based on ablation
    results showing it outperforms parallel composition on LUMINA.
  • For the parallel variant, `load_state_dict` transparently remaps old
    top-level keys (mamba.*, bigru.*, fuse.*, ...) to `temporal.*` so
    pre-refactor checkpoints still load without error.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("[warning] mamba_ssm not available — install with: pip install mamba-ssm")


VARIANTS_NEEDING_MAMBA = {"parallel", "mamba_only", "sequential"}


# ──────────────────────────────────────────────────────────────────────────────
# FRONTEND: 3D stem + 2D ResNet-18 (pretrained)
# ──────────────────────────────────────────────────────────────────────────────
class LipFrontend(nn.Module):
    """
    Input : [B, T, 1, H, W]    H = W = 80 after random/center crop  (grayscale)
    Output: [B, T, 512]        temporal resolution preserved
    """

    def __init__(self, pretrained: bool = True, in_channels: int = 1):
        super().__init__()

        # 3D conv stem: small temporal kernel captures local mouth dynamics.
        # Stride (1, 2, 2) keeps T intact while halving H, W.
        self.stem3d = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(5, 7, 7),
                      stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        # 2D ResNet-18 — reuse layer1..layer4 from torchvision weights.
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r18     = resnet18(weights=weights)

        self.resnet_body = nn.Sequential(
            r18.layer1,   # 64 →  64
            r18.layer2,   # 64 → 128
            r18.layer3,   # 128→ 256
            r18.layer4,   # 256→ 512
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Initialise the 3D stem weights (ResNet body keeps its pretrained weights)
        for m in self.stem3d.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape

        # Move C before T for Conv3d: [B, C, T, H, W]
        x = x.transpose(1, 2)
        x = self.stem3d(x)                        # [B, 64, T, H/4, W/4]

        # Apply 2D ResNet per frame → collapse B and T
        x = x.transpose(1, 2).contiguous()        # [B, T, 64, H/4, W/4]
        x = x.view(B * T, 64, x.size(3), x.size(4))

        x = self.resnet_body(x)                   # [B*T, 512, H'', W'']
        x = self.avgpool(x).flatten(1)            # [B*T, 512]
        x = x.view(B, T, 512)                     # [B, T, 512]
        return x


# ──────────────────────────────────────────────────────────────────────────────
# TEMPORAL BACKENDS  (each takes [B, T, D] and returns [B, T, D])
# ──────────────────────────────────────────────────────────────────────────────
class ParallelBackend(nn.Module):
    """
    Ablation variant: Mamba ‖ Bi-GRU (parallel), concatenate, project back to D.
    Retains the pre-refactor submodule names/shapes so legacy checkpoints load.
    """
    def __init__(self, cfg):
        super().__init__()
        D = cfg.hidden_dim

        self.mamba = Mamba(
            d_model = D,
            d_state = cfg.mamba_d_state,
            d_conv  = cfg.mamba_d_conv,
            expand  = cfg.mamba_expand,
        )
        self.mamba_norm = nn.LayerNorm(D)

        self.bigru = nn.GRU(
            input_size    = D,
            hidden_size   = cfg.gru_hidden,
            num_layers    = cfg.gru_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = cfg.dropout if cfg.gru_layers > 1 else 0.0,
        )
        bigru_out_dim = cfg.gru_hidden * 2
        self.bigru_norm = nn.LayerNorm(bigru_out_dim)

        self.fuse = nn.Sequential(
            nn.Linear(D + bigru_out_dim, D),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        m = self.mamba(f)
        m = self.mamba_norm(m)

        g, _ = self.bigru(f)
        g = self.bigru_norm(g)

        return self.fuse(torch.cat([m, g], dim=-1))


class BiGRUOnlyBackend(nn.Module):
    """Bi-GRU only, projected back to D."""
    def __init__(self, cfg):
        super().__init__()
        D = cfg.hidden_dim

        self.bigru = nn.GRU(
            input_size    = D,
            hidden_size   = cfg.gru_hidden,
            num_layers    = cfg.gru_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = cfg.dropout if cfg.gru_layers > 1 else 0.0,
        )
        bigru_out_dim = cfg.gru_hidden * 2
        self.bigru_norm = nn.LayerNorm(bigru_out_dim)

        self.proj = nn.Sequential(
            nn.Linear(bigru_out_dim, D),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        g, _ = self.bigru(f)
        g = self.bigru_norm(g)
        return self.proj(g)


class MambaOnlyBackend(nn.Module):
    """Mamba only, projected back to D (projection kept for param-count parity)."""
    def __init__(self, cfg):
        super().__init__()
        D = cfg.hidden_dim

        self.mamba = Mamba(
            d_model = D,
            d_state = cfg.mamba_d_state,
            d_conv  = cfg.mamba_d_conv,
            expand  = cfg.mamba_expand,
        )
        self.mamba_norm = nn.LayerNorm(D)

        self.proj = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        m = self.mamba(f)
        m = self.mamba_norm(m)
        return self.proj(m)


class SequentialBackend(nn.Module):
    """Mamba → Bi-GRU (stacked), projected back to D."""
    def __init__(self, cfg):
        super().__init__()
        D = cfg.hidden_dim

        self.mamba = Mamba(
            d_model = D,
            d_state = cfg.mamba_d_state,
            d_conv  = cfg.mamba_d_conv,
            expand  = cfg.mamba_expand,
        )
        self.mamba_norm = nn.LayerNorm(D)

        self.bigru = nn.GRU(
            input_size    = D,
            hidden_size   = cfg.gru_hidden,
            num_layers    = cfg.gru_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = cfg.dropout if cfg.gru_layers > 1 else 0.0,
        )
        bigru_out_dim = cfg.gru_hidden * 2
        self.bigru_norm = nn.LayerNorm(bigru_out_dim)

        self.proj = nn.Sequential(
            nn.Linear(bigru_out_dim, D),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        m = self.mamba(f)
        m = self.mamba_norm(m)
        g, _ = self.bigru(m)
        g = self.bigru_norm(g)
        return self.proj(g)


BACKENDS = {
    "parallel"  : ParallelBackend,
    "bigru_only": BiGRUOnlyBackend,
    "mamba_only": MambaOnlyBackend,
    "sequential": SequentialBackend,
}


# ──────────────────────────────────────────────────────────────────────────────
# MAIN MODEL
# ──────────────────────────────────────────────────────────────────────────────
class LUMINAModel(nn.Module):
    def __init__(self, vocab_size: int, cfg):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim

        if cfg.variant not in BACKENDS:
            raise ValueError(
                f"Unknown variant: {cfg.variant!r}. "
                f"Valid options: {list(BACKENDS.keys())}"
            )
        if cfg.variant in VARIANTS_NEEDING_MAMBA and not MAMBA_AVAILABLE:
            raise RuntimeError(
                f"mamba_ssm is required for variant {cfg.variant!r} — "
                f"pip install mamba-ssm"
            )

        print(f"[model] Using variant: {cfg.variant}")

        # Frontend (shared across all variants)
        self.frontend      = LipFrontend(pretrained=cfg.frontend_pretrained,
                                         in_channels=cfg.input_channels)
        self.frontend_norm = nn.LayerNorm(D)
        self.dropout_in    = nn.Dropout(cfg.dropout)

        # Temporal backend (swappable)
        self.temporal = BACKENDS[cfg.variant](cfg)

        # CTC classification head
        self.ctc_head = nn.Linear(D, vocab_size)

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Back-compat loader: pre-refactor checkpoints for the parallel variant
        stored its submodules at the top level (mamba.*, mamba_norm.*, bigru.*,
        bigru_norm.*, fuse.*). The refactor moves them under "temporal.".
        If we detect the old layout, remap keys transparently.
        """
        if self.cfg.variant == "parallel" and not any(
            k.startswith("temporal.") for k in state_dict
        ):
            old_prefixes = ("mamba.", "mamba_norm.", "bigru.", "bigru_norm.", "fuse.")
            remapped = {}
            for k, v in state_dict.items():
                if any(k.startswith(p) for p in old_prefixes):
                    remapped["temporal." + k] = v
                else:
                    remapped[k] = v
            state_dict = remapped
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        video : [B, T, 1, H, W]
        returns log_probs [T, B, vocab_size]  (ready for CTCLoss)
        """
        f = self.frontend(video)            # [B, T, 512]
        f = self.frontend_norm(f)
        f = self.dropout_in(f)

        h = self.temporal(f)                # [B, T, 512]

        logits = self.ctc_head(h)           # [B, T, vocab_size]
        # CTCLoss expects [T, B, V] log-probs
        return F.log_softmax(logits, dim=-1).transpose(0, 1)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

"""
model.py — sentence-level lip reading model with variant support.

Architecture:
    Video [B, T, 1, 80, 80]   (grayscale, single channel)
        │
        ▼
    3D conv stem  (temporal context, 1→64 channels)
        │
        ▼
    2D ResNet-18 per frame  (ImageNet pretrained, layer1–layer4)   → [B, T, 512]
        │
        ▼
    Temporal backend (variant-specific)                              → [B, T, 512]
        │
        ▼
    CTC head (Linear → vocab_size) → log_softmax                     → [T, B, V]

Variants (selected via cfg.variant):
  • "sequential" : Mamba → Bi-GRU (stacked), projection to D  (main model)
  • "parallel"   : Mamba ‖ Bi-GRU, concat + projection         (ablation variant)
  • "bigru_only" : Bi-GRU only, projection to D
  • "mamba_only" : Mamba only, projection to D  (kept for param-count parity)

Design decisions:
  • Frontend is shared across all variants — only the temporal backend differs
  • Sequential composition was selected as the main model based on ablation
    results showing it outperforms parallel composition on LUMINA.
  • For the parallel variant, `load_state_dict` transparently remaps old
    top-level keys (mamba.*, bigru.*, fuse.*, ...) to `temporal.*` so
    pre-refactor checkpoints still load without error.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("[warning] mamba_ssm not available — install with: pip install mamba-ssm")


VARIANTS_NEEDING_MAMBA = {"parallel", "mamba_only", "sequential"}


# ──────────────────────────────────────────────────────────────────────────────
# FRONTEND: 3D stem + 2D ResNet-18 (pretrained)
# ──────────────────────────────────────────────────────────────────────────────
class LipFrontend(nn.Module):
    """
    Input : [B, T, 1, H, W]    H = W = 80 after random/center crop  (grayscale)
    Output: [B, T, 512]        temporal resolution preserved
    """

    def __init__(self, pretrained: bool = True, in_channels: int = 1):
        super().__init__()

        # 3D conv stem: small temporal kernel captures local mouth dynamics.
        # Stride (1, 2, 2) keeps T intact while halving H, W.
        self.stem3d = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(5, 7, 7),
                      stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        # 2D ResNet-18 — reuse layer1..layer4 from torchvision weights.
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r18     = resnet18(weights=weights)

        self.resnet_body = nn.Sequential(
            r18.layer1,   # 64 →  64
            r18.layer2,   # 64 → 128
            r18.layer3,   # 128→ 256
            r18.layer4,   # 256→ 512
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Initialise the 3D stem weights (ResNet body keeps its pretrained weights)
        for m in self.stem3d.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape

        # Move C before T for Conv3d: [B, C, T, H, W]
        x = x.transpose(1, 2)
        x = self.stem3d(x)                        # [B, 64, T, H/4, W/4]

        # Apply 2D ResNet per frame → collapse B and T
        x = x.transpose(1, 2).contiguous()        # [B, T, 64, H/4, W/4]
        x = x.view(B * T, 64, x.size(3), x.size(4))

        x = self.resnet_body(x)                   # [B*T, 512, H'', W'']
        x = self.avgpool(x).flatten(1)            # [B*T, 512]
        x = x.view(B, T, 512)                     # [B, T, 512]
        return x


# ──────────────────────────────────────────────────────────────────────────────
# TEMPORAL BACKENDS  (each takes [B, T, D] and returns [B, T, D])
# ──────────────────────────────────────────────────────────────────────────────
class ParallelBackend(nn.Module):
    """
    Ablation variant: Mamba ‖ Bi-GRU (parallel), concatenate, project back to D.
    Retains the pre-refactor submodule names/shapes so legacy checkpoints load.
    """
    def __init__(self, cfg):
        super().__init__()
        D = cfg.hidden_dim

        self.mamba = Mamba(
            d_model = D,
            d_state = cfg.mamba_d_state,
            d_conv  = cfg.mamba_d_conv,
            expand  = cfg.mamba_expand,
        )
        self.mamba_norm = nn.LayerNorm(D)

        self.bigru = nn.GRU(
            input_size    = D,
            hidden_size   = cfg.gru_hidden,
            num_layers    = cfg.gru_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = cfg.dropout if cfg.gru_layers > 1 else 0.0,
        )
        bigru_out_dim = cfg.gru_hidden * 2
        self.bigru_norm = nn.LayerNorm(bigru_out_dim)

        self.fuse = nn.Sequential(
            nn.Linear(D + bigru_out_dim, D),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        m = self.mamba(f)
        m = self.mamba_norm(m)

        g, _ = self.bigru(f)
        g = self.bigru_norm(g)

        return self.fuse(torch.cat([m, g], dim=-1))


class BiGRUOnlyBackend(nn.Module):
    """Bi-GRU only, projected back to D."""
    def __init__(self, cfg):
        super().__init__()
        D = cfg.hidden_dim

        self.bigru = nn.GRU(
            input_size    = D,
            hidden_size   = cfg.gru_hidden,
            num_layers    = cfg.gru_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = cfg.dropout if cfg.gru_layers > 1 else 0.0,
        )
        bigru_out_dim = cfg.gru_hidden * 2
        self.bigru_norm = nn.LayerNorm(bigru_out_dim)

        self.proj = nn.Sequential(
            nn.Linear(bigru_out_dim, D),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        g, _ = self.bigru(f)
        g = self.bigru_norm(g)
        return self.proj(g)


class MambaOnlyBackend(nn.Module):
    """Mamba only, projected back to D (projection kept for param-count parity)."""
    def __init__(self, cfg):
        super().__init__()
        D = cfg.hidden_dim

        self.mamba = Mamba(
            d_model = D,
            d_state = cfg.mamba_d_state,
            d_conv  = cfg.mamba_d_conv,
            expand  = cfg.mamba_expand,
        )
        self.mamba_norm = nn.LayerNorm(D)

        self.proj = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        m = self.mamba(f)
        m = self.mamba_norm(m)
        return self.proj(m)


class SequentialBackend(nn.Module):
    """Mamba → Bi-GRU (stacked), projected back to D."""
    def __init__(self, cfg):
        super().__init__()
        D = cfg.hidden_dim

        self.mamba = Mamba(
            d_model = D,
            d_state = cfg.mamba_d_state,
            d_conv  = cfg.mamba_d_conv,
            expand  = cfg.mamba_expand,
        )
        self.mamba_norm = nn.LayerNorm(D)

        self.bigru = nn.GRU(
            input_size    = D,
            hidden_size   = cfg.gru_hidden,
            num_layers    = cfg.gru_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = cfg.dropout if cfg.gru_layers > 1 else 0.0,
        )
        bigru_out_dim = cfg.gru_hidden * 2
        self.bigru_norm = nn.LayerNorm(bigru_out_dim)

        self.proj = nn.Sequential(
            nn.Linear(bigru_out_dim, D),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        m = self.mamba(f)
        m = self.mamba_norm(m)
        g, _ = self.bigru(m)
        g = self.bigru_norm(g)
        return self.proj(g)


BACKENDS = {
    "parallel"  : ParallelBackend,
    "bigru_only": BiGRUOnlyBackend,
    "mamba_only": MambaOnlyBackend,
    "sequential": SequentialBackend,
}


# ──────────────────────────────────────────────────────────────────────────────
# MAIN MODEL
# ──────────────────────────────────────────────────────────────────────────────
class LUMINAModel(nn.Module):
    def __init__(self, vocab_size: int, cfg):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim

        if cfg.variant not in BACKENDS:
            raise ValueError(
                f"Unknown variant: {cfg.variant!r}. "
                f"Valid options: {list(BACKENDS.keys())}"
            )
        if cfg.variant in VARIANTS_NEEDING_MAMBA and not MAMBA_AVAILABLE:
            raise RuntimeError(
                f"mamba_ssm is required for variant {cfg.variant!r} — "
                f"pip install mamba-ssm"
            )

        print(f"[model] Using variant: {cfg.variant}")

        # Frontend (shared across all variants)
        self.frontend      = LipFrontend(pretrained=cfg.frontend_pretrained,
                                         in_channels=cfg.input_channels)
        self.frontend_norm = nn.LayerNorm(D)
        self.dropout_in    = nn.Dropout(cfg.dropout)

        # Temporal backend (swappable)
        self.temporal = BACKENDS[cfg.variant](cfg)

        # CTC classification head
        self.ctc_head = nn.Linear(D, vocab_size)

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Back-compat loader: pre-refactor checkpoints for the parallel variant
        stored its submodules at the top level (mamba.*, mamba_norm.*, bigru.*,
        bigru_norm.*, fuse.*). The refactor moves them under "temporal.".
        If we detect the old layout, remap keys transparently.
        """
        if self.cfg.variant == "parallel" and not any(
            k.startswith("temporal.") for k in state_dict
        ):
            old_prefixes = ("mamba.", "mamba_norm.", "bigru.", "bigru_norm.", "fuse.")
            remapped = {}
            for k, v in state_dict.items():
                if any(k.startswith(p) for p in old_prefixes):
                    remapped["temporal." + k] = v
                else:
                    remapped[k] = v
            state_dict = remapped
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        video : [B, T, 1, H, W]
        returns log_probs [T, B, vocab_size]  (ready for CTCLoss)
        """
        f = self.frontend(video)            # [B, T, 512]
        f = self.frontend_norm(f)
        f = self.dropout_in(f)

        h = self.temporal(f)                # [B, T, 512]

        logits = self.ctc_head(h)           # [B, T, vocab_size]
        # CTCLoss expects [T, B, V] log-probs
        return F.log_softmax(logits, dim=-1).transpose(0, 1)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
