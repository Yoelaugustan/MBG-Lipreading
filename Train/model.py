"""
model.py — sentence-level lip reading model.

Architecture:
    Video [B, T, 1, 80, 80]   (grayscale, single channel)
        │
        ▼
    3D conv stem  (temporal context, 1→64 channels)
        │
        ▼
    2D ResNet-18 per frame  (ImageNet pretrained, layer1–layer4)   → [B, T, 512]
        │
        ├──── Mamba branch    ──► [B, T, 512] ──┐
        │                                        │ concat
        └──── Bi-GRU branch   ──► [B, T, 512] ──┘
                                          │
                                     Linear proj (1024→512)
                                          │
                                     CTC head (Linear → vocab_size)
                                          │
                                     log_softmax  → [T, B, vocab_size]

Design decisions:
  • 3D stem + 2D ResNet-18 instead of r2plus1d_18 → preserves T=84 for char CTC
  • Parallel (not sequential) Mamba/Bi-GRU → shorter gradient paths
  • LayerNorm after each branch → stabilizes gradients
  • ImageNet pretrained ResNet-18 → strong prior for 80×80 crops
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
# MAIN MODEL
# ──────────────────────────────────────────────────────────────────────────────
class LUMINAModel(nn.Module):
    def __init__(self, vocab_size: int, cfg):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim        # 512

        # Frontend
        self.frontend      = LipFrontend(pretrained=cfg.frontend_pretrained,
                                         in_channels=cfg.input_channels)
        self.frontend_norm = nn.LayerNorm(D)
        self.dropout_in    = nn.Dropout(cfg.dropout)

        # ── Parallel branch A: Mamba ─────────────────────────────────────────
        if not MAMBA_AVAILABLE:
            raise RuntimeError("mamba_ssm is required — pip install mamba-ssm")
        self.mamba      = Mamba(
            d_model = D,
            d_state = cfg.mamba_d_state,
            d_conv  = cfg.mamba_d_conv,
            expand  = cfg.mamba_expand,
        )
        self.mamba_norm = nn.LayerNorm(D)

        # ── Parallel branch B: Bi-GRU ─────────────────────────────────────────
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

        # ── Fuse and classify ─────────────────────────────────────────────────
        fused_dim = D + bigru_out_dim                       # 512 + 512 = 1024
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, D),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.ctc_head = nn.Linear(D, vocab_size)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        video : [B, T, 1, H, W]
        returns log_probs [T, B, vocab_size]  (ready for CTCLoss)
        """
        f = self.frontend(video)                     # [B, T, 512]
        f = self.frontend_norm(f)
        f = self.dropout_in(f)

        # Parallel branches — both read the same frontend features
        m = self.mamba(f)                            # [B, T, 512]
        m = self.mamba_norm(m)

        g, _ = self.bigru(f)                         # [B, T, 512]
        g = self.bigru_norm(g)

        # Concatenate along channel axis, then project back to D
        h = torch.cat([m, g], dim=-1)                # [B, T, 1024]
        h = self.fuse(h)                             # [B, T, 512]

        logits = self.ctc_head(h)                    # [B, T, vocab_size]

        # CTCLoss expects [T, B, V] log-probs
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        return log_probs


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)