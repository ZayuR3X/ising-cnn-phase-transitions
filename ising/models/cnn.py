"""
ising/models/cnn.py

IsingCNN architecture for phase classification and temperature regression
in the 2D Ising Model.

Key design choices:
  - Circular (toroidal) padding in all conv layers to respect PBC
  - 4-channel physics-informed input
  - Global physics feature injection before the dense head
  - Dual output: phase classification + temperature regression

Reference:
    Tanaka & Tomiya (2017), J. Phys. Soc. Jpn. 86, 063001
    Carrasquilla & Melko (2017), Nature Physics 13, 431
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# Circular padding helper (respects toroidal / PBC boundary conditions)
# ---------------------------------------------------------------------------

class CircularPad2d(nn.Module):
    """
    Pads a 2D tensor with circular (wrap-around) padding on all four sides.
    Equivalent to periodic boundary conditions on the lattice.

    Parameters
    ----------
    padding : int — padding size on each side
    """

    def __init__(self, padding: int = 1):
        super().__init__()
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.pad order: (left, right, top, bottom)
        p = self.padding
        return F.pad(x, (p, p, p, p), mode="circular")


# ---------------------------------------------------------------------------
# Reusable conv block: CircularPad → Conv → BN → GELU
# ---------------------------------------------------------------------------

class CircularConvBlock(nn.Module):
    """
    Conv block with circular padding, batch normalization, and GELU activation.

    Parameters
    ----------
    in_channels  : input channels
    out_channels : output channels
    kernel_size  : convolution kernel size (square)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            CircularPad2d(pad),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Main IsingCNN
# ---------------------------------------------------------------------------

class IsingCNN(nn.Module):
    """
    Convolutional Neural Network for 2D Ising Model phase analysis.

    Architecture
    ------------
    Input  : (B, 4, L, L)  — 4-channel physics-informed spin representation
    Block 1: Local pattern detection     (3x3 conv, 32 filters) x2 + MaxPool
    Block 2: Mesoscale cluster geometry  (3x3 + 5x5 conv, 64 filters) + MaxPool
    Block 3: Long-range correlations     (3x3 conv, 128 filters) x2 + AvgPool
    Inject : global physics features [m, chi, E, U4] concatenated
    Head   : Dense layers → [phase_logit (2), T_pred (1)]

    Parameters
    ----------
    in_channels      : number of input channels (default 4)
    n_physics_feats  : number of global physics features to inject (default 4)
    dropout_rates    : (p1, p2) dropout after first two dense layers
    """

    def __init__(
        self,
        in_channels: int = 4,
        n_physics_feats: int = 4,
        dropout_rates: Tuple[float, float] = (0.3, 0.2),
    ):
        super().__init__()

        self.n_physics_feats = n_physics_feats

        # -----------------------------------------------------------
        # Block 1: local spin pattern detection
        # -----------------------------------------------------------
        self.block1 = nn.Sequential(
            CircularConvBlock(in_channels, 32, kernel_size=3),
            CircularConvBlock(32, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # -----------------------------------------------------------
        # Block 2: mesoscale cluster geometry
        # -----------------------------------------------------------
        self.block2 = nn.Sequential(
            CircularConvBlock(32, 64, kernel_size=3),
            CircularConvBlock(64, 64, kernel_size=5),   # wider receptive field
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # -----------------------------------------------------------
        # Block 3: long-range correlations
        # -----------------------------------------------------------
        self.block3 = nn.Sequential(
            CircularConvBlock(64, 128, kernel_size=3),
            CircularConvBlock(128, 128, kernel_size=3),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Flattened size: 128 * 4 * 4 = 2048
        flat_size = 128 * 4 * 4
        dense_in  = flat_size + n_physics_feats

        # -----------------------------------------------------------
        # Dense head (shared trunk)
        # -----------------------------------------------------------
        self.dense = nn.Sequential(
            nn.Linear(dense_in, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_rates[0]),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(p=dropout_rates[1]),
        )

        # -----------------------------------------------------------
        # Output heads
        # -----------------------------------------------------------
        self.phase_head = nn.Linear(64, 2)   # logits for {ferro, para}
        self.temp_head  = nn.Linear(64, 1)   # scalar temperature prediction

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        physics_feats: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x             : (B, 4, L, L) spin configuration tensor
        physics_feats : (B, n_physics_feats) global observables [m, chi, E, U4]
                        If None, a zero tensor is used (fallback).

        Returns
        -------
        phase_logits : (B, 2)  — raw logits for phase classification
        T_pred       : (B,)    — predicted temperature
        """
        # CNN backbone
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = h.flatten(start_dim=1)                 # (B, 2048)

        # Global physics feature injection
        if physics_feats is None:
            physics_feats = torch.zeros(
                x.size(0), self.n_physics_feats,
                device=x.device, dtype=x.dtype,
            )
        h = torch.cat([h, physics_feats], dim=1)   # (B, 2052)

        # Dense trunk
        h = self.dense(h)                          # (B, 64)

        # Dual output
        phase_logits = self.phase_head(h)          # (B, 2)
        T_pred       = self.temp_head(h).squeeze(1)# (B,)

        return phase_logits, T_pred

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def predict_phase(self, x: torch.Tensor, physics_feats=None) -> torch.Tensor:
        """Return predicted phase labels (0=ferro, 1=para) as int tensor."""
        logits, _ = self.forward(x, physics_feats)
        return logits.argmax(dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
