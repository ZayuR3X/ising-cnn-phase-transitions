"""
ising/visualization/gradcam.py

Grad-CAM visualization for IsingCNN.

Computes gradient-weighted class activation maps to show which spatial
regions of the spin configuration drive the phase prediction. Near Tc,
attention is expected to concentrate on domain walls and percolating
cluster boundaries — the geometrical hallmark of criticality.

Reference:
    Selvaraju et al. (2017). Grad-CAM. ICCV 2017.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional


# Custom colormap: transparent at 0, red at 1
_GCAM_CMAP = LinearSegmentedColormap.from_list(
    "gcam", [(0, "#00000000"), (0.4, "#ff990080"), (1.0, "#cc0000ff")]
)


class GradCAM:
    """
    Grad-CAM for IsingCNN.

    Hooks into the last conv layer (block3[-1] before pooling)
    and computes the gradient-weighted activation map.

    Parameters
    ----------
    model      : trained IsingCNN
    target_layer: name of the layer to hook (default: last conv in block3)
    """

    def __init__(self, model: torch.nn.Module, target_layer: str = "block3"):
        self.model        = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients   = None
        self._handles     = []
        self._register_hooks()

    def _register_hooks(self):
        # Hook the last CircularConvBlock in the target block
        block = dict(self.model.named_children())[self.target_layer]
        # Last conv block before AdaptiveAvgPool
        last_conv = None
        for layer in block:
            if hasattr(layer, "block"):  # CircularConvBlock
                last_conv = layer
        if last_conv is None:
            last_conv = block[-2]  # fallback

        def fwd_hook(module, inp, out):
            self._activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        h1 = last_conv.register_forward_hook(fwd_hook)
        h2 = last_conv.register_full_backward_hook(bwd_hook)
        self._handles = [h1, h2]

    def remove_hooks(self):
        for h in self._handles:
            h.remove()

    def compute(
        self,
        x:           torch.Tensor,
        physics:     torch.Tensor,
        target_class: int,
        device:      torch.device,
    ) -> np.ndarray:
        """
        Compute Grad-CAM for a single input.

        Parameters
        ----------
        x            : (1, 4, L, L) input tensor
        physics      : (1, 4) physics features
        target_class : 0 (ferro) or 1 (para) — class to explain
        device       : torch device

        Returns
        -------
        cam : (L, L) normalized attention map in [0, 1]
        """
        self.model.eval()
        x       = x.to(device).requires_grad_(False)
        physics = physics.to(device)

        # Forward
        logits, _ = self.model(x, physics)
        score     = logits[0, target_class]

        # Backward
        self.model.zero_grad()
        score.backward()

        # Grad-CAM: weight activations by global-averaged gradients
        grads = self._gradients        # (1, C, H', W')
        acts  = self._activations      # (1, C, H', W')

        weights = grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam     = F.relu(cam)

        # Upsample to input size
        L       = x.shape[-1]
        cam_up  = F.interpolate(cam, size=(L, L), mode="bilinear",
                                align_corners=False)
        cam_np  = cam_up[0, 0].cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)

        return cam_np


def plot_gradcam(
    model:        torch.nn.Module,
    configs:      np.ndarray,
    temperatures: np.ndarray,
    device:       torch.device,
    J:            float = 1.0,
    n_show:       int   = 5,
    save_path:    Optional[str] = None,
) -> plt.Figure:
    """
    Plot Grad-CAM overlaid on spin configurations at different temperatures.

    Parameters
    ----------
    model        : trained IsingCNN
    configs      : (N, L, L) int8 spin configurations
    temperatures : (N,) temperatures
    device       : torch device
    J            : coupling constant
    n_show       : number of temperature points to visualize
    save_path    : optional save path
    """
    from ising.simulation.observables import build_cnn_input

    SPIN_CMAP = plt.cm.gray

    gcam = GradCAM(model)

    # Select representative temperatures
    unique_T = np.unique(np.round(temperatures, 3))
    idxs     = np.linspace(0, len(unique_T) - 1, min(n_show, len(unique_T)), dtype=int)
    show_T   = unique_T[idxs]

    fig, axes = plt.subplots(2, len(show_T), figsize=(3 * len(show_T), 6))
    if len(show_T) == 1:
        axes = axes.reshape(2, 1)

    Tc = (2.0 * J) / np.log(1.0 + np.sqrt(2.0))

    for col, T in enumerate(show_T):
        mask  = np.where(np.abs(temperatures - T) < 1e-3)[0]
        spin  = configs[mask[0]]
        L     = spin.shape[0]

        tensor = torch.from_numpy(
            build_cnn_input(spin, J=J)
        ).unsqueeze(0).float()

        m   = abs(spin.sum()) / spin.size
        chi = (m ** 2) * (L ** 2) / T
        E   = -J * float(
            np.sum(spin * np.roll(spin,-1,axis=1) + spin * np.roll(spin,-1,axis=0))
        ) / spin.size
        phys = torch.tensor([[m, chi, E, 1.0/T]], dtype=torch.float32)

        target_cls = 0 if T < Tc else 1
        cam = gcam.compute(tensor, phys, target_cls, device)

        # Top row: raw spins
        ax_top = axes[0, col]
        ax_top.imshow(spin, cmap="gray", vmin=-1, vmax=1,
                      interpolation="nearest", aspect="equal")
        ax_top.set_title(f"T={T:.3f}", fontsize=9, pad=3)
        ax_top.axis("off")

        # Bottom row: spin + CAM overlay
        ax_bot = axes[1, col]
        ax_bot.imshow(spin, cmap="gray", vmin=-1, vmax=1,
                      interpolation="nearest", aspect="equal")
        ax_bot.imshow(cam, cmap=_GCAM_CMAP, alpha=0.65,
                      interpolation="bilinear", aspect="equal",
                      vmin=0, vmax=1)
        ax_bot.axis("off")

    axes[0, 0].set_ylabel("Spin config", fontsize=9, rotation=90, labelpad=5)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=9, rotation=90, labelpad=5)

    fig.suptitle("Grad-CAM: Spatial Attention at Different Temperatures",
                 fontsize=12, y=1.01)
    plt.tight_layout()

    gcam.remove_hooks()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
