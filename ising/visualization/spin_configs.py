"""
ising/visualization/spin_configs.py

Visualization of 2D Ising spin configurations.

Generates publication-quality figures of spin snapshots at different
temperatures, domain structure plots, and CNN input channel maps.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from pathlib import Path
from typing import List, Optional
import h5py


# Consistent colormap: white = +1 (up), black = -1 (down)
SPIN_CMAP = ListedColormap(["#1a1a2e", "#e8e8f0"])


def plot_snapshots(
    configs:     np.ndarray,
    temperatures: np.ndarray,
    n_show:      int = 5,
    title:       str = "Spin Configurations",
    save_path:   Optional[str] = None,
) -> plt.Figure:
    """
    Plot a row of spin configuration snapshots at different temperatures.

    Parameters
    ----------
    configs      : (N, L, L) int8 array
    temperatures : (N,) temperature for each config
    n_show       : number of snapshots to display
    title        : figure title
    save_path    : if given, save figure to this path

    Returns
    -------
    fig : matplotlib Figure
    """
    # Pick one config per temperature (evenly spaced across T range)
    unique_T = np.unique(np.round(temperatures, 3))
    idxs     = np.linspace(0, len(unique_T) - 1, min(n_show, len(unique_T)), dtype=int)
    show_T   = unique_T[idxs]

    selected = []
    for T in show_T:
        mask = np.where(np.abs(temperatures - T) < 1e-3)[0]
        selected.append((T, configs[mask[0]]))

    fig, axes = plt.subplots(1, len(selected), figsize=(3 * len(selected), 3.2))
    if len(selected) == 1:
        axes = [axes]

    for ax, (T, spin) in zip(axes, selected):
        ax.imshow(spin, cmap=SPIN_CMAP, vmin=-1, vmax=1,
                  interpolation="nearest", aspect="equal")
        ax.set_title(f"T = {T:.3f}", fontsize=10, pad=4)
        ax.axis("off")

    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_cnn_channels(
    config:    np.ndarray,
    J:         float = 1.0,
    T:         float = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot all 4 CNN input channels for a single spin configuration.

    Channels:
        0: raw spins
        1: local energy density
        2: coarse magnetization (3x3)
        3: Fourier power spectrum

    Parameters
    ----------
    config    : (L, L) int8 spin array
    J         : coupling constant
    T         : temperature label for title
    save_path : optional save path
    """
    from ising.simulation.observables import build_cnn_input

    tensor = build_cnn_input(config, J=J)   # (4, L, L)

    titles = [
        "Raw spins $s_i$",
        r"Local energy $-Js_i\sum_{nn}s_j$",
        r"Coarse $\langle s \rangle_{3\times3}$",
        r"Fourier $|S(\mathbf{k})|^2$",
    ]
    cmaps = [SPIN_CMAP, "RdBu_r", "RdBu_r", "inferno"]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    for ax, ch, title, cmap in zip(axes, tensor, titles, cmaps):
        im = ax.imshow(ch, cmap=cmap, interpolation="nearest", aspect="equal")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    sup = f"CNN Input Channels" + (f"  (T = {T:.3f})" if T else "")
    fig.suptitle(sup, fontsize=11, y=1.03)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_phase_gallery(
    h5_path:   str,
    n_per_row: int = 4,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Gallery: ordered low-T configs on left, critical in centre, disordered on right.

    Parameters
    ----------
    h5_path    : path to HDF5 dataset file
    n_per_row  : configs per temperature group
    save_path  : optional save path
    """
    with h5py.File(h5_path, "r") as hf:
        T_all = hf["temperature"][:].astype(np.float32)
        spins = hf["spins"]

        unique_T = np.unique(np.round(T_all, 3))
        # Pick 3 representative temperatures: min, middle (~Tc), max
        T_lo  = unique_T[0]
        T_mid = unique_T[np.argmin(np.abs(unique_T - 2.269))]
        T_hi  = unique_T[-1]

        rows = {}
        for T_target, label in [(T_lo, "Ferromagnetic"), (T_mid, "Critical"), (T_hi, "Paramagnetic")]:
            mask = np.where(np.abs(T_all - T_target) < 1e-3)[0][:n_per_row]
            rows[label] = (T_target, spins[mask].copy())

    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_per_row, figsize=(2.5 * n_per_row, 2.8 * n_rows))

    for row_i, (label, (T, configs)) in enumerate(rows.items()):
        for col_i in range(n_per_row):
            ax = axes[row_i, col_i]
            if col_i < len(configs):
                ax.imshow(configs[col_i], cmap=SPIN_CMAP, vmin=-1, vmax=1,
                          interpolation="nearest", aspect="equal")
            ax.axis("off")
            if col_i == 0:
                ax.set_ylabel(f"{label}\nT={T:.3f}", fontsize=9, rotation=0,
                              labelpad=70, va="center")

    fig.suptitle("Phase Gallery: Ferromagnetic | Critical | Paramagnetic",
                 fontsize=12, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
