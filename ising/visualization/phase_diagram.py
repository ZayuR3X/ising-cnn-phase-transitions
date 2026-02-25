"""
ising/visualization/phase_diagram.py

Thermodynamic observable plots for the 2D Ising Model.

Generates:
  - Magnetization |m|(T) curve with phase transition
  - Susceptibility chi(T) with peak at Tc
  - Specific heat C(T)
  - Binder cumulant U4(T) for multiple system sizes
  - Combined 4-panel summary figure
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from typing import Dict, Optional, List
import h5py


TC_ONSAGER = 2.2692
COLORS     = ["#2166ac", "#d6604d", "#4dac26", "#8073ac", "#f1a340"]


def _load_ensemble(h5_path: str) -> dict:
    """Load ensemble observables from HDF5 file."""
    with h5py.File(h5_path, "r") as hf:
        return {
            "T":   hf["ensemble/T"][:].astype(np.float64),
            "m":   hf["ensemble/m"][:].astype(np.float64),
            "chi": hf["ensemble/chi"][:].astype(np.float64),
            "C":   hf["ensemble/C"][:].astype(np.float64),
            "U4":  hf["ensemble/U4"][:].astype(np.float64),
            "L":   int(hf.attrs.get("L", 0)),
            "J":   float(hf.attrs.get("J", 1.0)),
        }


def plot_magnetization(
    h5_paths:  List[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot |m|(T) for one or more system sizes.

    Includes Onsager exact magnetization for reference:
        m_exact = (1 - sinh(2J/T)^{-4})^{1/8}  for T < Tc
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for i, path in enumerate(h5_paths):
        d  = _load_ensemble(path)
        ax.plot(d["T"], d["m"], "o-", color=COLORS[i % len(COLORS)],
                markersize=4, linewidth=1.5, label=f"L={d['L']}")

    # Onsager exact (J=1)
    T_exact = np.linspace(1.0, TC_ONSAGER - 0.01, 200)
    sinh_inv = 1.0 / np.sinh(2.0 / T_exact)
    m_exact  = np.clip(1.0 - sinh_inv ** 4, 0, None) ** 0.125
    ax.plot(T_exact, m_exact, "k--", linewidth=1.5, label="Onsager exact", zorder=5)

    ax.axvline(TC_ONSAGER, color="gray", linestyle=":", linewidth=1,
               label=f"$T_c$ = {TC_ONSAGER:.4f}")
    ax.set_xlabel("Temperature $T$", fontsize=12)
    ax.set_ylabel(r"Magnetization $|\langle m \rangle|$", fontsize=12)
    ax.set_title("Order Parameter vs Temperature", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(left=0.8)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_susceptibility(
    h5_paths:  List[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot magnetic susceptibility chi(T), peak indicates Tc."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for i, path in enumerate(h5_paths):
        d = _load_ensemble(path)
        ax.plot(d["T"], d["chi"], "o-", color=COLORS[i % len(COLORS)],
                markersize=4, linewidth=1.5, label=f"L={d['L']}")
        peak_T = d["T"][np.argmax(d["chi"])]
        ax.axvline(peak_T, color=COLORS[i % len(COLORS)],
                   linestyle="--", linewidth=0.8, alpha=0.6)

    ax.axvline(TC_ONSAGER, color="gray", linestyle=":", linewidth=1.2,
               label=f"$T_c^{{\\infty}}$ = {TC_ONSAGER:.4f}")
    ax.set_xlabel("Temperature $T$", fontsize=12)
    ax.set_ylabel(r"Susceptibility $\chi$", fontsize=12)
    ax.set_title("Magnetic Susceptibility vs Temperature", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_binder_cumulant(
    h5_paths:  List[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot Binder cumulant U4(T) for multiple system sizes.

    All curves cross at Tc(inf) — size-independent crossing point.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for i, path in enumerate(h5_paths):
        d = _load_ensemble(path)
        ax.plot(d["T"], d["U4"], "o-", color=COLORS[i % len(COLORS)],
                markersize=4, linewidth=1.5, label=f"L={d['L']}")

    ax.axhline(2.0 / 3.0, color="gray", linestyle="--", linewidth=1,
               label="$U_4 = 2/3$ (ordered limit)")
    ax.axvline(TC_ONSAGER, color="gray", linestyle=":", linewidth=1.2,
               label=f"$T_c$ = {TC_ONSAGER:.4f}")
    ax.set_xlabel("Temperature $T$", fontsize=12)
    ax.set_ylabel(r"Binder Cumulant $U_4$", fontsize=12)
    ax.set_title("Binder Cumulant — Crossing at $T_c$", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.1, 0.75)
    ax.grid(alpha=0.25)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_summary(
    h5_paths:  List[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    4-panel summary: |m|, chi, C, U4 — publication-ready layout.
    """
    fig = plt.figure(figsize=(12, 9))
    axes = [
        fig.add_subplot(2, 2, 1),
        fig.add_subplot(2, 2, 2),
        fig.add_subplot(2, 2, 3),
        fig.add_subplot(2, 2, 4),
    ]

    labels  = [r"$|\langle m \rangle|$", r"$\chi$",
               r"$C$", r"$U_4$"]
    keys    = ["m", "chi", "C", "U4"]

    for ax, label, key in zip(axes, labels, keys):
        for i, path in enumerate(h5_paths):
            d = _load_ensemble(path)
            ax.plot(d["T"], d[key], "o-", color=COLORS[i % len(COLORS)],
                    markersize=3, linewidth=1.3, label=f"L={d['L']}")
        ax.axvline(TC_ONSAGER, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("$T$", fontsize=11)
        ax.set_ylabel(label, fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    # Onsager exact on magnetization panel
    T_ex = np.linspace(1.0, TC_ONSAGER - 0.01, 200)
    m_ex = np.clip(1.0 - (1.0 / np.sinh(2.0 / T_ex)) ** 4, 0, None) ** 0.125
    axes[0].plot(T_ex, m_ex, "k--", linewidth=1.5, label="Onsager", zorder=5)
    axes[0].legend(fontsize=8)

    fig.suptitle("2D Ising Model: Thermodynamic Observables", fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
