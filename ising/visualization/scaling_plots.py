"""
ising/visualization/scaling_plots.py

Finite-size scaling plots and critical exponent visualization.

Generates:
  - Tc(L) vs L^{-1/nu} with FSS fit
  - Data collapse: m * L^{beta/nu} vs (T-Tc) * L^{1/nu}
  - Susceptibility scaling collapse
  - Critical exponent summary table
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import h5py


# 2D Ising exact critical exponents
EXPONENTS = {
    "beta":  1.0 / 8.0,
    "nu":    1.0,
    "gamma": 7.0 / 4.0,
    "eta":   1.0 / 4.0,
    "alpha": 0.0,       # logarithmic
}

COLORS = ["#2166ac", "#d6604d", "#4dac26", "#8073ac", "#f1a340"]


def plot_fss_fit(
    L_values:  np.ndarray,
    Tc_values: np.ndarray,
    Tc_errors: np.ndarray = None,
    fss_result: dict = None,
    save_path:  Optional[str] = None,
) -> plt.Figure:
    """
    Plot Tc(L) vs L^{-1/nu} with FSS extrapolation to L → inf.

    Parameters
    ----------
    L_values   : lattice sizes
    Tc_values  : Tc estimate at each L (from confusion sweep or chi peak)
    Tc_errors  : optional uncertainties
    fss_result : output of fss.extrapolate_Tc (for fit line)
    save_path  : optional save path
    """
    nu   = EXPONENTS["nu"]
    x    = np.array(L_values, dtype=float) ** (-1.0 / nu)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.errorbar(x, Tc_values, yerr=Tc_errors, fmt="o",
                color="#2166ac", markersize=7, capsize=4,
                linewidth=1.5, label="$T_c(L)$ estimates")

    # FSS fit line
    if fss_result is not None:
        Tc_inf = fss_result["Tc_inf"]
        a      = fss_result["a"]
        x_fit  = np.linspace(0, x.max() * 1.1, 200)
        y_fit  = Tc_inf + a * x_fit
        ax.plot(x_fit, y_fit, "--", color="#d6604d", linewidth=1.8,
                label=f"FSS fit: $T_c(\\infty)={Tc_inf:.4f}$")
        ax.axhline(Tc_inf, color="#d6604d", linestyle=":", linewidth=1,
                   alpha=0.6)

    ax.set_xlabel(r"$L^{-1/\nu}$  ($\nu=1$)", fontsize=12)
    ax.set_ylabel(r"$T_c(L)$", fontsize=12)
    ax.set_title(r"Finite-Size Scaling: $T_c(L) = T_c(\infty) + a\,L^{-1/\nu}$",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_magnetization_collapse(
    h5_paths:  List[str],
    Tc:        float = 2.2692,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Magnetization data collapse:
        m * L^{beta/nu}  vs  (T - Tc) * L^{1/nu}

    All curves should collapse onto a single universal function.
    """
    beta = EXPONENTS["beta"]
    nu   = EXPONENTS["nu"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for i, path in enumerate(h5_paths):
        with h5py.File(path, "r") as hf:
            T   = hf["ensemble/T"][:].astype(np.float64)
            m   = hf["ensemble/m"][:].astype(np.float64)
            L   = int(hf.attrs.get("L", 64))

        # Raw data
        axes[0].plot(T, m, "o-", color=COLORS[i % len(COLORS)],
                     markersize=3, linewidth=1.3, label=f"L={L}")

        # Scaled data
        x_scaled = (T - Tc) * (L ** (1.0 / nu))
        y_scaled = m * (L ** (beta / nu))
        axes[1].plot(x_scaled, y_scaled, "o", color=COLORS[i % len(COLORS)],
                     markersize=3, alpha=0.8, label=f"L={L}")

    axes[0].axvline(Tc, color="gray", linestyle=":", linewidth=1)
    axes[0].set_xlabel("$T$", fontsize=11)
    axes[0].set_ylabel(r"$|\langle m \rangle|$", fontsize=11)
    axes[0].set_title("Raw magnetization", fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    axes[1].axvline(0, color="gray", linestyle=":", linewidth=1)
    axes[1].set_xlabel(r"$(T - T_c)\,L^{1/\nu}$", fontsize=11)
    axes[1].set_ylabel(r"$|\langle m \rangle|\,L^{\beta/\nu}$", fontsize=11)
    axes[1].set_title(r"Scaling collapse  ($\beta=\frac{1}{8},\,\nu=1$)", fontsize=11)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)

    fig.suptitle("Magnetization Finite-Size Scaling Collapse", fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_exponent_summary(save_path: Optional[str] = None) -> plt.Figure:
    """
    Table figure summarizing 2D Ising critical exponents.
    Useful as a standalone reference panel in papers/posters.
    """
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")

    rows = [
        ["$\\beta$", "1/8", "0.125", "Magnetization: $m \\sim |t|^\\beta$"],
        ["$\\nu$",   "1",   "1.000", "Correlation length: $\\xi \\sim |t|^{-\\nu}$"],
        ["$\\gamma$","7/4", "1.750", "Susceptibility: $\\chi \\sim |t|^{-\\gamma}$"],
        ["$\\eta$",  "1/4", "0.250", "Corr. function: $G(r) \\sim r^{-(d-2+\\eta)}$"],
        ["$\\alpha$","0 (log)","—",  "Specific heat: $C \\sim \\ln|t|$"],
    ]
    cols = ["Exponent", "Exact (fraction)", "Decimal", "Observable"]

    table = ax.table(
        cellText=rows, colLabels=cols,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Header style
    for j in range(len(cols)):
        table[0, j].set_facecolor("#2166ac")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternating row shading
    for i in range(1, len(rows) + 1):
        for j in range(len(cols)):
            table[i, j].set_facecolor("#eaf0f8" if i % 2 == 0 else "white")

    ax.set_title("2D Ising Model: Critical Exponents", fontsize=12, pad=10)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
