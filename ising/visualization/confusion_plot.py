"""
ising/visualization/confusion_plot.py

Visualization of the learning-by-confusion accuracy curve A(W).

The characteristic peak at W = Tc is the key diagnostic of the method.
Also plots MC Dropout entropy curve for comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


TC_ONSAGER_J2 = 4.5138


def plot_confusion_curve(
    W_values:   np.ndarray,
    accuracies: np.ndarray,
    Tc_pred:    float,
    Tc_std:     float = None,
    J:          float = 2.0,
    save_path:  Optional[str] = None,
) -> plt.Figure:
    """
    Plot the confusion accuracy curve A(W).

    Parameters
    ----------
    W_values   : candidate critical temperatures (x-axis)
    accuracies : validation accuracy at each W (y-axis)
    Tc_pred    : predicted Tc (peak location)
    Tc_std     : uncertainty on Tc_pred
    J          : coupling constant of the target system
    save_path  : optional save path
    """
    Tc_exact = (2.0 * J) / np.log(1.0 + np.sqrt(2.0))

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(W_values, accuracies, "o-", color="#2166ac",
            markersize=5, linewidth=2, label="Confusion accuracy $A(W)$")

    # Peak marker
    ax.axvline(Tc_pred, color="#d6604d", linewidth=2, linestyle="--",
               label=f"$T_c^{{pred}}$ = {Tc_pred:.4f}")

    if Tc_std is not None:
        ax.axvspan(Tc_pred - Tc_std, Tc_pred + Tc_std,
                   alpha=0.15, color="#d6604d", label=f"$\pm${Tc_std:.4f}")

    # Onsager reference
    ax.axvline(Tc_exact, color="#4dac26", linewidth=1.5, linestyle=":",
               label=f"Onsager $T_c$ = {Tc_exact:.4f}")

    ax.set_xlabel("Confusion point $W$", fontsize=12)
    ax.set_ylabel("Validation accuracy $A(W)$", fontsize=12)
    ax.set_title(f"Learning by Confusion — $J={J}$", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_entropy_curve(
    T_values:     np.ndarray,
    entropy_mean: np.ndarray,
    entropy_std:  np.ndarray = None,
    Tc_pred:      float = None,
    J:            float = 2.0,
    save_path:    Optional[str] = None,
) -> plt.Figure:
    """
    Plot MC Dropout predictive entropy H(T).

    Peak at Tc indicates maximum model uncertainty at the phase boundary.
    """
    Tc_exact = (2.0 * J) / np.log(1.0 + np.sqrt(2.0))

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(T_values, entropy_mean, "o-", color="#8073ac",
            markersize=5, linewidth=2, label="Predictive entropy $H[p](T)$")

    if entropy_std is not None:
        ax.fill_between(T_values,
                        entropy_mean - entropy_std,
                        entropy_mean + entropy_std,
                        alpha=0.2, color="#8073ac")

    if Tc_pred is not None:
        ax.axvline(Tc_pred, color="#d6604d", linewidth=2, linestyle="--",
                   label=f"$T_c^{{pred}}$ = {Tc_pred:.4f}")

    ax.axvline(Tc_exact, color="#4dac26", linewidth=1.5, linestyle=":",
               label=f"Onsager $T_c$ = {Tc_exact:.4f}")

    ax.set_xlabel("Temperature $T$", fontsize=12)
    ax.set_ylabel("Predictive entropy $H[p]$", fontsize=12)
    ax.set_title(f"MC Dropout Predictive Uncertainty — $J={J}$", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_transfer_summary(
    confusion_result: dict,
    entropy_result:   dict,
    J:                float = 2.0,
    save_path:        Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side: confusion curve (left) + entropy curve (right).
    Publication-ready 2-panel layout.
    """
    Tc_exact = (2.0 * J) / np.log(1.0 + np.sqrt(2.0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # --- Left: confusion ---
    W   = confusion_result["W_values"]
    acc = confusion_result["accuracies"]
    Tc_c= confusion_result["Tc_pred"]
    ax1.plot(W, acc, "o-", color="#2166ac", markersize=4, linewidth=2)
    ax1.axvline(Tc_c,    color="#d6604d", linewidth=2, linestyle="--",
                label=f"pred: {Tc_c:.4f}")
    ax1.axvline(Tc_exact,color="#4dac26", linewidth=1.5, linestyle=":",
                label=f"exact: {Tc_exact:.4f}")
    ax1.set_xlabel("Confusion point $W$", fontsize=11)
    ax1.set_ylabel("Accuracy $A(W)$", fontsize=11)
    ax1.set_title("(a) Learning by Confusion", fontsize=12)
    ax1.legend(fontsize=9); ax1.grid(alpha=0.2); ax1.set_ylim(0, 1.05)

    # --- Right: entropy ---
    T_v = entropy_result["T_values"]
    H_m = entropy_result["entropy_mean"]
    H_s = entropy_result.get("entropy_std")
    Tc_e= entropy_result["Tc_pred"]
    ax2.plot(T_v, H_m, "o-", color="#8073ac", markersize=4, linewidth=2)
    if H_s is not None:
        ax2.fill_between(T_v, H_m - H_s, H_m + H_s, alpha=0.2, color="#8073ac")
    ax2.axvline(Tc_e,    color="#d6604d", linewidth=2, linestyle="--",
                label=f"pred: {Tc_e:.4f}")
    ax2.axvline(Tc_exact,color="#4dac26", linewidth=1.5, linestyle=":",
                label=f"exact: {Tc_exact:.4f}")
    ax2.set_xlabel("Temperature $T$", fontsize=11)
    ax2.set_ylabel("Predictive entropy $H[p]$", fontsize=11)
    ax2.set_title("(b) MC Dropout Entropy", fontsize=12)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.2)

    fig.suptitle(f"Transfer Learning $T_c$ Prediction  ($J={J}$,  "
                 f"Onsager: {Tc_exact:.4f})", fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
