"""
ising/transfer/mc_dropout.py

Predictive uncertainty estimation via MC Dropout for zero-shot Tc prediction.

With dropout active at inference time, K stochastic forward passes produce a
distribution over class probabilities. The predictive entropy H[p] peaks at
the critical temperature where the model is maximally uncertain — exactly
where phase identity becomes ambiguous.

This method requires zero additional training: the frozen J=1 backbone is
applied directly to J=2 configurations.

Reference:
    Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation.
    ICML 2016.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
import h5py


def enable_dropout(model: nn.Module) -> None:
    """Set all Dropout layers to train mode (active at inference)."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    """
    Compute entropy of the mean predictive distribution.

    H[p] = -sum_c p_bar_c * log(p_bar_c)

    Parameters
    ----------
    probs : (N, K, C) array — N samples, K MC passes, C classes

    Returns
    -------
    entropy : (N,) array
    """
    p_bar = probs.mean(axis=1)                         # (N, C)
    p_bar = np.clip(p_bar, 1e-10, 1.0)
    return -np.sum(p_bar * np.log(p_bar), axis=1)      # (N,)


def mc_dropout_inference(
    model:      nn.Module,
    h5_path:    str,
    n_passes:   int   = 100,
    batch_size: int   = 64,
    device:     torch.device = None,
    verbose:    bool  = True,
) -> dict:
    """
    Run MC Dropout inference on a J=2 dataset and compute predictive entropy.

    Parameters
    ----------
    model      : trained IsingCNN (used frozen except dropout is active)
    h5_path    : path to J=2 HDF5 file
    n_passes   : number of stochastic forward passes per sample
    batch_size : batch size for inference
    device     : torch device
    verbose    : print progress

    Returns
    -------
    dict with keys:
        "T_values"       : (N,) unique temperatures sorted
        "entropy_mean"   : (n_T,) mean entropy per temperature bin
        "entropy_std"    : (n_T,) std of entropy per temperature bin
        "Tc_pred"        : temperature of maximum mean entropy
        "probs_all"      : (N, K, 2) raw MC probabilities
        "entropy_all"    : (N,) per-sample entropy
        "T_all"          : (N,) per-sample temperature
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    with h5py.File(h5_path, "r") as hf:
        cnn_np  = hf["cnn_input"][:].astype(np.float32)
        T_np    = hf["temperature"][:].astype(np.float32)
        m_np    = hf["magnetization"][:].astype(np.float32)
        E_np    = hf["energy"][:].astype(np.float32)

    N = len(T_np)
    L = cnn_np.shape[-1]
    chi_np  = (m_np ** 2) * (L ** 2) / T_np
    beta_np = 1.0 / T_np
    phys_np = np.stack([m_np, chi_np, E_np, beta_np], axis=1).astype(np.float32)

    cnn_t  = torch.from_numpy(cnn_np)
    phys_t = torch.from_numpy(phys_np)

    # Set model: eval mode (BN frozen) but dropout active
    model.to(device)
    model.eval()
    enable_dropout(model)

    probs_all = np.zeros((N, n_passes, 2), dtype=np.float32)

    if verbose:
        print(f"MC Dropout inference: N={N}, K={n_passes} passes")

    with torch.no_grad():
        for k in range(n_passes):
            preds_k = []
            for start in range(0, N, batch_size):
                end   = min(start + batch_size, N)
                x_b   = cnn_t[start:end].to(device)
                ph_b  = phys_t[start:end].to(device)
                logits, _ = model(x_b, ph_b)
                p = torch.softmax(logits, dim=1).cpu().numpy()
                preds_k.append(p)
            probs_all[:, k, :] = np.concatenate(preds_k, axis=0)

            if verbose and (k % 25 == 0 or k == n_passes - 1):
                print(f"  pass {k+1:3d}/{n_passes}", flush=True)

    # Per-sample entropy
    entropy_all = predictive_entropy(probs_all)   # (N,)

    # Aggregate by temperature bin
    unique_T = np.unique(np.round(T_np, 4))
    entropy_mean = np.zeros(len(unique_T))
    entropy_std  = np.zeros(len(unique_T))

    for i, T in enumerate(unique_T):
        mask = np.abs(T_np - T) < 1e-3
        entropy_mean[i] = entropy_all[mask].mean()
        entropy_std[i]  = entropy_all[mask].std()

    peak_idx = int(np.argmax(entropy_mean))
    Tc_pred  = float(unique_T[peak_idx])

    if verbose:
        print(f"\nTc prediction (max entropy) : {Tc_pred:.4f}")
        print(f"Onsager exact               : {(4.0 / np.log(1.0 + np.sqrt(2.0))):.4f}")

    # Restore model to pure eval (dropout off)
    model.eval()

    return {
        "T_values":     unique_T,
        "entropy_mean": entropy_mean,
        "entropy_std":  entropy_std,
        "Tc_pred":      Tc_pred,
        "probs_all":    probs_all,
        "entropy_all":  entropy_all,
        "T_all":        T_np,
    }
