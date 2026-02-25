"""
ising/transfer/confusion.py

Learning-by-confusion method for critical temperature prediction.

The core idea (van Nieuwenburg et al., 2017): train a binary classifier
with an artificially imposed "critical temperature" W. Sweep W across the
temperature axis. The classifier achieves maximum accuracy when W = Tc,
because at the true critical temperature the data is maximally confusable
from both sides.

Protocol:
  1. Freeze the CNN backbone (all convolutional layers from J=1 training)
  2. For each candidate W in linspace(W_min, W_max, n_steps):
       a. Re-label all J=2 configs: phase=0 if T<W, phase=1 if T>=W
       b. Train only the classification head for head_epochs epochs
       c. Record validation accuracy A(W)
  3. T_c estimate = argmax A(W)

Reference:
    van Nieuwenburg, E.P.L., Liu, Y.H., & Huber, S.D. (2017).
    Learning phase transitions by confusion. Nature Physics 13, 435-439.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
import h5py


def _load_pilot_data(h5_path: str, device: torch.device) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray
]:
    """
    Load J=2 pilot dataset from HDF5.

    Returns
    -------
    cnn_inputs   : (N, 4, L, L) float32 tensor
    physics_feats: (N, 4)       float32 tensor
    T_values     : (N,)         float32 tensor
    T_np         : (N,)         numpy array (for re-labeling)
    """
    with h5py.File(h5_path, "r") as hf:
        cnn    = torch.tensor(hf["cnn_input"][:],     dtype=torch.float32)
        T_np   = hf["temperature"][:].astype(np.float32)
        m_np   = hf["magnetization"][:].astype(np.float32)
        E_np   = hf["energy"][:].astype(np.float32)

    T_vals = torch.tensor(T_np, dtype=torch.float32)

    # Build physics features [m, m^2*L^2/T, E, 1/T]
    L       = cnn.shape[-1]
    chi_np  = (m_np ** 2) * (L ** 2) / T_np
    beta_np = 1.0 / T_np
    phys    = torch.tensor(
        np.stack([m_np, chi_np, E_np, beta_np], axis=1),
        dtype=torch.float32,
    )

    return cnn.to(device), phys.to(device), T_vals.to(device), T_np


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the classification head."""
    for name, param in model.named_parameters():
        if "phase_head" not in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """Restore all parameters to trainable."""
    for param in model.parameters():
        param.requires_grad = True


def _reset_head(model: nn.Module, device: torch.device) -> None:
    """Re-initialize the phase_head weights for a fresh confusion sweep."""
    nn.init.xavier_uniform_(model.phase_head.weight)
    nn.init.zeros_(model.phase_head.bias)
    model.phase_head.to(device)


def confusion_sweep(
    model:       nn.Module,
    h5_path:     str,
    W_min:       float = 3.0,
    W_max:       float = 6.5,
    n_steps:     int   = 70,
    head_epochs: int   = 30,
    head_lr:     float = 1e-3,
    batch_size:  int   = 64,
    val_frac:    float = 0.2,
    device:      torch.device = None,
    verbose:     bool  = True,
) -> dict:
    """
    Run the learning-by-confusion W-sweep on a J=2 pilot dataset.

    Parameters
    ----------
    model       : trained IsingCNN (backbone will be frozen)
    h5_path     : path to J=2 pilot HDF5 file
    W_min/W_max : range of candidate critical temperatures to sweep
    n_steps     : number of W values evaluated
    head_epochs : epochs to train the head per W value
    head_lr     : learning rate for head retraining
    batch_size  : batch size for head training
    val_frac    : fraction of pilot data held out for accuracy measurement
    device      : torch device (auto-detected if None)
    verbose     : print progress

    Returns
    -------
    dict with keys:
        "W_values"   : (n_steps,) array of candidate temperatures
        "accuracies" : (n_steps,) validation accuracy for each W
        "Tc_pred"    : predicted critical temperature (argmax)
        "Tc_std"     : uncertainty from Gaussian fit to accuracy peak
        "peak_idx"   : index of the peak in W_values
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # Load pilot data
    cnn, phys, T_tensor, T_np = _load_pilot_data(h5_path, device)
    N = len(T_np)

    # Train/val split (fixed, not re-shuffled per W)
    rng     = np.random.default_rng(42)
    idx_all = rng.permutation(N)
    n_val   = max(1, int(val_frac * N))
    val_idx = idx_all[:n_val]
    tr_idx  = idx_all[n_val:]

    cnn_tr,  phys_tr,  T_tr  = cnn[tr_idx],  phys[tr_idx],  T_np[tr_idx]
    cnn_val, phys_val, T_val = cnn[val_idx], phys[val_idx], T_np[val_idx]

    W_values   = np.linspace(W_min, W_max, n_steps)
    accuracies = np.zeros(n_steps)

    # Freeze backbone
    freeze_backbone(model)

    if verbose:
        print(f"\nConfusion sweep: W in [{W_min:.2f}, {W_max:.2f}], {n_steps} steps")
        print(f"Pilot dataset: N={N}, train={len(tr_idx)}, val={len(val_idx)}")
        print(f"{'W':>7}  {'Acc':>6}")
        print("-" * 16)

    for step_i, W in enumerate(W_values):
        # Re-label with confusion boundary W
        labels_tr  = torch.tensor((T_tr  >= W).astype(np.int64), device=device)
        labels_val = torch.tensor((T_val >= W).astype(np.int64), device=device)

        # Reset head weights for each W
        _reset_head(model, device)

        # Train head only
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=head_lr,
        )
        criterion = nn.CrossEntropyLoss()

        ds_tr = TensorDataset(cnn_tr, phys_tr, labels_tr)
        loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)

        model.train()
        for _ in range(head_epochs):
            for x_b, ph_b, lb_b in loader:
                logits, _ = model(x_b, ph_b)
                loss = criterion(logits, lb_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate on val set
        model.eval()
        with torch.no_grad():
            logits_val, _ = model(cnn_val, phys_val)
            preds_val     = logits_val.argmax(dim=1)
            acc = (preds_val == labels_val).float().mean().item()

        accuracies[step_i] = acc

        if verbose and (step_i % 10 == 0 or step_i == n_steps - 1):
            print(f"  {W:5.3f}  {acc:6.4f}")

    # Restore all parameters
    unfreeze_all(model)

    # Find Tc from peak
    peak_idx = int(np.argmax(accuracies))
    Tc_pred  = float(W_values[peak_idx])

    # Uncertainty: fit Gaussian to top-5 points around peak
    lo  = max(0, peak_idx - 5)
    hi  = min(n_steps, peak_idx + 6)
    try:
        from scipy.optimize import curve_fit
        def gauss(x, mu, sig, a, b):
            return a * np.exp(-0.5 * ((x - mu) / sig) ** 2) + b
        p0 = [Tc_pred, (W_max - W_min) / 10, accuracies[peak_idx], 0.5]
        popt, pcov = curve_fit(gauss, W_values[lo:hi], accuracies[lo:hi], p0=p0)
        Tc_std = abs(float(popt[1]))
    except Exception:
        Tc_std = float(W_values[1] - W_values[0])  # fallback: step size

    if verbose:
        print(f"\nTc prediction : {Tc_pred:.4f}  +/- {Tc_std:.4f}")
        print(f"Onsager exact : {(4.0 / np.log(1.0 + np.sqrt(2.0))):.4f}")

    return {
        "W_values":   W_values,
        "accuracies": accuracies,
        "Tc_pred":    Tc_pred,
        "Tc_std":     Tc_std,
        "peak_idx":   peak_idx,
    }
