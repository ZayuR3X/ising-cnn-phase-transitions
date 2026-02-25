"""
ising/transfer/fss.py

Finite-size scaling (FSS) extrapolation of Tc estimates.

For a finite lattice of linear size L, the pseudo-critical temperature
shifts as:

    Tc(L) = Tc(inf) + a * L^{-1/nu}

with nu = 1 (exact for 2D Ising). Given Tc estimates from multiple
lattice sizes L, this module fits the FSS ansatz via least-squares
to extrapolate Tc(inf).

Also implements Binder cumulant crossing detection as a cross-validation
reference: U4(T) curves for different L all intersect at Tc(inf),
independent of system size.

Reference:
    Binder, K. (1981). Z. Phys. B 43, 119.
    Carrasquilla & Melko (2017). Nature Physics 13, 431.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# FSS ansatz fit
# ---------------------------------------------------------------------------

def fss_ansatz(L: float, Tc_inf: float, a: float, nu: float = 1.0) -> float:
    """Tc(L) = Tc_inf + a * L^{-1/nu}"""
    return Tc_inf + a * L ** (-1.0 / nu)


def extrapolate_Tc(
    L_values:  np.ndarray,
    Tc_values: np.ndarray,
    Tc_errors: np.ndarray = None,
    nu:        float = 1.0,
    verbose:   bool  = True,
) -> dict:
    """
    Fit the FSS ansatz Tc(L) = Tc_inf + a * L^{-1/nu} to extrapolate Tc(inf).

    Parameters
    ----------
    L_values  : array of lattice sizes [L1, L2, ...]
    Tc_values : array of Tc estimates at each L
    Tc_errors : optional array of uncertainties on Tc_values
    nu        : critical exponent (fixed at 1.0 for 2D Ising)
    verbose   : print results

    Returns
    -------
    dict with keys:
        "Tc_inf"    : extrapolated critical temperature
        "Tc_inf_err": uncertainty on Tc_inf
        "a"         : FSS prefactor
        "nu_fixed"  : nu value used
        "chi2"      : reduced chi-squared of the fit
        "Tc_fit"    : fitted Tc(L) values for the input L_values
    """
    L_arr  = np.asarray(L_values,  dtype=float)
    Tc_arr = np.asarray(Tc_values, dtype=float)
    sigma  = np.asarray(Tc_errors, dtype=float) if Tc_errors is not None else None

    # Fix nu, fit (Tc_inf, a)
    def model(L, Tc_inf, a):
        return fss_ansatz(L, Tc_inf, a, nu=nu)

    # Initial guess: Tc_inf from smallest finite-size correction
    p0 = [Tc_arr.min(), 1.0]

    try:
        popt, pcov = curve_fit(
            model, L_arr, Tc_arr,
            p0=p0, sigma=sigma, absolute_sigma=(sigma is not None),
        )
        perr = np.sqrt(np.diag(pcov))
    except RuntimeError:
        # Fallback: linear extrapolation in L^{-1/nu}
        x    = L_arr ** (-1.0 / nu)
        coeffs = np.polyfit(x, Tc_arr, deg=1)
        popt = [float(coeffs[1]), float(coeffs[0])]
        perr = [0.0, 0.0]

    Tc_inf, a = popt
    Tc_inf_err = perr[0]

    # Reduced chi-squared
    Tc_fit = model(L_arr, *popt)
    residuals = Tc_arr - Tc_fit
    dof = max(1, len(L_arr) - 2)
    if sigma is not None:
        chi2 = float(np.sum((residuals / sigma) ** 2) / dof)
    else:
        chi2 = float(np.sum(residuals ** 2) / dof)

    if verbose:
        print(f"\nFSS Extrapolation (nu={nu} fixed):")
        print(f"  {'L':>5}  {'Tc(L)':>8}  {'Tc_fit':>8}  {'residual':>10}")
        for Li, Tci, Tfi in zip(L_arr, Tc_arr, Tc_fit):
            print(f"  {int(Li):5d}  {Tci:8.4f}  {Tfi:8.4f}  {Tci-Tfi:+10.4f}")
        print(f"\n  Tc(inf) = {Tc_inf:.4f}  +/- {Tc_inf_err:.4f}")
        print(f"  a       = {a:.4f}")
        print(f"  chi2_r  = {chi2:.4f}")
        Tc_onsager = 4.0 / np.log(1.0 + np.sqrt(2.0))
        print(f"  Onsager = {Tc_onsager:.4f}  (error: {abs(Tc_inf - Tc_onsager):.4f})")

    return {
        "Tc_inf":     float(Tc_inf),
        "Tc_inf_err": float(Tc_inf_err),
        "a":          float(a),
        "nu_fixed":   nu,
        "chi2":       chi2,
        "Tc_fit":     Tc_fit,
    }


# ---------------------------------------------------------------------------
# Binder cumulant crossing (physics cross-validation)
# ---------------------------------------------------------------------------

def binder_crossing(
    T_grid:     np.ndarray,
    U4_by_size: Dict[int, np.ndarray],
    verbose:    bool = True,
) -> dict:
    """
    Estimate Tc from the crossing of Binder cumulant curves U4(T) for
    different lattice sizes. The crossing point is size-independent at Tc.

    Parameters
    ----------
    T_grid     : 1D array of temperatures (same for all sizes)
    U4_by_size : dict {L: U4_array} where U4_array[i] = U4 at T_grid[i]
    verbose    : print results

    Returns
    -------
    dict with:
        "Tc_crossings" : dict {(L1,L2): Tc_cross}
        "Tc_mean"      : mean of all pairwise crossings
        "Tc_std"       : std of all pairwise crossings
    """
    sizes    = sorted(U4_by_size.keys())
    crossings = {}

    for i in range(len(sizes)):
        for j in range(i + 1, len(sizes)):
            L1, L2 = sizes[i], sizes[j]
            U1 = U4_by_size[L1]
            U2 = U4_by_size[L2]

            diff = U1 - U2
            # Find zero crossing via linear interpolation
            sign_changes = np.where(np.diff(np.sign(diff)))[0]

            if len(sign_changes) == 0:
                continue

            # Take the crossing closest to expected Tc
            best_idx = sign_changes[0]
            T_lo = T_grid[best_idx]
            T_hi = T_grid[best_idx + 1]
            d_lo = diff[best_idx]
            d_hi = diff[best_idx + 1]

            if abs(d_hi - d_lo) > 1e-10:
                T_cross = T_lo - d_lo * (T_hi - T_lo) / (d_hi - d_lo)
            else:
                T_cross = (T_lo + T_hi) / 2.0

            crossings[(L1, L2)] = float(T_cross)

    if not crossings:
        return {"Tc_crossings": {}, "Tc_mean": float("nan"), "Tc_std": float("nan")}

    Tc_vals = list(crossings.values())
    Tc_mean = float(np.mean(Tc_vals))
    Tc_std  = float(np.std(Tc_vals))

    if verbose:
        print("\nBinder Cumulant Crossings:")
        for (L1, L2), Tc in crossings.items():
            print(f"  L={L1} x L={L2}: Tc = {Tc:.4f}")
        print(f"  Mean Tc = {Tc_mean:.4f}  +/- {Tc_std:.4f}")

    return {
        "Tc_crossings": crossings,
        "Tc_mean":      Tc_mean,
        "Tc_std":       Tc_std,
    }


# ---------------------------------------------------------------------------
# Summary: combine all Tc estimates
# ---------------------------------------------------------------------------

def summarize_Tc_predictions(
    fss_result:      dict,
    confusion_result: dict = None,
    entropy_result:   dict = None,
    binder_result:    dict = None,
    J:               float = 2.0,
) -> None:
    """Print a comparison table of all Tc prediction methods."""
    Tc_onsager = (2.0 * J) / np.log(1.0 + np.sqrt(2.0))

    print(f"\n{'='*60}")
    print(f"Critical Temperature Predictions  (J={J}, Onsager={Tc_onsager:.4f})")
    print(f"{'='*60}")
    print(f"  {'Method':<30}  {'Tc':>7}  {'Error':>7}")
    print(f"  {'-'*48}")

    def row(name, Tc, err=None):
        err_str = f"{abs(Tc - Tc_onsager):7.4f}" if not np.isnan(Tc) else "   N/A"
        unc_str = f" +/-{err:.4f}" if err is not None else ""
        print(f"  {name:<30}  {Tc:7.4f}{unc_str}  {err_str}")

    row("FSS extrapolation",
        fss_result["Tc_inf"], fss_result["Tc_inf_err"])

    if confusion_result:
        row("Learning by confusion",
            confusion_result["Tc_pred"], confusion_result.get("Tc_std"))

    if entropy_result:
        row("MC Dropout entropy",
            entropy_result["Tc_pred"])

    if binder_result and not np.isnan(binder_result["Tc_mean"]):
        row("Binder cumulant (physics)",
            binder_result["Tc_mean"], binder_result["Tc_std"])

    row("Onsager exact", Tc_onsager)
    print(f"{'='*60}")
