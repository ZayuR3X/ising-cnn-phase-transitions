"""
ising/simulation/autocorrelation.py

Integrated autocorrelation time (tau_int) estimation for Monte Carlo time series.

Uses the Madras-Sokal automatic windowing estimator, which finds the optimal
window W such that the bias-variance tradeoff in tau_int is minimized.
This is the standard method for determining independent-sample spacing in
Metropolis simulations near critical points.

Reference:
    Madras & Sokal (1988). J. Stat. Phys. 50, 109.
    Sokal, A.D. (1997). Functional integration: Basics and applications.
"""

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------

def normalized_autocorr(series: np.ndarray, max_lag: int = None) -> np.ndarray:
    """
    Compute the normalized autocorrelation function C(t) / C(0).

    Uses FFT-based computation for efficiency.

    Parameters
    ----------
    series  : 1D array of observable measurements (e.g. magnetization)
    max_lag : maximum lag to compute (default: len(series) // 2)

    Returns
    -------
    rho : 1D array of normalized autocorrelations, rho[0] = 1.0
    """
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    if max_lag is None:
        max_lag = n // 2

    # Subtract mean
    x = series - series.mean()

    # FFT-based autocorrelation (efficient for large n)
    # Zero-pad to next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < 2 * n:
        fft_size <<= 1

    f = np.fft.rfft(x, n=fft_size)
    acf_full = np.fft.irfft(f * np.conj(f))[:n]

    # Normalize by number of pairs and C(0)
    acf = acf_full[:max_lag + 1]
    counts = np.arange(n, n - max_lag - 1, -1, dtype=np.float64)
    acf = acf / counts

    if acf[0] == 0.0:
        return np.zeros(max_lag + 1)

    return acf / acf[0]


def tau_int_madras_sokal(
    series: np.ndarray,
    c_factor: float = 6.0,
    max_lag: int = None,
) -> Tuple[float, float, int]:
    """
    Estimate the integrated autocorrelation time using the Madras-Sokal
    automatic windowing algorithm.

    The window W is chosen as the smallest W such that W >= c_factor * tau_int(W),
    where tau_int(W) = 0.5 + sum_{t=1}^{W} rho(t).
    This balances statistical noise (grows with W) against truncation bias
    (decreases with W).

    Parameters
    ----------
    series   : 1D array of time-ordered observable measurements
    c_factor : window constant (Sokal recommends 6; higher = more conservative)
    max_lag  : maximum lag to consider

    Returns
    -------
    tau_int  : integrated autocorrelation time (in units of sweeps)
    tau_err  : statistical error on tau_int
    window_W : optimal window size chosen by the algorithm
    """
    series = np.asarray(series, dtype=np.float64)
    n = len(series)

    if max_lag is None:
        max_lag = n // 3  # conservative upper bound

    rho = normalized_autocorr(series, max_lag=max_lag)

    # Madras-Sokal windowing
    tau_running = 0.5
    window_W = 1

    for t in range(1, max_lag + 1):
        tau_running += rho[t]
        if t >= c_factor * tau_running:
            window_W = t
            break
    else:
        # Window never closed — series too short or too correlated
        window_W = max_lag

    tau_int = tau_running

    # Statistical error (Sokal 1997, eq. 3.11)
    # var(tau_int) ~ (2W + 1) * tau_int^2 / n
    tau_err = tau_int * np.sqrt((2 * window_W + 1) / n)

    return float(tau_int), float(tau_err), int(window_W)


# ---------------------------------------------------------------------------
# Convenience: measure tau_int from a live simulation
# ---------------------------------------------------------------------------

def measure_tau_int(
    ca,
    T: float,
    n_measure_sweeps: int = 5000,
    c_factor: float = 6.0,
    verbose: bool = False,
) -> dict:
    """
    Run the CA for n_measure_sweeps at temperature T and estimate tau_int
    for the magnetization time series.

    Parameters
    ----------
    ca               : thermalized IsingCA instance
    T                : simulation temperature
    n_measure_sweeps : number of sweeps to collect for estimation
    c_factor         : Madras-Sokal window factor
    verbose          : print result

    Returns
    -------
    dict with keys:
        "tau_int"     : integrated autocorrelation time
        "tau_err"     : statistical error
        "window_W"    : optimal window
        "spacing"     : recommended sweep spacing (ceil(2 * tau_int))
        "m_series"    : raw magnetization time series
    """
    beta = 1.0 / T
    m_series = np.empty(n_measure_sweeps)

    for k in range(n_measure_sweeps):
        ca.sweep(beta)
        m_series[k] = ca.magnetization()

    tau, tau_err, W = tau_int_madras_sokal(m_series, c_factor=c_factor)

    # Spacing: at least 2*tau_int, minimum 1
    spacing = max(1, int(np.ceil(2.0 * tau)))

    if verbose:
        print(
            f"  T={T:.3f}: tau_int={tau:.1f} +/- {tau_err:.1f}  "
            f"W={W}  spacing={spacing}"
        )

    return {
        "tau_int": tau,
        "tau_err": tau_err,
        "window_W": W,
        "spacing": spacing,
        "m_series": m_series,
    }


# ---------------------------------------------------------------------------
# Batch estimation across temperature grid
# ---------------------------------------------------------------------------

def estimate_spacing_grid(
    ca,
    temperatures: np.ndarray,
    n_measure_sweeps: int = 3000,
    default_spacing: int = 20,
    verbose: bool = True,
) -> np.ndarray:
    """
    Estimate recommended sample spacing for each temperature in a grid.

    Useful for pre-computing spacings before the full dataset generation run.
    The CA is re-thermalized at each temperature before measurement.

    Parameters
    ----------
    ca               : IsingCA instance
    temperatures     : 1D array of temperatures
    n_measure_sweeps : sweeps used for tau_int estimation per temperature
    default_spacing  : fallback if tau_int estimation fails
    verbose          : print progress

    Returns
    -------
    spacings : 1D int array of recommended spacings, same length as temperatures
    """
    from ising.simulation.thermalization import smart_init, thermalize

    spacings = np.full(len(temperatures), default_spacing, dtype=int)

    for idx, T in enumerate(temperatures):
        try:
            smart_init(ca, T)
            thermalize(ca, T, n_sweeps_default=5000, n_sweeps_critical=15000,
                       verbose=False)
            result = measure_tau_int(ca, T, n_measure_sweeps=n_measure_sweeps,
                                     verbose=verbose)
            spacings[idx] = result["spacing"]
        except Exception as e:
            if verbose:
                print(f"  T={T:.3f}: tau_int estimation failed ({e}), "
                      f"using default={default_spacing}")
            spacings[idx] = default_spacing

    return spacings
