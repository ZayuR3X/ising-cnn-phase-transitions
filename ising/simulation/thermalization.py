"""
ising/simulation/thermalization.py

Thermalization (burn-in) detection for the 2D Ising Model CA simulation.

Monitors the running magnetization time series and declares thermalization
when the system has converged to equilibrium. Near the critical temperature,
thermalization times diverge as tau ~ xi^z (z ~ 2.17), so extended burn-in
is applied automatically within the critical window.

Reference:
    Sokal, A.D. (1997). Monte Carlo methods in statistical mechanics.
    Madras & Sokal (1988). J. Stat. Phys. 50, 109.
"""

import numpy as np
from typing import Tuple
from ising.simulation.ca_ising import IsingCA


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TC_J1 = 2.2692   # Onsager Tc for J=1, used as reference for window checks


# ---------------------------------------------------------------------------
# Core thermalization routine
# ---------------------------------------------------------------------------

def thermalize(
    ca: IsingCA,
    T: float,
    n_sweeps_default: int = 10_000,
    n_sweeps_critical: int = 50_000,
    critical_window: float = 0.3,
    convergence_eps: float = 0.005,
    check_interval: int = 500,
    verbose: bool = False,
) -> dict:
    """
    Run burn-in sweeps until the system reaches thermal equilibrium.

    Strategy:
    - Run in blocks of `check_interval` sweeps.
    - After each block, compare the mean |m| of the last block vs the
      previous block. Declare convergence when the difference is below
      `convergence_eps` for two consecutive checks.
    - Hard cap at n_sweeps_default (or n_sweeps_critical near Tc).

    Parameters
    ----------
    ca               : IsingCA instance (modified in-place)
    T                : simulation temperature
    n_sweeps_default : maximum sweeps away from Tc
    n_sweeps_critical: maximum sweeps near Tc
    critical_window  : |T - Tc| threshold for extended thermalization
    convergence_eps  : convergence criterion on running |m|
    check_interval   : sweeps per convergence check block
    verbose          : print convergence progress

    Returns
    -------
    info : dict with keys
        "n_sweeps_run"   : total sweeps executed
        "converged"      : True if convergence criterion was met
        "final_m"        : |m| at end of thermalization
        "m_history"      : magnetization recorded at each check point
    """
    beta = 1.0 / T
    Tc_eff = (2.0 * ca.J) / np.log(1.0 + np.sqrt(2.0))

    near_critical = abs(T - Tc_eff) < critical_window
    n_max = n_sweeps_critical if near_critical else n_sweeps_default

    m_history = []
    n_sweeps_run = 0
    converged = False
    prev_mean_m = None
    n_converged_checks = 0  # require 2 consecutive passing checks

    while n_sweeps_run < n_max:
        # Run one block
        block_m = []
        for _ in range(check_interval):
            ca.sweep(beta)
            block_m.append(ca.magnetization())
        n_sweeps_run += check_interval

        current_mean_m = float(np.mean(block_m))
        m_history.append(current_mean_m)

        if verbose:
            print(
                f"  sweep {n_sweeps_run:6d} / {n_max}  "
                f"|m| = {current_mean_m:.4f}",
                end=""
            )

        # Convergence check
        if prev_mean_m is not None:
            delta = abs(current_mean_m - prev_mean_m)
            if verbose:
                print(f"  delta = {delta:.5f}", end="")
            if delta < convergence_eps:
                n_converged_checks += 1
                if n_converged_checks >= 2:
                    converged = True
                    if verbose:
                        print("  --> CONVERGED")
                    break
            else:
                n_converged_checks = 0

        if verbose:
            print()

        prev_mean_m = current_mean_m

    return {
        "n_sweeps_run": n_sweeps_run,
        "converged": converged,
        "final_m": ca.magnetization(),
        "m_history": np.array(m_history),
    }


# ---------------------------------------------------------------------------
# Smart initialization strategy
# ---------------------------------------------------------------------------

def smart_init(ca: IsingCA, T: float) -> None:
    """
    Choose hot vs cold start based on temperature relative to Tc.

    - T > Tc : hot start (random) — avoids metastable ordered state
    - T < Tc : cold start (ordered) — avoids getting stuck in disordered state
    - T ~ Tc : hot start (critical fluctuations need full exploration)

    Parameters
    ----------
    ca : IsingCA instance
    T  : simulation temperature
    """
    Tc_eff = (2.0 * ca.J) / np.log(1.0 + np.sqrt(2.0))

    if T >= Tc_eff - 0.1:
        ca.init_random()
    else:
        # Cold start with random sign to avoid systematic bias
        sign = 1 if ca.rng.random() > 0.5 else -1
        ca.init_ordered(sign=sign)


# ---------------------------------------------------------------------------
# Batch thermalization (used by generator.py)
# ---------------------------------------------------------------------------

def thermalize_temperature_series(
    ca: IsingCA,
    temperatures: np.ndarray,
    cfg: dict,
    verbose: bool = False,
) -> list:
    """
    Thermalize the CA across a series of temperatures.

    For each temperature, re-initializes the spin configuration using
    smart_init and runs thermalization. Returns a list of info dicts.

    Parameters
    ----------
    ca           : IsingCA instance
    temperatures : 1D array of temperatures (ascending or descending)
    cfg          : dict from simulation.yaml mc section
    verbose      : print progress

    Returns
    -------
    infos : list of thermalization info dicts, one per temperature
    """
    infos = []
    n = len(temperatures)

    for idx, T in enumerate(temperatures):
        if verbose:
            print(f"[{idx+1}/{n}] Thermalizing at T={T:.3f} ...", flush=True)

        smart_init(ca, T)

        info = thermalize(
            ca,
            T=T,
            n_sweeps_default=cfg.get("n_thermalize", 10_000),
            n_sweeps_critical=cfg.get("n_thermalize_critical", 50_000),
            critical_window=cfg.get("critical_window", 0.3),
            verbose=verbose,
        )
        infos.append(info)

        if verbose:
            status = "converged" if info["converged"] else "max sweeps reached"
            print(
                f"  -> {status} after {info['n_sweeps_run']} sweeps, "
                f"|m|={info['final_m']:.4f}"
            )

    return infos
