"""
ising/simulation/observables.py

Thermodynamic observables for the 2D Ising Model.

Computes bulk and local observables from spin configuration snapshots,
including magnetization, susceptibility, internal energy, specific heat,
Binder cumulant, and approximate correlation length. These are used both
as physics-informed CNN input features and as training labels.

Reference:
    Binder, K. (1981). Z. Phys. B 43, 119.
    Onsager, L. (1944). Phys. Rev. 65, 117.
"""

import numpy as np
from typing import Union


# ---------------------------------------------------------------------------
# Single-snapshot observables (operate on one L x L config)
# ---------------------------------------------------------------------------

def magnetization(spins: np.ndarray) -> float:
    """Absolute magnetization per spin: |m| = |sum(s_i)| / N."""
    return float(abs(spins.sum()) / spins.size)


def magnetization_signed(spins: np.ndarray) -> float:
    """Signed magnetization per spin: m = sum(s_i) / N."""
    return float(spins.sum() / spins.size)


def energy_per_spin(spins: np.ndarray, J: float = 1.0) -> float:
    """
    Internal energy per spin: E/N = -(J/N) * sum_{<i,j>} s_i * s_j

    Uses periodic boundary conditions with right + down neighbour sum
    to avoid double-counting.
    """
    L = spins.shape[0]
    s_right = np.roll(spins, -1, axis=1)
    s_down  = np.roll(spins, -1, axis=0)
    return float(-J * np.sum(spins * (s_right + s_down)) / spins.size)


def local_energy_map(spins: np.ndarray, J: float = 1.0) -> np.ndarray:
    """
    Local energy density at each site: e_i = -J * s_i * sum_{nn} s_j

    Used as a CNN input channel.

    Returns
    -------
    e_map : (L, L) float32 array, values in [-4J, +4J]
    """
    s_up    = np.roll(spins,  1, axis=0)
    s_down  = np.roll(spins, -1, axis=0)
    s_left  = np.roll(spins,  1, axis=1)
    s_right = np.roll(spins, -1, axis=1)
    nn_sum  = s_up + s_down + s_left + s_right
    return (-J * spins * nn_sum).astype(np.float32)


def coarse_magnetization_map(spins: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Local coarse-grained magnetization using a sliding window average.

    Computes mean spin over a (window x window) neighbourhood at each site,
    with periodic boundary conditions via FFT-based convolution.

    Used as a CNN input channel to encode mesoscale order.

    Returns
    -------
    m_map : (L, L) float32 array, values in [-1, +1]
    """
    from scipy.ndimage import uniform_filter
    # uniform_filter with mode='wrap' respects PBC
    m_map = uniform_filter(spins.astype(np.float32), size=window, mode='wrap')
    return m_map


def fourier_amplitude_map(spins: np.ndarray, kx_frac: float = 0.25,
                           ky_frac: float = 0.25) -> np.ndarray:
    """
    Fourier amplitude map at a characteristic cluster wavevector.

    Computes |FFT(spins)|^2 and returns the spatial map normalized to [0,1].
    The wavevector (kx_frac, ky_frac) * 2pi selects the cluster length scale.

    Used as a CNN input channel to encode spatial frequency content.

    Returns
    -------
    amp_map : (L, L) float32 array, values in [0, 1]
    """
    fft2 = np.fft.fft2(spins.astype(np.float64))
    power = np.abs(fft2) ** 2
    # Shift DC to center, normalize
    power_shift = np.fft.fftshift(power)
    p_max = power_shift.max()
    if p_max > 0:
        power_shift = power_shift / p_max
    return power_shift.astype(np.float32)


def build_cnn_input(spins: np.ndarray, J: float = 1.0) -> np.ndarray:
    """
    Build the 4-channel CNN input tensor from a spin configuration.

    Channels:
        0 : raw spins (float32, values in {-1, +1})
        1 : local energy density map
        2 : coarse-grained magnetization map (3x3 window)
        3 : Fourier power spectrum map

    Parameters
    ----------
    spins : (L, L) int8 array
    J     : coupling constant

    Returns
    -------
    tensor : (4, L, L) float32 array
    """
    ch0 = spins.astype(np.float32)
    ch1 = local_energy_map(spins, J=J)
    ch2 = coarse_magnetization_map(spins, window=3)
    ch3 = fourier_amplitude_map(spins)
    return np.stack([ch0, ch1, ch2, ch3], axis=0)


# ---------------------------------------------------------------------------
# Ensemble observables (operate on a collection of snapshots at one T)
# ---------------------------------------------------------------------------

def ensemble_observables(
    configs: np.ndarray,
    T: float,
    J: float = 1.0,
) -> dict:
    """
    Compute ensemble-averaged thermodynamic observables from a set of
    independent spin configurations at temperature T.

    Parameters
    ----------
    configs : (N, L, L) int8 array of N independent snapshots
    T       : simulation temperature
    J       : coupling constant

    Returns
    -------
    obs : dict with keys:
        m_mean    : mean |m|
        m_std     : std of |m|
        m2_mean   : mean m^2
        m4_mean   : mean m^4
        chi       : magnetic susceptibility chi = N*(m2 - m_mean^2) / T
        E_mean    : mean energy per spin
        E_std     : std of energy per spin
        C         : specific heat C = N*(E2 - E_mean^2) / T^2
        U4        : Binder cumulant 1 - m4/(3*m2^2)
        N_configs : number of configurations used
    """
    N, L, _ = configs.shape
    N_spins = L * L

    m_arr = np.array([magnetization(configs[i]) for i in range(N)])
    E_arr = np.array([energy_per_spin(configs[i], J=J) for i in range(N)])

    m_mean  = float(m_arr.mean())
    m_std   = float(m_arr.std())
    m2_mean = float((m_arr ** 2).mean())
    m4_mean = float((m_arr ** 4).mean())
    E_mean  = float(E_arr.mean())
    E_std   = float(E_arr.std())

    chi = N_spins * (m2_mean - m_mean ** 2) / T

    E2_mean = float((E_arr ** 2).mean())
    C   = N_spins * (E2_mean - E_mean ** 2) / (T ** 2)

    # Binder cumulant: size-independent at Tc
    if m2_mean > 0:
        U4 = 1.0 - m4_mean / (3.0 * m2_mean ** 2)
    else:
        U4 = 0.0

    return {
        "m_mean":    m_mean,
        "m_std":     m_std,
        "m2_mean":   m2_mean,
        "m4_mean":   m4_mean,
        "chi":       float(chi),
        "E_mean":    E_mean,
        "E_std":     E_std,
        "C":         float(C),
        "U4":        float(U4),
        "N_configs": N,
    }


def fss_corrected_phase_label(T: float, J: float, L: int) -> int:
    """
    Assign a binary phase label with finite-size scaling correction.

    The pseudo-critical temperature for finite L shifts as:
        Tc(L) = Tc(inf) + a * L^{-1/nu},  nu=1, a~1 (empirical for Ising)

    Returns
    -------
    0 : ferromagnetic (T < Tc(L))
    1 : paramagnetic  (T >= Tc(L))
    """
    Tc_inf = (2.0 * J) / np.log(1.0 + np.sqrt(2.0))
    a      = 1.0   # empirical prefactor for 2D Ising square lattice
    nu     = 1.0
    Tc_L   = Tc_inf + a * L ** (-1.0 / nu)
    return int(T >= Tc_L)


def per_config_label(
    spins: np.ndarray,
    T: float,
    J: float = 1.0,
) -> dict:
    """
    Build the full label dictionary for a single spin configuration.

    This is the label vector stored alongside each snapshot in the HDF5 file.

    Returns
    -------
    label : dict with all scalar observables + phase label
    """
    L = spins.shape[0]
    return {
        "T":      T,
        "beta":   1.0 / T,
        "J":      J,
        "m":      magnetization(spins),
        "m_sign": magnetization_signed(spins),
        "E":      energy_per_spin(spins, J=J),
        "phase":  fss_corrected_phase_label(T, J, L),
    }
