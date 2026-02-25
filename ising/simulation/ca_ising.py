"""
ising/simulation/ca_ising.py

Metropolis-Hastings Cellular Automaton for the 2D Ising Model.

Implements a checkerboard (bipartite sublattice) update schedule that satisfies
detailed balance while maintaining a CA-compatible structure. Periodic boundary
conditions are enforced throughout.

Reference:
    Tanaka & Tomiya (2017), J. Phys. Soc. Jpn. 86, 063001
    Metropolis et al. (1953), J. Chem. Phys. 21, 1087
"""

import numpy as np
from numba import njit, prange
from typing import Tuple


# ---------------------------------------------------------------------------
# Low-level JIT-compiled kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _metropolis_sweep_sublattice(
    spins: np.ndarray,
    L: int,
    beta: float,
    J: float,
    sublattice: int,
    rng_flat: np.ndarray,
    rng_accept: np.ndarray,
) -> int:
    """
    One checkerboard sublattice sweep over L*L/2 candidate spins.

    Parameters
    ----------
    spins       : (L, L) int8 array of spin values in {-1, +1}
    L           : lattice linear size
    beta        : inverse temperature 1 / (k_B * T)
    J           : exchange coupling constant
    sublattice  : 0 = A sublattice (i+j even), 1 = B sublattice (i+j odd)
    rng_flat    : pre-drawn uniform [0,1) floats, length L*L//2 (site selection)
    rng_accept  : pre-drawn uniform [0,1) floats, length L*L//2 (acceptance)

    Returns
    -------
    n_accepted : number of accepted spin flips
    """
    n_accepted = 0
    idx = 0

    for i in range(L):
        for j in range(L):
            if (i + j) % 2 != sublattice:
                continue

            # Periodic boundary neighbours
            s_up    = spins[(i - 1) % L, j]
            s_down  = spins[(i + 1) % L, j]
            s_left  = spins[i, (j - 1) % L]
            s_right = spins[i, (j + 1) % L]

            nn_sum  = s_up + s_down + s_left + s_right
            s_i     = spins[i, j]
            delta_E = 2.0 * J * s_i * nn_sum

            # Metropolis acceptance criterion
            if delta_E <= 0.0 or rng_accept[idx] < np.exp(-beta * delta_E):
                spins[i, j] = -s_i
                n_accepted += 1

            idx += 1

    return n_accepted


@njit(cache=True)
def _compute_energy(spins: np.ndarray, L: int, J: float) -> float:
    """Total energy per spin: E/N = -J * <s_i s_j>."""
    energy = 0.0
    for i in range(L):
        for j in range(L):
            s_right = spins[i, (j + 1) % L]
            s_down  = spins[(i + 1) % L, j]
            energy -= J * spins[i, j] * (s_right + s_down)
    return energy / (L * L)


@njit(cache=True)
def _compute_magnetization(spins: np.ndarray, L: int) -> float:
    """Absolute magnetization per spin: |m| = |sum(s_i)| / N."""
    total = 0.0
    for i in range(L):
        for j in range(L):
            total += spins[i, j]
    return abs(total) / (L * L)


# ---------------------------------------------------------------------------
# Public simulation class
# ---------------------------------------------------------------------------

class IsingCA:
    """
    2D Ising Model Metropolis-Hastings Cellular Automaton.

    Parameters
    ----------
    L : int
        Lattice linear size. Total spins = L * L.
    J : float
        Exchange coupling constant. J > 0 is ferromagnetic.
    seed : int, optional
        NumPy random seed for reproducibility.

    Examples
    --------
    >>> ca = IsingCA(L=64, J=1.0, seed=42)
    >>> ca.init_random()
    >>> ca.run(T=2.0, n_sweeps=1000)
    >>> snapshot = ca.spins.copy()
    """

    def __init__(self, L: int, J: float = 1.0, seed: int = 42) -> None:
        self.L    = L
        self.J    = J
        self.rng  = np.random.default_rng(seed)
        self.spins: np.ndarray = np.ones((L, L), dtype=np.int8)

        self._n_half = (L * L) // 2  # spins per sublattice

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_random(self) -> None:
        """Hot start: random spin configuration (T -> infinity)."""
        self.spins = self.rng.choice(
            np.array([-1, 1], dtype=np.int8),
            size=(self.L, self.L),
        )

    def init_ordered(self, sign: int = 1) -> None:
        """Cold start: all spins aligned (T -> 0)."""
        self.spins = np.full((self.L, self.L), sign, dtype=np.int8)

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def sweep(self, beta: float) -> Tuple[int, int]:
        """
        One full Monte Carlo sweep = one A-sublattice + one B-sublattice pass.

        Draws fresh random numbers for each half-sweep so that the Numba
        kernel remains stateless and cache-friendly.

        Returns
        -------
        (n_acc_A, n_acc_B) : accepted flips per sublattice
        """
        rng_accept_A = self.rng.random(self._n_half)
        rng_flat_A   = self.rng.random(self._n_half)  # reserved for future use
        n_A = _metropolis_sweep_sublattice(
            self.spins, self.L, beta, self.J, 0, rng_flat_A, rng_accept_A
        )

        rng_accept_B = self.rng.random(self._n_half)
        rng_flat_B   = self.rng.random(self._n_half)
        n_B = _metropolis_sweep_sublattice(
            self.spins, self.L, beta, self.J, 1, rng_flat_B, rng_accept_B
        )

        return n_A, n_B

    def run(self, T: float, n_sweeps: int) -> None:
        """
        Run n_sweeps Monte Carlo sweeps at temperature T.
        Modifies self.spins in-place.

        Parameters
        ----------
        T        : temperature in units where k_B = 1
        n_sweeps : number of full sweeps (each sweep = L*L flip attempts)
        """
        beta = 1.0 / T
        for _ in range(n_sweeps):
            self.sweep(beta)

    # ------------------------------------------------------------------
    # Observables (fast, in-place)
    # ------------------------------------------------------------------

    def energy(self) -> float:
        """Internal energy per spin."""
        return _compute_energy(self.spins, self.L, self.J)

    def magnetization(self) -> float:
        """Absolute magnetization per spin |m|."""
        return _compute_magnetization(self.spins, self.L)

    def snapshot(self) -> np.ndarray:
        """Return a copy of the current spin configuration."""
        return self.spins.copy()

    # ------------------------------------------------------------------
    # Sampling loop (used by generator.py)
    # ------------------------------------------------------------------

    def sample(
        self,
        T: float,
        n_samples: int,
        spacing: int,
        n_thermalize: int = 0,
    ) -> np.ndarray:
        """
        Collect n_samples independent spin configurations at temperature T.

        Parameters
        ----------
        T            : simulation temperature
        n_samples    : number of snapshots to collect
        spacing      : sweeps between consecutive snapshots (should be >= 2*tau_int)
        n_thermalize : additional burn-in sweeps before sampling begins

        Returns
        -------
        configs : (n_samples, L, L) int8 array of spin snapshots
        """
        beta = 1.0 / T

        if n_thermalize > 0:
            for _ in range(n_thermalize):
                self.sweep(beta)

        configs = np.empty((n_samples, self.L, self.L), dtype=np.int8)
        for k in range(n_samples):
            for _ in range(spacing):
                self.sweep(beta)
            configs[k] = self.snapshot()

        return configs

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"IsingCA(L={self.L}, J={self.J}, "
            f"N={self.L**2}, spins_dtype={self.spins.dtype})"
        )
