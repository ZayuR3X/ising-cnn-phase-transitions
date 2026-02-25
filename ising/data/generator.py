"""
ising/data/generator.py

Dataset generation orchestrator for the 2D Ising Model.

Runs the full simulation pipeline:
    1. Build temperature grid (coarse + dense near Tc)
    2. For each temperature: thermalize → estimate tau_int → collect samples
    3. Compute per-config labels and ensemble observables
    4. Save everything to a compressed HDF5 file

Usage (via script):
    python scripts/generate_dataset.py --config configs/simulation.yaml

Or directly:
    from ising.data.generator import DatasetGenerator
    gen = DatasetGenerator.from_yaml("configs/simulation.yaml")
    gen.run(L=64)
"""

import time
import numpy as np
import h5py
from pathlib import Path
from typing import Optional
import yaml

from ising.simulation.ca_ising import IsingCA
from ising.simulation.thermalization import smart_init, thermalize
from ising.simulation.autocorrelation import measure_tau_int
from ising.simulation.observables import (
    build_cnn_input,
    ensemble_observables,
    per_config_label,
    fss_corrected_phase_label,
)


# ---------------------------------------------------------------------------
# Temperature grid builder
# ---------------------------------------------------------------------------

def build_temperature_grid(
    coarse_min: float,
    coarse_max: float,
    coarse_step: float,
    dense_min: float,
    dense_max: float,
    dense_step: float,
) -> np.ndarray:
    """
    Build a merged temperature grid: coarse away from Tc, dense near Tc.
    Duplicates at the boundary are removed and the array is sorted.
    """
    coarse = np.arange(coarse_min, coarse_max + 1e-9, coarse_step)
    dense  = np.arange(dense_min,  dense_max  + 1e-9, dense_step)
    combined = np.concatenate([coarse, dense])
    combined = np.round(combined, decimals=6)
    merged   = np.unique(combined)
    return merged


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class DatasetGenerator:
    """
    Orchestrates the full simulation → HDF5 pipeline.

    Parameters
    ----------
    J                    : exchange coupling constant
    n_samples            : independent snapshots per temperature
    n_thermalize         : burn-in sweeps (away from Tc)
    n_thermalize_critical: burn-in sweeps near Tc
    critical_window      : |T - Tc| threshold for extended thermalization
    spacing_default      : fallback spacing if tau_int estimation is skipped
    n_autocorr_sweeps    : sweeps used for tau_int measurement
    output_path_template : path with {L} placeholder, e.g. "data/raw/J1_L{L}.h5"
    compression          : HDF5 compression filter
    compression_opts     : compression level
    seed                 : random seed
    estimate_tau         : whether to measure tau_int before sampling
    verbose              : print progress
    """

    def __init__(
        self,
        J: float = 1.0,
        n_samples: int = 500,
        n_thermalize: int = 10_000,
        n_thermalize_critical: int = 50_000,
        critical_window: float = 0.3,
        spacing_default: int = 20,
        n_autocorr_sweeps: int = 3_000,
        output_path_template: str = "data/raw/J{J_int}_L{L}_snapshots.h5",
        compression: str = "gzip",
        compression_opts: int = 4,
        seed: int = 42,
        estimate_tau: bool = True,
        verbose: bool = True,
    ):
        self.J                     = J
        self.n_samples             = n_samples
        self.n_thermalize          = n_thermalize
        self.n_thermalize_critical = n_thermalize_critical
        self.critical_window       = critical_window
        self.spacing_default       = spacing_default
        self.n_autocorr_sweeps     = n_autocorr_sweeps
        self.output_path_template  = output_path_template
        self.compression           = compression
        self.compression_opts      = compression_opts
        self.seed                  = seed
        self.estimate_tau          = estimate_tau
        self.verbose               = verbose

        self.Tc = (2.0 * J) / np.log(1.0 + np.sqrt(2.0))

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "DatasetGenerator":
        """Instantiate from a simulation.yaml config file."""
        with open(path) as f:
            cfg = yaml.safe_load(f)

        sim = cfg["simulation"]
        mc  = sim["mc"]
        out = sim["output"]
        T   = sim["temperature"]

        return cls(
            J                     = sim["J"],
            n_samples             = mc["n_samples"],
            n_thermalize          = mc["n_thermalize"],
            n_thermalize_critical = mc["n_thermalize_critical"],
            critical_window       = mc["critical_window"],
            spacing_default       = mc["spacing_default"],
            n_autocorr_sweeps     = mc["n_autocorr_sweeps"],
            output_path_template  = out["path"],
            compression           = out.get("compression", "gzip"),
            compression_opts      = out.get("compression_opts", 4),
            seed                  = sim.get("seed", 42),
        )

    # ------------------------------------------------------------------
    # Core run
    # ------------------------------------------------------------------

    def run(
        self,
        L: int,
        temperatures: Optional[np.ndarray] = None,
    ) -> Path:
        """
        Generate the full dataset for lattice size L.

        Parameters
        ----------
        L            : lattice linear size
        temperatures : custom temperature array (overrides yaml grid)

        Returns
        -------
        output_path : Path to the written HDF5 file
        """
        if temperatures is None:
            raise ValueError(
                "Pass a temperatures array or use run_from_config()."
            )

        J_int = int(round(self.J))
        out_path = Path(
            self.output_path_template.format(L=L, J_int=J_int)
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ca = IsingCA(L=L, J=self.J, seed=self.seed)
        n_T = len(temperatures)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Dataset generation: J={self.J}, L={L}, Tc={self.Tc:.4f}")
            print(f"Temperatures: {n_T} points in [{temperatures[0]:.3f}, {temperatures[-1]:.3f}]")
            print(f"Samples/T: {self.n_samples}")
            print(f"Output: {out_path}")
            print(f"{'='*60}\n")

        t_start = time.time()

        with h5py.File(out_path, "w") as hf:
            # --- metadata ---
            hf.attrs["J"]         = self.J
            hf.attrs["L"]         = L
            hf.attrs["Tc_onsager"]= self.Tc
            hf.attrs["n_samples"] = self.n_samples
            hf.attrs["seed"]      = self.seed
            hf.attrs["created"]   = time.strftime("%Y-%m-%dT%H:%M:%S")

            # --- pre-allocate datasets ---
            N_total = n_T * self.n_samples
            ds_spins  = hf.create_dataset(
                "spins", shape=(N_total, L, L), dtype=np.int8,
                chunks=(min(64, N_total), L, L),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )
            ds_cnn = hf.create_dataset(
                "cnn_input", shape=(N_total, 4, L, L), dtype=np.float32,
                chunks=(min(32, N_total), 4, L, L),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )
            ds_T      = hf.create_dataset("temperature", shape=(N_total,), dtype=np.float32)
            ds_phase  = hf.create_dataset("phase",       shape=(N_total,), dtype=np.int8)
            ds_m      = hf.create_dataset("magnetization",shape=(N_total,), dtype=np.float32)
            ds_E      = hf.create_dataset("energy",       shape=(N_total,), dtype=np.float32)

            # --- ensemble observables (one row per temperature) ---
            ens_grp = hf.create_group("ensemble")
            ens_T   = ens_grp.create_dataset("T",   shape=(n_T,), dtype=np.float32)
            ens_m   = ens_grp.create_dataset("m",   shape=(n_T,), dtype=np.float32)
            ens_chi = ens_grp.create_dataset("chi", shape=(n_T,), dtype=np.float32)
            ens_C   = ens_grp.create_dataset("C",   shape=(n_T,), dtype=np.float32)
            ens_U4  = ens_grp.create_dataset("U4",  shape=(n_T,), dtype=np.float32)
            ens_E   = ens_grp.create_dataset("E",   shape=(n_T,), dtype=np.float32)
            ens_tau = ens_grp.create_dataset("tau_int", shape=(n_T,), dtype=np.float32)

            # --- main loop ---
            global_idx = 0
            for t_idx, T in enumerate(temperatures):
                t0 = time.time()
                beta = 1.0 / T
                near_tc = abs(T - self.Tc) < self.critical_window

                if self.verbose:
                    print(f"[{t_idx+1:3d}/{n_T}] T={T:.3f}  ", end="", flush=True)

                # 1. Smart init + thermalize
                smart_init(ca, T)
                therm_info = thermalize(
                    ca, T,
                    n_sweeps_default   = self.n_thermalize,
                    n_sweeps_critical  = self.n_thermalize_critical,
                    critical_window    = self.critical_window,
                    verbose            = False,
                )

                # 2. Estimate tau_int
                if self.estimate_tau:
                    tau_result = measure_tau_int(
                        ca, T,
                        n_measure_sweeps = self.n_autocorr_sweeps,
                        verbose          = False,
                    )
                    spacing = tau_result["spacing"]
                    tau_val = tau_result["tau_int"]
                else:
                    spacing = self.spacing_default
                    tau_val = float("nan")

                if self.verbose:
                    print(
                        f"therm={'OK' if therm_info['converged'] else 'MAX':3s}  "
                        f"tau={tau_val:5.1f}  spacing={spacing:3d}  ",
                        end="", flush=True,
                    )

                # 3. Collect independent samples
                configs = ca.sample(
                    T           = T,
                    n_samples   = self.n_samples,
                    spacing     = spacing,
                    n_thermalize= 0,   # already thermalized
                )

                # 4. Compute per-config labels and CNN inputs
                start = global_idx
                end   = global_idx + self.n_samples

                for k in range(self.n_samples):
                    ds_spins[start + k]  = configs[k]
                    ds_cnn[start + k]    = build_cnn_input(configs[k], J=self.J)
                    lbl = per_config_label(configs[k], T=T, J=self.J)
                    ds_T[start + k]     = lbl["T"]
                    ds_phase[start + k] = lbl["phase"]
                    ds_m[start + k]     = lbl["m"]
                    ds_E[start + k]     = lbl["E"]

                # 5. Ensemble observables
                ens_obs = ensemble_observables(configs, T=T, J=self.J)
                ens_T[t_idx]   = T
                ens_m[t_idx]   = ens_obs["m_mean"]
                ens_chi[t_idx] = ens_obs["chi"]
                ens_C[t_idx]   = ens_obs["C"]
                ens_U4[t_idx]  = ens_obs["U4"]
                ens_E[t_idx]   = ens_obs["E_mean"]
                ens_tau[t_idx] = tau_val

                global_idx += self.n_samples

                if self.verbose:
                    dt = time.time() - t0
                    print(f"m={ens_obs['m_mean']:.3f}  chi={ens_obs['chi']:.2f}  [{dt:.1f}s]")

        total_time = time.time() - t_start
        if self.verbose:
            print(f"\nDone. {N_total} configs saved to {out_path}")
            print(f"Total time: {total_time/60:.1f} min")

        return out_path

    def run_from_config(self, L: int, cfg_path: str = "configs/simulation.yaml") -> Path:
        """Load temperature grid from yaml and run."""
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        T_cfg = cfg["simulation"]["temperature"]
        temperatures = build_temperature_grid(
            coarse_min  = T_cfg["coarse"]["T_min"],
            coarse_max  = T_cfg["coarse"]["T_max"],
            coarse_step = T_cfg["coarse"]["step"],
            dense_min   = T_cfg["dense"]["T_min"],
            dense_max   = T_cfg["dense"]["T_max"],
            dense_step  = T_cfg["dense"]["step"],
        )
        return self.run(L=L, temperatures=temperatures)
