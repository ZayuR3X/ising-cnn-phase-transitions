"""
scripts/generate_dataset.py

CLI entry point for dataset generation.

Examples
--------
# Full dataset, default config:
python scripts/generate_dataset.py

# Custom lattice size:
python scripts/generate_dataset.py --L 32

# Quick smoke test (small, fast):
python scripts/generate_dataset.py --smoke-test

# Custom config:
python scripts/generate_dataset.py --config configs/simulation.yaml --L 64
"""

import argparse
import numpy as np
import sys
from pathlib import Path

# Make sure the package root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ising.data.generator import DatasetGenerator, build_temperature_grid


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 2D Ising Model spin configuration dataset."
    )
    parser.add_argument(
        "--config", default="configs/simulation.yaml",
        help="Path to simulation YAML config (default: configs/simulation.yaml)"
    )
    parser.add_argument(
        "--L", type=int, default=None,
        help="Lattice size (overrides yaml lattice_sizes, runs single L)"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick test: L=16, few temperatures, few samples"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.smoke_test:
        # ----------------------------------------------------------------
        # Smoke test: tiny dataset, runs in ~30 seconds on CPU
        # ----------------------------------------------------------------
        print("Running smoke test (L=16, 5 temperatures, 10 samples each)...")
        gen = DatasetGenerator(
            J                     = 1.0,
            n_samples             = 10,
            n_thermalize          = 500,
            n_thermalize_critical = 1_000,
            critical_window       = 0.3,
            spacing_default       = 5,
            n_autocorr_sweeps     = 200,
            output_path_template  = "data/raw/smoke_test_L{L}.h5",
            seed                  = 42,
            estimate_tau          = True,
            verbose               = True,
        )
        temperatures = np.array([1.5, 2.0, 2.269, 2.8, 3.5])
        out = gen.run(L=16, temperatures=temperatures)
        print(f"\nSmoke test complete. File: {out}")
        _verify_hdf5(out)
        return

    # ----------------------------------------------------------------
    # Full run from config
    # ----------------------------------------------------------------
    gen = DatasetGenerator.from_yaml(args.config)

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    lattice_sizes = (
        [args.L] if args.L is not None
        else cfg["simulation"]["lattice_sizes"]
    )

    for L in lattice_sizes:
        gen.run_from_config(L=L, cfg_path=args.config)


def _verify_hdf5(path: Path):
    """Basic integrity check on a generated HDF5 file."""
    import h5py
    print(f"\nVerifying {path} ...")
    with h5py.File(path, "r") as hf:
        print(f"  Keys       : {list(hf.keys())}")
        print(f"  spins shape: {hf['spins'].shape}")
        print(f"  cnn_input  : {hf['cnn_input'].shape}")
        print(f"  T range    : [{hf['temperature'][:].min():.3f}, "
              f"{hf['temperature'][:].max():.3f}]")
        print(f"  phase dist : {dict(zip(*np.unique(hf['phase'][:], return_counts=True)))}")
        print(f"  Ensemble T : {hf['ensemble/T'][:].tolist()}")
        print("  OK")


if __name__ == "__main__":
    main()
