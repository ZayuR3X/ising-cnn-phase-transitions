"""
scripts/visualize_results.py

CLI entry point for generating all project figures.

Examples
--------
# All figures (requires trained model + full dataset):
python scripts/visualize_results.py --fig all

# Individual figures:
python scripts/visualize_results.py --fig snapshots
python scripts/visualize_results.py --fig phase_diagram
python scripts/visualize_results.py --fig gradcam
python scripts/visualize_results.py --fig scaling

# Smoke test (works with smoke_test_L16.h5 + untrained model):
python scripts/visualize_results.py --smoke-test
"""

import argparse
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, safe on headless servers
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


FIGURES_DIR = Path("figures")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate project figures.")
    parser.add_argument("--fig", default="all",
                        choices=["all","snapshots","channels","gallery",
                                 "phase_diagram","confusion","gradcam","scaling"])
    parser.add_argument("--data",  default="data/raw/smoke_test_L16.h5")
    parser.add_argument("--model", default="checkpoints/best_model.pth")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def ensure_dirs():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def fig_snapshots(h5_path: str):
    from ising.visualization.spin_configs import plot_snapshots
    import h5py
    with h5py.File(h5_path, "r") as hf:
        spins = hf["spins"][:50].copy()
        temps = hf["temperature"][:50].copy()
    fig = plot_snapshots(spins, temps, n_show=5,
                         save_path=str(FIGURES_DIR / "spin_snapshots.png"))
    plt.close(fig)
    print("  saved: figures/spin_snapshots.png")


def fig_channels(h5_path: str, J: float = 1.0):
    from ising.visualization.spin_configs import plot_cnn_channels
    import h5py
    with h5py.File(h5_path, "r") as hf:
        spin = hf["spins"][0].copy()
        T    = float(hf["temperature"][0])
    fig = plot_cnn_channels(spin, J=J, T=T,
                             save_path=str(FIGURES_DIR / "cnn_channels.png"))
    plt.close(fig)
    print("  saved: figures/cnn_channels.png")


def fig_gallery(h5_path: str):
    from ising.visualization.spin_configs import plot_phase_gallery
    fig = plot_phase_gallery(h5_path,
                              save_path=str(FIGURES_DIR / "phase_gallery.png"))
    plt.close(fig)
    print("  saved: figures/phase_gallery.png")


def fig_phase_diagram(h5_paths: list):
    from ising.visualization.phase_diagram import plot_summary
    fig = plot_summary(h5_paths,
                       save_path=str(FIGURES_DIR / "phase_diagram.png"))
    plt.close(fig)
    print("  saved: figures/phase_diagram.png")


def fig_confusion(confusion_result: dict, entropy_result: dict, J: float):
    from ising.visualization.confusion_plot import plot_transfer_summary
    fig = plot_transfer_summary(
        confusion_result, entropy_result, J=J,
        save_path=str(FIGURES_DIR / "transfer_learning.png"),
    )
    plt.close(fig)
    print("  saved: figures/transfer_learning.png")


def fig_gradcam(h5_path: str, model_path: str):
    from ising.visualization.gradcam import plot_gradcam
    from ising.models.cnn import IsingCNN
    import h5py

    device = torch.device("cpu")
    model  = IsingCNN()
    if Path(model_path).exists():
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state["model"])
    model.eval()

    with h5py.File(h5_path, "r") as hf:
        spins = hf["spins"][:].copy()
        temps = hf["temperature"][:].copy()
        J     = float(hf.attrs.get("J", 1.0))

    fig = plot_gradcam(model, spins, temps, device, J=J, n_show=5,
                       save_path=str(FIGURES_DIR / "gradcam.png"))
    plt.close(fig)
    print("  saved: figures/gradcam.png")


def fig_scaling(h5_paths: list):
    from ising.visualization.scaling_plots import plot_exponent_summary
    fig = plot_exponent_summary(
        save_path=str(FIGURES_DIR / "exponent_table.png")
    )
    plt.close(fig)
    print("  saved: figures/exponent_table.png")


def smoke_test():
    """Run all visualization functions with the smoke test dataset."""
    h5_path = "data/raw/smoke_test_L16.h5"
    ensure_dirs()

    print("--- Snapshots ---")
    fig_snapshots(h5_path)

    print("--- CNN Channels ---")
    fig_channels(h5_path, J=1.0)

    print("--- Phase Gallery ---")
    fig_gallery(h5_path)

    print("--- Phase Diagram ---")
    fig_phase_diagram([h5_path])

    print("--- Scaling / Exponent Table ---")
    fig_scaling([h5_path])

    print("--- Grad-CAM (untrained model) ---")
    fig_gradcam(h5_path, model_path="checkpoints/best_model.pth")

    print("\nSmoke test complete. Check figures/ directory.")


def main():
    args = parse_args()
    ensure_dirs()

    if args.smoke_test:
        smoke_test()
        return

    which = args.fig
    h5    = args.data

    if which in ("all", "snapshots"):  fig_snapshots(h5)
    if which in ("all", "channels"):   fig_channels(h5)
    if which in ("all", "gallery"):    fig_gallery(h5)
    if which in ("all", "phase_diagram"): fig_phase_diagram([h5])
    if which in ("all", "gradcam"):    fig_gradcam(h5, args.model)
    if which in ("all", "scaling"):    fig_scaling([h5])

    print(f"\nAll requested figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
