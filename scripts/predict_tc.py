"""
scripts/predict_tc.py

CLI entry point for critical temperature prediction on a new system (J=2)
using transfer learning from the trained J=1 model.

Examples
--------
# Learning by confusion:
python scripts/predict_tc.py --method confusion --model checkpoints/best_model.pth

# MC Dropout entropy (zero-shot):
python scripts/predict_tc.py --method entropy --model checkpoints/best_model.pth

# Smoke test (tiny pilot dataset generated on the fly):
python scripts/predict_tc.py --smoke-test
"""

import argparse
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ising.models.cnn import IsingCNN
from ising.transfer.confusion import confusion_sweep
from ising.transfer.mc_dropout import mc_dropout_inference
from ising.transfer.fss import extrapolate_Tc, summarize_Tc_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Tc via transfer learning.")
    parser.add_argument("--method",     choices=["confusion", "entropy", "both"],
                        default="both")
    parser.add_argument("--model",      default="checkpoints/best_model.pth")
    parser.add_argument("--data",       default="data/raw/J2_L64_pilot.h5",
                        help="J=2 pilot HDF5 dataset")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Generate a tiny J=2 dataset on the fly and run both methods")
    return parser.parse_args()


def load_model(path: str, device: torch.device) -> IsingCNN:
    model = IsingCNN()
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    print(f"Loaded model from {path}  (epoch={state['epoch']}, val_acc={state['val_acc']:.4f})")
    return model


def generate_smoke_pilot(out_path: str = "data/raw/smoke_J2_L16.h5"):
    """Generate a tiny J=2 pilot dataset for the smoke test."""
    import h5py, time
    from ising.simulation.ca_ising import IsingCA
    from ising.simulation.thermalization import smart_init, thermalize
    from ising.simulation.observables import build_cnn_input, per_config_label

    print("\nGenerating tiny J=2 pilot dataset (L=16, 5T, 8 samples)...")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    J    = 2.0
    L    = 16
    temps= np.array([3.5, 4.0, 4.538, 5.2, 6.0])
    N_s  = 8

    ca = IsingCA(L=L, J=J, seed=99)

    with h5py.File(out_path, "w") as hf:
        hf.attrs.update({"J": J, "L": L, "Tc_onsager": 4.5138})
        N = len(temps) * N_s
        ds_cnn = hf.create_dataset("cnn_input",    shape=(N,4,L,L), dtype=np.float32)
        ds_T   = hf.create_dataset("temperature",  shape=(N,),      dtype=np.float32)
        ds_m   = hf.create_dataset("magnetization",shape=(N,),      dtype=np.float32)
        ds_E   = hf.create_dataset("energy",       shape=(N,),      dtype=np.float32)
        ds_ph  = hf.create_dataset("phase",        shape=(N,),      dtype=np.int8)

        idx = 0
        for T in temps:
            smart_init(ca, T)
            thermalize(ca, T, n_sweeps_default=1000, n_sweeps_critical=2000,
                       critical_window=0.5, verbose=False)
            configs = ca.sample(T=T, n_samples=N_s, spacing=5)
            for k in range(N_s):
                lbl = per_config_label(configs[k], T=T, J=J)
                ds_cnn[idx] = build_cnn_input(configs[k], J=J)
                ds_T[idx]   = T
                ds_m[idx]   = lbl["m"]
                ds_E[idx]   = lbl["E"]
                ds_ph[idx]  = lbl["phase"]
                idx += 1

    print(f"Pilot dataset saved: {out_path}  ({N} configs)")
    return out_path


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.smoke_test:
        _smoke_test(device)
        return

    model    = load_model(args.model, device)
    results  = {}
    fss_data = {"L": [], "Tc": [], "Tc_err": []}

    if args.method in ("confusion", "both"):
        print("\n--- Learning by Confusion ---")
        conf = confusion_sweep(
            model, args.data,
            W_min=3.0, W_max=6.5, n_steps=70,
            head_epochs=30, device=device, verbose=True,
        )
        results["confusion"] = conf
        fss_data["L"].append(64)
        fss_data["Tc"].append(conf["Tc_pred"])
        fss_data["Tc_err"].append(conf["Tc_std"])

    if args.method in ("entropy", "both"):
        print("\n--- MC Dropout Predictive Entropy ---")
        ent = mc_dropout_inference(
            model, args.data,
            n_passes=100, device=device, verbose=True,
        )
        results["entropy"] = ent

    # FSS extrapolation (single L here, needs multi-L for full analysis)
    if fss_data["L"]:
        fss_res = extrapolate_Tc(
            np.array(fss_data["L"]),
            np.array(fss_data["Tc"]),
            np.array(fss_data["Tc_err"]) if fss_data["Tc_err"] else None,
            verbose=True,
        )
        summarize_Tc_predictions(
            fss_result       = fss_res,
            confusion_result = results.get("confusion"),
            entropy_result   = results.get("entropy"),
            J                = 2.0,
        )


def _smoke_test(device: torch.device):
    """End-to-end smoke test: generate J=2 data → run both methods."""
    pilot_path = generate_smoke_pilot()

    # Build an untrained model (smoke test only checks pipeline, not accuracy)
    model = IsingCNN().to(device)
    model.eval()
    print("\nUsing untrained model for smoke test pipeline check.")

    print("\n--- Confusion sweep (5 steps only) ---")
    conf = confusion_sweep(
        model, pilot_path,
        W_min=3.5, W_max=6.0, n_steps=5,
        head_epochs=3, batch_size=16,
        device=device, verbose=True,
    )
    print(f"Confusion Tc_pred: {conf['Tc_pred']:.3f}")

    print("\n--- MC Dropout entropy (10 passes) ---")
    ent = mc_dropout_inference(
        model, pilot_path,
        n_passes=10, batch_size=16,
        device=device, verbose=True,
    )
    print(f"Entropy Tc_pred: {ent['Tc_pred']:.3f}")

    print("\nSmoke test complete. Pipeline OK.")
    print("Note: predictions are random (untrained model) — accuracy not expected.")


if __name__ == "__main__":
    main()
