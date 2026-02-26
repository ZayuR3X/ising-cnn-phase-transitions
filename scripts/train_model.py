"""
scripts/train_model.py

CLI entry point for IsingCNN training.

Examples
--------
# Full training run:
python scripts/train_model.py

# Smoke test (2 epochs, small dataset):
python scripts/train_model.py --smoke-test

# Custom config:
python scripts/train_model.py --config configs/model.yaml --data data/raw/J1_L64_snapshots.h5
"""

import argparse
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ising.models.cnn import IsingCNN
from ising.models.losses import IsingLoss
from ising.data.dataset import make_dataloaders
from ising.training.trainer import Trainer
from ising.training.metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train IsingCNN.")
    parser.add_argument("--config",     default="configs/model.yaml")
    parser.add_argument("--data",       default="data/raw/smoke_test_L16.h5")
    parser.add_argument("--smoke-test", action="store_true",
                        help="2 epochs, smoke test dataset, quick validation")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.smoke_test:
        print("\n--- Smoke test mode (2 epochs) ---")
        _smoke_test(device)
        return

    # Full training
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tr = cfg["training"]
    train_loader, val_loader, test_loader = make_dataloaders(
        h5_path          = args.data,
        batch_size       = tr["batch_size"],
        train_frac       = tr["data_split"]["train"],
        val_frac         = tr["data_split"]["val"],
        critical_strip   = tuple(tr["data_split"]["critical_strip"]),
        critical_oversample_factor = tr["critical_oversampling"]["factor"],
        augment          = True,
        spin_flip        = True,
        num_workers      = 0,   # 0 = main process only, faster on CPU
        seed             = tr["seed"],
    )

    model   = IsingCNN()
    trainer = Trainer.from_yaml(model, train_loader, val_loader,
                                cfg_path=args.config, device=device)
    history = trainer.fit()
    trainer.load_best()

    # Final evaluation on test set
    print("\n--- Test set evaluation ---")
    import numpy as np
    all_preds, all_labels, all_Tp, all_Tt = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x      = batch["cnn_input"].to(device)
            phys   = batch["physics"].to(device)
            labels = batch["phase"]
            T_true = batch["T"]
            logits, T_pred = model(x, phys)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_Tp.extend(T_pred.cpu().numpy().tolist())
            all_Tt.extend(T_true.numpy().tolist())

    metrics = compute_metrics(
        np.array(all_preds), np.array(all_labels),
        np.array(all_Tp),    np.array(all_Tt),
    )
    print(f"  accuracy : {metrics['accuracy']:.4f}")
    print(f"  acc_ferro: {metrics['acc_ferro']:.4f}")
    print(f"  acc_para : {metrics['acc_para']:.4f}")
    print(f"  acc_crit : {metrics['acc_crit']:.4f}")
    print(f"  T_rmse   : {metrics['T_rmse']:.4f}")


def _smoke_test(device: torch.device):
    """2-epoch smoke test on the small L=16 dataset."""
    import numpy as np

    train_loader, val_loader, _ = make_dataloaders(
        h5_path          = "data/raw/smoke_test_L16.h5",
        batch_size       = 16,
        critical_strip   = None,
        critical_oversample_factor = 1,
        augment          = False,
        spin_flip        = False,
        num_workers      = 0,
    )

    model     = IsingCNN()
    criterion = IsingLoss(Tc=2.2692)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.to(device)

    for epoch in range(1, 3):
        model.train()
        for batch in train_loader:
            x      = batch["cnn_input"].to(device)
            phys   = batch["physics"].to(device)
            labels = batch["phase"].to(device)
            T_true = batch["T"].to(device)
            m_true = batch["m"].to(device)

            logits, T_pred = model(x, phys)
            losses = criterion(logits, T_pred, labels, T_true, m_true)
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

        # Val
        model.eval()
        preds_all, labels_all, Tp_all, Tt_all = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                x    = batch["cnn_input"].to(device)
                phys = batch["physics"].to(device)
                logits, T_pred = model(x, phys)
                preds_all.extend(logits.argmax(1).cpu().numpy().tolist())
                labels_all.extend(batch["phase"].numpy().tolist())
                Tp_all.extend(T_pred.cpu().numpy().tolist())
                Tt_all.extend(batch["T"].numpy().tolist())

        metrics = compute_metrics(
            np.array(preds_all), np.array(labels_all),
            np.array(Tp_all),    np.array(Tt_all),
        )
        print(f"  Epoch {epoch}: acc={metrics['accuracy']:.3f}  "
              f"T_rmse={metrics['T_rmse']:.3f}")

    print("\nSmoke test passed. Model forward/backward OK.")


if __name__ == "__main__":
    main()
