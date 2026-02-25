"""
ising/training/trainer.py

Training loop for IsingCNN with:
  - Composite loss (classification + regression + physics regularizer)
  - AdamW optimizer + CosineAnnealingWarmRestarts scheduler
  - Checkpoint saving (best val loss)
  - Early stopping
  - Per-epoch metrics logging
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional
import yaml

from ising.models.cnn import IsingCNN
from ising.models.losses import IsingLoss
from ising.training.metrics import compute_metrics


class EarlyStopping:
    """Stops training when monitored metric stops improving."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best       = float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """
    Orchestrates IsingCNN training.

    Parameters
    ----------
    model          : IsingCNN instance
    train_loader   : training DataLoader
    val_loader     : validation DataLoader
    cfg            : dict loaded from model.yaml
    device         : torch device
    checkpoint_dir : directory to save checkpoints
    """

    def __init__(
        self,
        model:          IsingCNN,
        train_loader:   DataLoader,
        val_loader:     DataLoader,
        cfg:            dict,
        device:         torch.device,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.cfg          = cfg

        tr_cfg  = cfg["training"]
        Tc      = cfg["model"].get("Tc", 2.2692)

        # Loss
        lc = tr_cfg["loss"]
        self.criterion = IsingLoss(
            lambda_cls  = lc["lambda_cls"],
            lambda_reg  = lc["lambda_reg"],
            lambda_phys = lc["lambda_phys"],
            Tc          = Tc,
        )

        # Optimizer
        opt = tr_cfg["optimizer"]
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr           = opt["lr"],
            weight_decay = opt["weight_decay"],
        )

        # Scheduler
        sch = tr_cfg["scheduler"]
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0   = sch["T_0"],
            T_mult= sch["T_mult"],
        )

        # Early stopping
        es = tr_cfg["early_stopping"]
        self.early_stopping = EarlyStopping(patience=es["patience"])

        self.epochs        = tr_cfg["epochs"]
        self.checkpoint_dir= Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")
        self.history       = []

    # ------------------------------------------------------------------
    # One epoch
    # ------------------------------------------------------------------

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict:
        self.model.train(train)
        total_loss = cls_loss = reg_loss = phys_loss = 0.0
        all_preds, all_labels, all_T_pred, all_T_true = [], [], [], []

        with torch.set_grad_enabled(train):
            for batch in loader:
                x      = batch["cnn_input"].to(self.device)
                phys   = batch["physics"].to(self.device)
                labels = batch["phase"].to(self.device)
                T_true = batch["T"].to(self.device)
                m_true = batch["m"].to(self.device)

                logits, T_pred = self.model(x, phys)
                losses = self.criterion(logits, T_pred, labels, T_true, m_true)

                if train:
                    self.optimizer.zero_grad()
                    losses["total"].backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                bs = x.size(0)
                total_loss += losses["total"].item() * bs
                cls_loss   += losses["cls"].item()   * bs
                reg_loss   += losses["reg"].item()   * bs
                phys_loss  += losses["phys"].item()  * bs

                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                all_T_pred.extend(T_pred.detach().cpu().numpy().tolist())
                all_T_true.extend(T_true.cpu().numpy().tolist())

        n = len(loader.dataset)
        metrics = compute_metrics(
            np.array(all_preds),
            np.array(all_labels),
            np.array(all_T_pred),
            np.array(all_T_true),
        )
        metrics.update({
            "loss":       total_loss / n,
            "loss_cls":   cls_loss   / n,
            "loss_reg":   reg_loss   / n,
            "loss_phys":  phys_loss  / n,
        })
        return metrics

    # ------------------------------------------------------------------
    # Full training run
    # ------------------------------------------------------------------

    def fit(self) -> list:
        """
        Run the full training loop.

        Returns
        -------
        history : list of per-epoch metric dicts
        """
        print(f"\n{'='*65}")
        print(f"Training IsingCNN  |  device={self.device}  |  epochs={self.epochs}")
        print(f"{'='*65}")
        print(f"{'Epoch':>5}  {'TrainLoss':>9}  {'ValLoss':>9}  "
              f"{'Acc':>6}  {'T_rmse':>7}  {'LR':>8}  {'Time':>6}")
        print(f"{'-'*65}")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_m = self._run_epoch(self.train_loader, train=True)
            val_m   = self._run_epoch(self.val_loader,   train=False)

            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]
            dt = time.time() - t0

            row = {
                "epoch":       epoch,
                "train_loss":  train_m["loss"],
                "val_loss":    val_m["loss"],
                "val_acc":     val_m["accuracy"],
                "val_T_rmse":  val_m["T_rmse"],
                "lr":          lr,
            }
            self.history.append(row)

            print(
                f"{epoch:5d}  {train_m['loss']:9.4f}  {val_m['loss']:9.4f}  "
                f"{val_m['accuracy']:6.3f}  {val_m['T_rmse']:7.4f}  "
                f"{lr:8.2e}  {dt:5.1f}s"
            )

            # Checkpoint
            if val_m["loss"] < self.best_val_loss:
                self.best_val_loss = val_m["loss"]
                self._save_checkpoint(epoch, val_m, best=True)

            # Early stopping
            if self.early_stopping.step(val_m["loss"]):
                print(f"\nEarly stopping at epoch {epoch}.")
                break

        print(f"\nBest val loss: {self.best_val_loss:.4f}")
        return self.history

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, metrics: dict, best: bool = False):
        state = {
            "epoch":      epoch,
            "model":      self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "scheduler":  self.scheduler.state_dict(),
            "val_loss":   metrics["loss"],
            "val_acc":    metrics["accuracy"],
        }
        path = self.checkpoint_dir / ("best_model.pth" if best else f"epoch_{epoch:03d}.pth")
        torch.save(state, path)

    def load_best(self):
        """Load the best checkpoint into self.model."""
        path = self.checkpoint_dir / "best_model.pth"
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        print(f"Loaded best model from epoch {state['epoch']} "
              f"(val_loss={state['val_loss']:.4f}, acc={state['val_acc']:.4f})")

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        model:          IsingCNN,
        train_loader:   DataLoader,
        val_loader:     DataLoader,
        cfg_path:       str = "configs/model.yaml",
        device:         Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints",
    ) -> "Trainer":
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        return cls(model, train_loader, val_loader, cfg, device, checkpoint_dir)
