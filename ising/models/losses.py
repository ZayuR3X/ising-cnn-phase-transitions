"""
ising/models/losses.py

Composite loss function for IsingCNN training.

Combines three terms:
  1. CrossEntropy  — phase classification
  2. MSE           — temperature regression
  3. Physics reg   — penalizes nonzero predicted magnetization above Tc

Reference:
    Tanaka & Tomiya (2017), J. Phys. Soc. Jpn. 86, 063001
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IsingLoss(nn.Module):
    """
    Composite loss for IsingCNN.

    L = λ_cls  * CrossEntropy(phase_logits, phase_labels)
      + λ_reg  * MSE(T_pred, T_true)
      + λ_phys * physics_regularizer(phase_logits, m_true, T_true, Tc)

    Physics regularizer: penalizes the model when it predicts a high
    probability of the ferromagnetic phase (large |m|) for configurations
    that are clearly above Tc. This injects the known physics that m → 0
    for T > Tc.

    Parameters
    ----------
    lambda_cls  : weight for classification loss
    lambda_reg  : weight for temperature regression loss
    lambda_phys : weight for physics regularizer
    Tc          : critical temperature (used in regularizer)
    """

    def __init__(
        self,
        lambda_cls:  float = 1.0,
        lambda_reg:  float = 0.5,
        lambda_phys: float = 0.1,
        Tc:          float = 2.2692,
    ):
        super().__init__()
        self.lambda_cls  = lambda_cls
        self.lambda_reg  = lambda_reg
        self.lambda_phys = lambda_phys
        self.Tc          = Tc

        self.ce_loss  = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        phase_logits: torch.Tensor,   # (B, 2)
        T_pred:       torch.Tensor,   # (B,)
        phase_labels: torch.Tensor,   # (B,) int
        T_true:       torch.Tensor,   # (B,) float
        m_true:       torch.Tensor,   # (B,) float — |m| from simulation
    ) -> dict:
        """
        Compute composite loss and return breakdown dict.

        Returns
        -------
        dict with keys: "total", "cls", "reg", "phys"
        """
        # 1. Classification loss
        loss_cls = self.ce_loss(phase_logits, phase_labels)

        # 2. Temperature regression loss (normalize by Tc for scale invariance)
        loss_reg = self.mse_loss(T_pred / self.Tc, T_true / self.Tc)

        # 3. Physics regularizer
        # For configurations above Tc, the ferro probability should be low.
        # We penalize: P(ferro | T > Tc) * |m_true|
        ferro_prob  = F.softmax(phase_logits, dim=1)[:, 0]   # P(phase=0=ferro)
        above_tc    = (T_true > self.Tc).float()
        loss_phys   = (above_tc * ferro_prob * m_true).mean()

        # Total
        loss_total = (
            self.lambda_cls  * loss_cls
            + self.lambda_reg  * loss_reg
            + self.lambda_phys * loss_phys
        )

        return {
            "total": loss_total,
            "cls":   loss_cls,
            "reg":   loss_reg,
            "phys":  loss_phys,
        }
