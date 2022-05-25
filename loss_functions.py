from typing import Callable

import torch
from torch import nn, Tensor


class PonderLoss(nn.Module):
    def __init__(
        self,
        task_loss_fn: Callable,
        beta: float,
        lambda_prior: float,
        max_ponder_steps: int,
    ):
        """
        Args:
            beta: Weight for the regularization loss term.
            lambda_reg: Parameterizes the (Bernoulli) prior.
            task_loss_fn: Loss function for the actual task (e.g. MSE or CE).
            max_ponder_steps
        """
        super().__init__()
        self.task_loss_fn = task_loss_fn
        self.beta = beta
        self.lambda_prior = lambda_prior
        self.KL = nn.KLDivLoss(reduction="batchmean")

        prior = lambda_prior * (1 - lambda_prior) ** torch.arange(max_ponder_steps)
        prior = prior / prior.sum()
        self.register_buffer("log_prior", prior.log())

    def forward(self, preds: Tensor, p: Tensor, halted_at: Tensor, targets: Tensor):
        """
        Args:
            `preds`: Predictions of shape (ponder_steps, batch_size, logits)
            `p`: Cumulative probability of reaching and then stopping at each
                step of shape (step, batch)
            `halted_at`: Indices of steps where each sample actually stopped of
                shape (batch)
            `targets`: Targets of shape (batch_size)

        """
        n_steps, batch_size, _ = preds.shape

        # Reconstruction term
        task_losses = self.task_loss_fn(
            preds.view(
                -1, preds.size(-1)
            ),  # View pred steps as individual classifications.
            targets.repeat(n_steps),  # Repeat targets as needed to match.
            reduction="none",
        ).view(n_steps, batch_size)
        l_rec = (task_losses * p).sum(0).mean()

        # Regularization term
        p_t = p.transpose(1, 0)
        l_reg = self.KL(self.log_prior[:n_steps].expand_as(p_t), p_t)

        return l_rec, self.beta * l_reg
