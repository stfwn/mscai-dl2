from typing import Callable

import torch
from torch import nn, Tensor
import torch.distributions.beta as dist_beta


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
        self.register_buffer("prior", prior)

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
            targets[
                (torch.arange(targets.size(0)).repeat(n_steps))
            ],  # Repeat targets as needed to match.
            reduction="none",
        ).view(n_steps, batch_size)
        l_rec = torch.einsum("ij,ij->j", p, task_losses).mean()

        # Regularization term
        l_reg = self.beta * self.KL(
            p.transpose(1, 0).log(),
            self.prior[:n_steps].unsqueeze(0).expand(batch_size, n_steps),  # type: ignore
        )

        return l_rec + l_reg

class PonderBayesianLoss(nn.Module):
    def __init__(
        self,
        task_loss_fn: Callable,
        beta_prior: tuple,
        max_ponder_steps: int,
    ):
        """
        Args:
            # beta: Weight for the regularization loss term.
            lambda_reg: Parameterizes the (Bernoulli) prior.
            task_loss_fn: Loss function for the actual task (e.g. MSE or CE).
            max_ponder_steps
        """
        super().__init__()
        self.task_loss_fn = task_loss_fn
        # self.beta = beta
        self.beta_prior = beta_prior
        self.KL = nn.KLDivLoss(reduction="batchmean")

        self.prior = dist_beta.Beta(beta_prior[0], beta_prior[1])

    def forward(self, preds: Tensor, p: Tensor, lambdas: Tensor, halted_at: Tensor, targets: Tensor):
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
            targets[
                (torch.arange(targets.size(0)).repeat(n_steps))
            ],  # Repeat targets as needed to match.
            reduction="none",
        ).view(n_steps, batch_size)
        l_rec = torch.log(torch.einsum("ij,ij->j", p, task_losses).mean())

        # Regularization term
        l_reg = self.KL(
            lambdas.transpose(1, 0).log(),
            self.prior.rsample(sample_shape=(batch_size, n_steps)),  # type: ignore
        )

        return l_rec + l_reg
