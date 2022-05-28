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
        prior = prior / prior.sum()
        self.register_buffer("log_prior", prior.log())

    def forward(self, preds: Tensor, p: Tensor, halted_at: Tensor, targets: Tensor, **kwargs):
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


class PonderBayesianLoss(nn.Module):
    def __init__(
        self,
        task_loss_fn: Callable,
        beta_prior: tuple,
        max_ponder_steps: int,
        scale_reg: float,
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
        self.beta_prior = beta_prior
        self.KL = nn.KLDivLoss(reduction="none")
        self.scale_reg = scale_reg

        self.prior = dist_beta.Beta(beta_prior[0], beta_prior[1])

    def forward(self, preds: Tensor, p: Tensor, halted_at: Tensor, targets: Tensor, **kwargs):
        """
        Args:
            `preds`: Predictions of shape (ponder_steps, batch_size, logits)
            `p`: Cumulative probability of reaching and then stopping at each
                step of shape (step, batch)
            `halted_at`: Indices of steps where each sample actually stopped of
                shape (batch)
            `targets`: Targets of shape (batch_size)

        """
        assert "lambdas" in kwargs, "Must provide lambdas!"

        lambdas = kwargs["lambdas"]

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
        l_reg = (
            self.KL(
                self.prior.rsample(sample_shape=(batch_size, n_steps)).to(lambdas.device).log(),  # type: ignore
                lambdas.transpose(1, 0),
            )
            .sum(1)
            .mean()
        )  # Sum over the number of steps, then mean over the batch.

        return l_rec, self.scale_reg * l_reg
