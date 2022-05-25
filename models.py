from typing import Literal, Optional

import torch
import torch.distributions.beta as dist_beta
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning import LightningModule
from torch import nn, optim

from loss_functions import PonderBayesianLoss, PonderLoss


class PonderNet(LightningModule):
    def __init__(
        self,
        step_function: Literal["mlp", "rnn", "seq_rnn"],
        step_function_args: dict,
        task: str,
        max_ponder_steps: int,
        preds_reduction_method: Literal["ponder", "bayesian"] = "ponder",
        encoder: Optional[str] = None,
        encoder_args: Optional[dict] = None,
        learning_rate: float = 3e-4,
        lambda_prior: float = 0.2,
        loss_beta: float = 0.01,
        ponder_epsilon: float = 0.05,
    ):
        """
        Args:
            preds_reduction_method: Method by which the predictions at each
                step are reduced to a final prediction.

                - `ponder` (default): use the prediction when the coin flip
                  parameterized by lamdbda_n landed on stop, or when the
                  cumulative probability of stopping was > 1 - epsilon (true to
                  the paper).
                - `bayesian`: use the weighted average of the predictions at
                  every step, where the weights are decided by the geometric
                  prior, parameterized as in the loss.
        """
        super().__init__()
        self.save_hyperparameters()

        # Metrics
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

        # Encoder
        if encoder:
            raise NotImplementedError
            self.encoder = {}[encoder](**encoder_args)
        else:
            self.encoder = lambda x: x  # type: ignore

        # Step function
        self.allow_early_return = preds_reduction_method == "ponder"
        sf_class = {
            "mlp": PonderMLP,
            "rnn": PonderRNN,
            "seq_rnn": PonderSequentialRNN,
        }.get(step_function)
        if not sf_class:
            raise ValueError(f"Unknown step function: '{step_function}'")
        self.step_function = sf_class(**step_function_args)

        # Loss
        lf_class = {
            "classification": (
                lambda: PonderLoss(
                    task_loss_fn=F.cross_entropy,
                    beta=loss_beta,
                    lambda_prior=lambda_prior,
                    max_ponder_steps=max_ponder_steps,
                )
            ),
            "bayesian-classification": (
                lambda: PonderBayesianLoss(
                    task_loss_fn=F.cross_entropy,
                    beta_prior=(10, 10),
                    max_ponder_steps = max_ponder_steps,
                    scale_reg = loss_beta,
                )
            )
        }.get(task)
        if not lf_class:
            raise NotImplementedError(f"Unknown task: '{task}'")
        self.loss_function = lf_class()

        # Prediction reduction
        prior = lambda_prior * (1 - lambda_prior) ** torch.arange(max_ponder_steps)
        self.register_buffer("prior", prior)
        preds_reduction_fn = {
            "ponder": self.reduce_preds_ponder,
            "bayesian": self.reduce_preds_bayesian,
        }.get(preds_reduction_method)
        if not preds_reduction_fn:
            raise ValueError(
                f"Unknown preds reduction method: '{preds_reduction_method}'"
            )
        self.preds_reduction_fn = preds_reduction_fn

        # Optimizer
        self.optimizer_class = optim.Adam

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            params=self.parameters(), lr=self.hparams.learning_rate
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.1
        )
        return [optimizer], [{"scheduler": scheduler, "monitor": "loss/val"}]

    def reduce_preds(self, preds, halted_at):
        """
        Get the final predictions where the network decided to halt.

        Args:
            preds: (step, batch, logit)
            halted_at: (batch)

        Returns:
            (batch, logit)
        """
        return self.preds_reduction_fn(preds, halted_at)

    @staticmethod
    def reduce_preds_ponder(preds, halted_at):
        """
       Reduces predictons from multiple ponder steps to one prediction,
       using halted_at.
       out_dict: dictionary containing:
               preds: (ponder_steps, batch_size, logits)
               p: halting probability (ponder_steps, batch_size)
       :return: predictions (batch_size, logits)
        """
        return preds.permute(1, 2, 0)[torch.arange(preds.size(1)), :, halted_at]

    @staticmethod
    def reduce_preds_bayesian(out_dict):
        """
        Reduces predictons from multiple ponder steps to one prediction,
        using weighted average.
        out_dict: dictionary containing:
                preds: (ponder_steps, batch_size, logits)
                p: halting probability (ponder_steps, batch_size)
        :return: predictions (batch_size, logits)
        """
        preds = out_dict["preds"]
        p = out_dict["p"]
        return torch.einsum("sbl,sb->bl", preds, p)

    def forward(self, x):
        batch_size = x.size(0)
        state = None  # State that transfers across steps

        x = self.encoder(x)

        # Probabilities of not having halted so far.
        p = []  # Probabilities of halting at each step.
        cum_p_n = x.new_zeros(batch_size)  # Cumulative probability of halting.
        prob_not_halted_so_far = 1
        halted_at = x.new_zeros(batch_size)
        y_hat = []

        for n in range(1, self.hparams.max_ponder_steps + 1):
            # 1) Pass through model
            y_hat_n, state, lambda_n = self.step_function(x, state)

            lambda_n = lambda_n.squeeze().sigmoid()
            y_hat.append(y_hat_n)
            p.append(prob_not_halted_so_far * lambda_n)
            prob_not_halted_so_far = prob_not_halted_so_far * (1 - lambda_n)
            cum_p_n += p[n - 1]

            # Update halted_at where needed (one-liner courtesy of jankrepl on GitHub)
            halted_at = (n * (halted_at == 0) * lambda_n.bernoulli()).max(halted_at)

            # If the probability is over epsilon we always stop.
            halted_at[
                torch.logical_and(
                    (cum_p_n > (1 - self.hparams.ponder_epsilon)), (halted_at == 0)
                )
            ] = n

            if self.allow_early_return and halted_at.all():
                break

        # Last step should be used if halting prob never reached above 1-epsilon
        halted_at[halted_at == 0] = self.hparams.max_ponder_steps

        # Normalize p so it's an actual distribution
        p = torch.stack(p)
        p = p / p.sum(0)

        return (
            torch.stack(y_hat),  # (step, batch, logit)
            p,  # (step, batch)
            (halted_at - 1).long(),  # (batch)
        )

    def training_step(self, batch, batch_idx):
        x, targets = batch
        preds, p, halted_at = self(x)
        rec_loss, reg_loss = self.loss_function(preds, p, halted_at, targets)
        loss = rec_loss + reg_loss
        self.log("loss/rec_train", rec_loss)
        self.log("loss/reg_train", reg_loss)

        self.train_acc(self.reduce_preds(preds, halted_at), targets)
        self.log("acc/train", self.train_acc, on_step=False, on_epoch=True)
        self.log("loss/train", loss)

        halted_at = halted_at.float()
        self.log("halted_at/mean/train", halted_at.mean(), on_step=True, on_epoch=True)
        self.log("halted_at/std/train", halted_at.std(), on_step=True, on_epoch=True)
        self.log(
            "halted_at/median/train", halted_at.median(), on_step=True, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        preds, p, halted_at = self(x)
        rec_loss, reg_loss = self.loss_function(preds, p, halted_at, targets)
        loss = rec_loss + reg_loss
        self.log("loss/rec_val", rec_loss)
        self.log("loss/reg_val", reg_loss)

        self.val_acc(self.reduce_preds(preds, halted_at), targets)
        self.log("loss/val", loss)
        self.log("acc/val", self.val_acc, on_step=False, on_epoch=True)

        halted_at = halted_at.float()
        self.log("halted_at/mean/val", halted_at.mean(), on_step=True, on_epoch=True)
        self.log("halted_at/std/val", halted_at.std(), on_step=True, on_epoch=True)
        self.log(
            "halted_at/median/val", halted_at.median(), on_step=True, on_epoch=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, targets = batch
        preds, p, halted_at = self(x)
        rec_loss, reg_loss = self.loss_function(preds, p, halted_at, targets)
        loss = rec_loss + reg_loss
        self.log("loss/rec_test", rec_loss)
        self.log("loss/reg_test", reg_loss)

        self.test_acc(self.reduce_preds(preds, halted_at), targets)
        self.log("loss/test", loss)
        self.log("acc/test", self.test_acc, on_step=False, on_epoch=True)

        halted_at = halted_at.float()
        self.log("halted_at/mean/test", halted_at.mean(), on_step=True, on_epoch=True)
        self.log("halted_at/std/test", halted_at.std(), on_step=True, on_epoch=True)
        self.log(
            "halted_at/median/test", halted_at.median(), on_step=True, on_epoch=True
        )
        return loss


class PonderMLP(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dims: list[int], out_dim: int, state_dim: int
    ):
        """
        Args:
            allow_early_return: Allow returning once all the halting variables
                from the batch landed on halt. Set this to `False` if your
                preds reduction method requires preds from all steps.
        """
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.state_dim = state_dim

        dims = [in_dim + state_dim] + hidden_dims + [out_dim + state_dim + 1]
        layers = [nn.Linear(dims[0], dims[1])]
        for in_, out_ in zip(dims[1:], dims[2:]):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(in_, out_))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, state=None):
        batch_size = x.size(0)

        if state is None:
            # Create initial state
            state = x.new_zeros(batch_size, self.state_dim)

        y_hat_n, state, lambda_n = self.layers(
            torch.concat((x, state), dim=1)
        ).tensor_split(
            indices=(
                self.out_dim,
                self.out_dim + self.state_dim,
            ),
            dim=1,
        )

        return y_hat_n, state, lambda_n


class PonderSequentialRNN(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, state_dim: int, rnn_type: str = "rnn"
    ):
        super().__init__()
        self.out_dim = out_dim
        self.state_dim = state_dim
        self.rnn_type = rnn_type.lower()

        total_out_dim = out_dim + 1

        rnn_cls = {
            "rnn": nn.RNN,
            "gru": nn.GRU,
            "lstm": nn.LSTM,
        }[self.rnn_type]

        self.rnn = rnn_cls(
            input_size=in_dim,
            hidden_size=state_dim,
            batch_first=True,
        )
        self.projection = nn.Linear(
            in_features=state_dim,
            out_features=total_out_dim,
        )

    def forward(self, x, state=None):
        x = x.unsqueeze(-1)

        _, state = self.rnn(x, state)
        state = F.relu(state)
        # LSTM state == (c_n, h_n)
        projection_state = state if self.rnn_type != "lstm" else state[1]
        y_hat_n, lambda_n = self.projection(projection_state.squeeze(0)).tensor_split(
            indices=(self.out_dim,),
            dim=1,
        )

        return y_hat_n, state, lambda_n


class PonderRNN(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, state_dim: int, rnn_type: str = "rnn"
    ):
        super().__init__()
        self.out_dim = out_dim
        self.state_dim = state_dim
        self.rnn_type = rnn_type.lower()

        total_out_dim = out_dim + 1

        rnn_cls = {
            "rnn": nn.RNNCell,
            "gru": nn.GRUCell,
            "lstm": nn.LSTMCell,
        }[self.rnn_type]

        self.rnn = rnn_cls(
            input_size=in_dim,
            hidden_size=state_dim,
        )
        self.projection = nn.Linear(
            in_features=state_dim,
            out_features=total_out_dim,
        )

    def forward(self, x, state=None):
        state = F.relu(self.rnn(x, state))
        # LSTM state == (c_n, h_n)
        projection_state = state if self.rnn_type != "lstm" else state[1]
        y_hat_n, lambda_n = self.projection(projection_state).tensor_split(
            indices=(self.out_dim,),
            dim=1,
        )

        return y_hat_n, state, lambda_n


# TODO: Needs to be refactored a lot
class PonderBayesianMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        state_dim: int,
        max_ponder_steps: int,
        ponder_epsilon: float,
        allow_early_return: bool = True,
    ):
        """
        Args:
            allow_early_return: Allow returning once all the halting variables
                from the batch landed on halt. Set this to `False` if your
                preds reduction method requires preds from all steps.
        """
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.state_dim = state_dim
        self.max_ponder_steps = max_ponder_steps
        self.ponder_epsilon = ponder_epsilon
        self.allow_early_return = allow_early_return

        total_out_dim = (
            out_dim + state_dim + 2
        )  # add two extra items for dimension for alpha and beta
        if hidden_dims:
            layers: list[nn.Module] = [nn.Linear(in_dim + state_dim, hidden_dims[0])]
            for in_, out in zip(hidden_dims[:-1], hidden_dims[1:]):
                layers += [nn.ReLU(), nn.Linear(in_, out)]
            layers += [nn.ReLU(), nn.Linear(hidden_dims[-1], total_out_dim)]
            self.layers = nn.Sequential(*layers)
        else:
            self.layers = nn.Sequential(nn.Linear(in_dim + state_dim, total_out_dim))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        # State that transfers across steps
        state = x.new_zeros(batch_size, self.state_dim)

        lambdas = []
        p = []  # Probabilities of halting at each step.
        cum_p_n = x.new_zeros(batch_size)  # Cumulative probability of halting.
        prob_not_halted_so_far = 1
        halted_at = x.new_zeros(batch_size)
        preds = []

        for n in range(1, self.max_ponder_steps + 1):
            # 1) Pass through model
            preds_n, state, lambda_params = self.layers(
                torch.concat((x, state), dim=1)
            ).tensor_split(
                indices=(
                    self.out_dim,
                    self.out_dim + self.state_dim,
                ),
                dim=1,
            )

            # 2) Sample lambda_n from beta-distribution
            lambda_params = F.relu(lambda_params) + 1e-7
            alpha, beta = lambda_params[:, 0], lambda_params[:, 1]
            distribution = dist_beta.Beta(alpha, beta)
            lambda_n = distribution.rsample()

            # 3) Store preds and probabilities
            preds.append(preds_n)
            lambdas.append(lambda_n)
            p.append(prob_not_halted_so_far * lambda_n)
            prob_not_halted_so_far = prob_not_halted_so_far * (1 - lambda_n)
            cum_p_n += p[n - 1]

            # 4) Update halted_at where needed (one-liner courtesy of jankrepl on GitHub)
            halted_at = (n * (halted_at == 0) * lambda_n.bernoulli()).max(halted_at)

            # If the probability is over epsilon we always stop.
            halted_at[
                (cum_p_n > (1 - self.ponder_epsilon)).bool() & (halted_at == 0)
            ] = n

            if self.allow_early_return and halted_at.all():
                break

        # Last step should be used if halting prob never reached above 1-epsilon
        halted_at[halted_at == 0] = self.max_ponder_steps
        lambdas_stacked = torch.stack(lambdas)
        p_stacked = torch.stack(p)  # (step, batch)
        preds_stacked = torch.stack(preds)  # (step, batch, logit)

        return {
            "halted_at": (halted_at - 1).long(),  # (batch) (zero-indexed)
            "lambdas": lambdas_stacked,  # (step, batch)
            "p": p_stacked,  # (step, batch)
            "preds": preds_stacked,  # (step, batch, logit)
        }
