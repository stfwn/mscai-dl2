from typing import Literal, Optional

import torch
import torch.distributions.geometric as dist_geometric
import torch.distributions.beta as dist_beta
import torch.distributions.categorical as dist_categorical
import torch.nn.functional as F
import torchmetrics
import torchvision
from pytorch_lightning import LightningModule
from torch import nn, optim

from loss_functions import PonderBayesianLoss, PonderLoss

# First party
from utils import *

class PonderNet(LightningModule):
    def __init__(
        self,
        step_function: Literal["mlp", "rnn", "seq_rnn", "bay_mlp", "bay_rnn"],
        step_function_args: dict,
        task: Literal["classification", "bayesian-classification"],
        max_ponder_steps: int,
        preds_reduction_method: Literal["ponder", "bayesian", "bayesian_sampling"] = "ponder",
        encoder: Optional[Literal["efficientnet"]] = None,
        encoder_args: Optional[dict] = None,
        learning_rate: float = 3e-4,
        lambda_prior: float = 0.2,
        scale_reg: float = 0.01,
        ponder_epsilon: float = 0.05,
        fixed_ponder_steps: int = 0,
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

            fixed_ponder_steps: One-indexed int to set an exact number of steps
                that the network should ponder for. This overrides halting
                probabilities and max ponder steps. It's useful for configuring
                baseline experiments.

                - 0 (default): Don't fix the number of steps.
                - Positive int: Ponder for exactly this number of steps.
        """
        super().__init__()
        self.save_hyperparameters()

        # Metrics
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

        # Encoder
        if encoder:
            encoder_class = {
                "efficientnet": EfficientNetEncoder,
            }.get(encoder)
            if not encoder_class:
                raise ValueError(f"Unknown encoder: '{encoder}'")
            self.encoder = encoder_class(**encoder_args)
        else:
            self.encoder = lambda x: x  # type: ignore

        # Step function
        self.allow_early_return = preds_reduction_method == "ponder" and "bayesian" not in task
        sf_class = {
            "mlp": PonderMLP,
            "rnn": PonderRNN,
            "seq_rnn": PonderSequentialRNN,
            "bay_mlp": PonderBayesianMLP,
            "bay_rnn": PonderBayesianRNN,
        }.get(step_function)
        if not sf_class:
            raise ValueError(f"Unknown step function: '{step_function}'")
        self.step_function = sf_class(**step_function_args)

        # Loss
        lf_class = {
            "classification": (
                lambda: PonderLoss(
                    task_loss_fn=F.cross_entropy,
                    scale_reg=scale_reg,
                    lambda_prior=lambda_prior,
                    max_ponder_steps=max_ponder_steps,
                )
            ),
            "bayesian-classification": (
                lambda: PonderBayesianLoss(
                    task_loss_fn=F.cross_entropy,
                    beta_prior=(10, 10),
                    max_ponder_steps=max_ponder_steps,
                    scale_reg=scale_reg,
                )
            ),
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
            "bayesian_sampling": self.reduce_preds_bayesian_sampling,
        }.get(preds_reduction_method)
        if not preds_reduction_fn:
            raise ValueError(
                f"Unknown preds reduction method: '{preds_reduction_method}'"
            )
        self.preds_reduction_fn = preds_reduction_fn

        # Optimizer
        self.optimizer_class = optim.Adam

        if fixed_ponder_steps and loss_beta != 0:
            raise ValueError(
                "Using a fixed number of ponder steps with a non-zero KL multiplier (loss_beta) does not make sense."
            )

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            params=self.parameters(), lr=self.hparams.learning_rate
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.1
        )
        return [optimizer], [{"scheduler": scheduler, "monitor": "loss/val"}]

    def reduce_preds(self, preds, halted_at, p, beta_params):
        """
        Get the final predictions where the network decided to halt.

        Args:
            preds: (step, batch, logit)
            halted_at: (batch)

        Returns:
            (batch, logit)
        """
        return self.preds_reduction_fn(preds, halted_at, p, beta_params)

    @staticmethod
    def reduce_preds_ponder(preds, halted_at, p):
        """
        Reduces predictons from multiple ponder steps to one prediction,
        using halted_at.
        out_dict: dictionary containing:
                preds: (ponder_steps, batch_size, logits)
                p: halting probability (ponder_steps, batch_size)
        :return: predictions (batch_size, logits)
        """
        return preds.permute(1, 2, 0)[torch.arange(preds.size(1)), :, halted_at], None

    @staticmethod
    def reduce_preds_bayesian(preds, halted_at, p):
        """
        Reduces predictons from multiple ponder steps to one prediction,
        using weighted average.
        out_dict: dictionary containing:
                preds: (ponder_steps, batch_size, logits)
                p: halting probability (ponder_steps, batch_size)
        :return: predictions (batch_size, logits)
        """
        return torch.einsum("sbl,sb->bl", preds, p), None

    @staticmethod
    def reduce_preds_bayesian_sampling(preds, halted_at, p, beta_params):
        """
        Reduces predictons from multiple ponder steps to one prediction,
        using mode of multiple sampled predictions.
        (sampling lambda --> halted_at --> prediction)
        :param preds: (ponder_steps, batch_size, logits)
        :param halted_at:
        :param p:
        :param alpha:
        :param beta: ()
        :return: predictions (batch_size)
        """
        n_samples = 100  # TODO: Make hyperparam
        alpha, beta = beta_params

        # 1) Sample lambdas
        # lambdas = dist_beta.Beta(alpha, beta).sample()  # get tensor (step, batch)
        lambdas = dist_beta.Beta(alpha, beta).sample((n_samples,)).to(preds.device).permute(1,0,2) # (step, sample, batch)

        lambdas[-1,:,:] = 1

        # 2) Calculate halting probabilities per step
        p = []  # Probabilities of halting at each step.
        prob_not_halted_so_far = 1 # Probabilities of not having halted so far.

        for lambda_n in lambdas: # lambda_n (sample, batch)
            p.append(prob_not_halted_so_far * lambda_n)
            prob_not_halted_so_far = prob_not_halted_so_far * (1 - lambda_n)

            # TODO: do we need this??
            # # If the probability is over epsilon we always stop.
            # halted_at[
            #     torch.logical_and(
            #         (cum_p_n > (1 - self.hparams.ponder_epsilon)), (halted_at == 0)
            #     )
            # ] = n

        # 3) Sample the halted at step.
        p = torch.stack(p).to(preds.device)  # (step, sample, batch_size)
        halted_at = torch.distributions.categorical.Categorical(probs=p.permute(2, 1, 0)).sample().to(preds.device)  # (batch_size, sample)

        # Index the preds using the halting_at which is of size (batch_size, sample)

        #final_preds = preds.permute(1, 2, 0)[torch.arange(preds.size(1)), :, halted_at]  # (batch_size, num_classes)

        # 4) Get prediction scores of each halted step
        final_preds = torch.zeros((n_samples, preds.size(1), preds.size(2))).to(preds.device)  # (samples, batch_size, num_classes)

        # Unvectorized code
        # TODO: Please help vectorizing this
        for i in range(n_samples):
            # preds.permute(1, 2, 0) -> (batch_size, num_classes, ponder_steps)
            final_preds[i, :, :] = preds.permute(1, 2, 0)[torch.arange(preds.size(1)), :, halted_at[:,i]]

        # Our idea: but doesn't work
        # final_preds = preds.permute(1, 2, 0)[torch.arange(n_samples).expand(preds.size(1), -1), :, halted_at] # (batch_size, steps, num_classes)

        # 5) Sample the class predictions:
        final_preds_logits = final_preds.log_softmax(dim=2)  # (sample, batch_size, num_classes)
        sampled_class_preds = torch.distributions.categorical.Categorical(logits=final_preds_logits.permute(1, 0, 2))\
            .sample().to(preds.device)  # (batch_size, samples)

        return sampled_class_preds.mode(1)[0], sampled_class_preds  # (batch_size), (batch_size, samples)


    def forward(self, x):
        batch_size = x.size(0)
        max_steps = self.hparams.fixed_ponder_steps or (self.hparams.max_ponder_steps)
        state = None  # State that transfers across steps

        x = self.encoder(x)

        # Probabilities of not having halted so far.
        p = []  # Probabilities of halting at each step.
        cum_p_n = x.new_zeros(batch_size)  # Cumulative probability of halting.
        prob_not_halted_so_far = 1
        halted_at = x.new_zeros(batch_size)
        y_hat = []
        lambdas = []
        alphas, betas = [], []

        for n in range(1, max_steps + 1):
            # 1) Pass through model
            y_hat_n, state, lambda_n, beta_params = self.step_function(x, state)

            alphas.append(beta_params[0])
            betas.append(beta_params[1])

            lambdas.append(lambda_n)
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
            if (
                self.allow_early_return
                and halted_at.all()
                and not self.hparams.fixed_ponder_steps
            ):
                break

        # Last step should be used if halting prob never reached above 1-epsilon
        halted_at[halted_at == 0] = max_steps

        # Normalize p so it's an actual distribution
        p = torch.stack(p)
        p = p / p.sum(0)

        if self.hparams.fixed_ponder_steps:
            halted_at[:] = self.hparams.fixed_ponder_steps
            p[:-1, :] = 0
            p[-1, :] = 1

        return (
            torch.stack(y_hat),  # (step, batch, logit)
            p,  # (step, batch)
            (halted_at - 1).long(),  # (batch)
            torch.stack(lambdas),  # (step, batch)
            (torch.stack(alphas), torch.stack(betas)),  # (step, batch), (step, batch)
        )

    def training_step(self, batch, batch_idx):
        x, targets = batch
        preds, p, halted_at, lambdas, (alphas, betas) = self(x)
        rec_loss, reg_loss = self.loss_function(preds, p, halted_at, targets, lambdas=lambdas, beta_params=(alphas, betas))
        loss = rec_loss + reg_loss
        self.log("loss/rec_train", rec_loss)
        self.log("loss/reg_train", reg_loss)

        final_preds, samples_final_preds = self.reduce_preds(preds, halted_at, p, (alphas, betas))

        if samples_final_preds is not None:
            agreement_result = mode_agreement_metric(final_preds, samples_final_preds)
            self.log("agreement_preds/train", agreement_result, on_step=True, on_epoch=True)

        self.train_acc(final_preds, targets)
        self.log("acc/train", self.train_acc, on_step=False, on_epoch=True)
        self.log("loss/train", loss)

        lambdas = lambdas.float()
        self.log(
            "lambda/first/train",
            lambdas[0, :].mean(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "lambda/first/train_std",
            lambdas[0, :].std(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "lambda/last/train",
            lambdas[-1, :].mean(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "lambda/last/train_std",
            lambdas[-1, :].std(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "beta_std/first/train",
            calculate_beta_std(alphas[0],betas[0]).mean(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "beta_std/last/train",
            calculate_beta_std(alphas[-1],betas[-1]).mean(),
            on_step=True,
            on_epoch=True,
        )

        halted_at = halted_at.float()
        self.log("halted_at/mean/train", halted_at.mean(), on_step=True, on_epoch=True)
        self.log("halted_at/std/train", halted_at.std(), on_step=True, on_epoch=True)
        self.log(
            "halted_at/median/train", halted_at.median(), on_step=True, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        preds, p, halted_at, lambdas, (alphas, betas) = self(x)
        rec_loss, reg_loss = self.loss_function(preds, p, halted_at, targets, lambdas=lambdas, beta_params=(alphas, betas))
        loss = rec_loss + reg_loss
        self.log("loss/rec_val", rec_loss)
        self.log("loss/reg_val", reg_loss)

        final_preds, samples_final_preds = self.reduce_preds(preds, halted_at, p, (alphas, betas))

        if samples_final_preds is not None:
            agreement_result = mode_agreement_metric(final_preds, samples_final_preds)
            self.log("agreement_preds/val", agreement_result, on_step=True, on_epoch=True)

        self.val_acc(final_preds, targets)
        self.log("loss/val", loss)
        self.log("acc/val", self.val_acc, on_step=False, on_epoch=True)

        lambdas = lambdas.float()
        self.log(
            "lambda/first/val",
            lambdas[0, :].mean(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "lambda/first/valid_std",
            lambdas[0, :].std(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "lambda/last/val",
            lambdas[-1, :].mean(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "lambda/last/valid_std",
            lambdas[-1, :].std(),
            on_step=True,
            on_epoch=True,
        )

        self.log(
            "beta_std/first/valid",
            calculate_beta_std(alphas[0], betas[0]).mean(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "beta_std/last/valid",
            calculate_beta_std(alphas[-1], betas[-1]).mean(),
            on_step=True,
            on_epoch=True,
        )

        halted_at = halted_at.float()
        self.log("halted_at/mean/val", halted_at.mean(), on_step=True, on_epoch=True)
        self.log("halted_at/std/val", halted_at.std(), on_step=True, on_epoch=True)
        self.log(
            "halted_at/median/val", halted_at.median(), on_step=True, on_epoch=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, targets = batch
        preds, p, halted_at, lambdas, (alphas, betas) = self(x)
        rec_loss, reg_loss = self.loss_function(preds, p, halted_at, targets, lambdas=lambdas, beta_params=(alphas, betas))
        loss = rec_loss + reg_loss
        self.log("loss/rec_test", rec_loss)
        self.log("loss/reg_test", reg_loss)

        final_preds, samples_final_preds = self.reduce_preds(preds, halted_at, p, (alphas, betas))

        if samples_final_preds is not None:
            agreement_result = mode_agreement_metric(final_preds, samples_final_preds)
            self.log("agreement_preds/test", agreement_result, on_step=True, on_epoch=True)

        self.test_acc(final_preds, targets)
        self.log("loss/test", loss)
        self.log("acc/test", self.test_acc, on_step=False, on_epoch=True)

        lambdas = lambdas.float()
        self.log(
            "lambda/first/test",
            lambdas[0, :].mean(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "lambda/first/test_std",
            lambdas[0, :].std(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "lambda/last/test",
            lambdas[-1, :].mean(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "lambda/last/test_std",
            lambdas[-1, :].std(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "beta_std/first/test",
            calculate_beta_std(alphas[0], betas[0]).mean(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "beta_std/last/test",
            calculate_beta_std(alphas[-1], betas[-1]).mean(),
            on_step=True,
            on_epoch=True,
        )

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
        x = x.view(batch_size, -1)

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

        lambda_n = lambda_n.squeeze().sigmoid()

        return y_hat_n, state, lambda_n, None


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

        lambda_n = lambda_n.squeeze().sigmoid()

        return y_hat_n, state, lambda_n, None


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

        lambda_n = lambda_n.squeeze().sigmoid()

        return y_hat_n, state, lambda_n, None

class PonderBayesianRNN(nn.Module):
    def __init__(
            self, in_dim: int, out_dim: int, state_dim: int, rnn_type: str = "rnn"
    ):
        super().__init__()
        self.out_dim = out_dim
        self.state_dim = state_dim
        self.rnn_type = rnn_type.lower()

        total_out_dim = out_dim + 2

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
        y_hat_n, lambda_params = self.projection(projection_state).tensor_split(
            indices=(self.out_dim,),
            dim=1,
        )

        # 2) Sample lambda_n from beta-distribution
        lambda_params = F.relu(lambda_params) + 1e-7
        alpha, beta = lambda_params[:, 0], lambda_params[:, 1]
        distribution = dist_beta.Beta(alpha, beta)
        lambda_n = distribution.rsample()

        return y_hat_n, state, lambda_n, (alpha, beta)

class PonderBayesianMLP(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dims: list[int],
            out_dim: int,
            state_dim: int,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.state_dim = state_dim

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

    def forward(self, x, state=None):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        if state is None:
            # Create initial state
            state = x.new_zeros(batch_size, self.state_dim)

        y_hat_n, state, lambda_params = self.layers(
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

        return y_hat_n, state, lambda_n, (alpha, beta)


class EfficientNetEncoder(nn.Module):
    def __init__(self, variant: int):
        assert 0 <= variant <= 7
        super().__init__()
        self.model = getattr(
            torchvision.models,
            f"efficientnet_b{variant}",
        )(pretrained=True, stochastic_depth_prob=0)
        self.model.classifier = nn.Identity()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)
