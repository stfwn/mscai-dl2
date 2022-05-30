# stdlib
import os

# third party
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities import seed
import torch

# first party
import datamodules
import models
import utils


def main():
    seed.seed_everything(420)
    # datamodule = datamodules.TinyImageNet200DataModule(num_workers=os.cpu_count())
    datamodule = datamodules.FashionMNISTDataModule(num_workers=os.cpu_count())
    model = models.PonderNet(
        encoder=None,
        encoder_args=dict(
            variant=0,
        ),
        ##################
        # FMNIST setting #
        ##################
        step_function="bay_mlp",
        step_function_args=dict(
            in_dim=torch.tensor(datamodule.dims).prod().item(),  # 1
            out_dim=datamodule.num_classes,
            state_dim=100,
            hidden_dims=[300, 200],
        ),
        ###########################
        # TinyImageNet200 setting #
        ###########################
        # step_function="bay_rnn",
        # step_function_args=dict(
        #     # Fill in correct embedding dim if you're using an encoder.
        #     # tinyimagenet with b0: 1280
        #     in_dim=784,
        #     out_dim=datamodule.num_classes,
        #     state_dim=256,
        #     rnn_type="rnn",
        # ),
        beta_prior=(3, 3),
        max_ponder_steps=20,
        preds_reduction_method="bayesian_sampling",
        task="classification",
        learning_rate=0.001,
        scale_reg=0.01,
        ponder_epsilon=0.05,
        # Extra args just to log them
        dataset=type(datamodule).__name__,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        callbacks=[
            EarlyStopping(monitor="loss/val", patience=5),
            ModelCheckpoint(
                save_top_k=1,
                monitor="acc/val",
                mode="max",
            ),
            LearningRateMonitor(logging_interval="epoch"),
            utils.RegularizationWarmup(
                start=1e-8,
                slope=10,
                model_attr="regularization_warmup_factor",
            ),
        ],
        deterministic=True,
        devices="auto",
        logger=[
            TensorBoardLogger(
                "logs",
                default_hp_metric=True,
            ),
            WandbLogger(
                name=None,
                entity="mscai-dl2",
                project="mscai-dl2",
                log_model=True,
            ),
        ],
        max_epochs=50,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, ckpt_path="best", datamodule=datamodule)


if __name__ == "__main__":
    main()
