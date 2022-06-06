# stdlib
import os

# third party
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# first party
import datamodules
import models
import utils


def main():
    seed = 421
    pl.seed_everything(seed)
    datamodule = datamodules.ParityDatamodule(
        path="./data/parity/",
        num_problems=(250_000, 25_000, 25_000),
        num_workers=os.cpu_count(),
        batch_size=128,
        vector_size=30,
        extrapolate=True,
        uniform=False,
    )
    model = models.PonderNet(
        step_function="rnn",
        step_function_args=dict(
            in_dim=torch.tensor(datamodule.dims).prod().item(),
            out_dim=datamodule.num_classes,
            state_dim=128,
            rnn_type="gru",
            activation="tanh",
        ),
        preds_reduction_method="ponder",
        task="classification",
        max_ponder_steps=10,
        learning_rate=3e-4,
        scale_reg=0.01,
        ponder_epsilon=0.05,
        # Extra args just to log them
        dataset=type(datamodule).__name__,
        seed=seed,
        regularization_warmup=False,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        callbacks=[
            ModelCheckpoint(
                save_top_k=1,
                monitor="acc/val",
                mode="max",
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        deterministic=True,
        devices="auto",
        logger=[
            TensorBoardLogger(
                "logs",
                default_hp_metric=True,
            ),
            WandbLogger(
                name="PonderNet",
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
