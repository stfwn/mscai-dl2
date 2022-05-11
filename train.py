# stdlib
import os
from argparse import ArgumentParser

# third party
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# first party
import datamodules
import models


def main(args):
    datamodule = datamodules.FashionMNISTDataModule(
        data_dir="./data", num_workers=os.cpu_count()
    )
    model = models.PonderNet(
        encoder=None,
        encoding_dim=torch.tensor(datamodule.dims).prod(),
        step_function="mlp",
        step_function_args={
            "hidden_dims": [300, 200],
            "state_dim": 100,
            "ponder_epsilon": 0.05,
        },
        max_ponder_steps=10,
        preds_reduction_method="bayesian",
        out_dim=datamodule.num_classes,
        task="classification",
    )
    trainer = pl.Trainer(
        accelerator="auto",
        callbacks=[
            ModelCheckpoint(
                save_top_k=1,
                monitor="acc/val",
                mode="max",
                filename=args.model_name,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        deterministic=True,
        devices="auto",
        logger=[
            TensorBoardLogger(
                "logs",
                name=args.model_name,
                default_hp_metric=False,
                # log_graph=True,
            ),
            WandbLogger(
                project="mscai-dl2",
                log_model=True,
            ),
        ],
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-m", "--model-name", type=str, choices=["pondermlp"], default="pondermlp"
    )
    args = parser.parse_args()
    main(args)
