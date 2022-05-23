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
    pl.seed_everything(420, workers=True)
    datamodule = datamodules.FashionMNISTDataModule(
        data_dir="./data", num_workers=args.num_workers
    )
    model = models.PonderNet(
        encoder=None,
        encoding_dim=torch.tensor(datamodule.dims).prod(),
        step_function=args.step_function,
        step_function_args={
            "hidden_dims": [300, 200],
            "state_dim": 100,
            "ponder_epsilon": 0.05,
        },
        max_ponder_steps=10,
        preds_reduction_method=args.preds_reduction_method,
        out_dim=datamodule.num_classes,
        learning_rate=args.learning_rate,
        task=args.task,
        loss_beta=args.loss_beta,
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
    parser.add_argument(
        "-s",
        "--step-function",
        type=str,
        choices=["bayesian_mlp", "mlp"],
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        choices=["classification", "bayesian_classification"],
        help="Used to determine loss function during training.",
    )
    parser.add_argument(
        "-pr",
        "--preds-reduction-method",
        choices=["ponder", "bayesian"],
        type=str,
        help="Method to use for reducing the predictions for each ponder step to a single value.",
    )
    parser.add_argument(
        "--num-workers",
        default=os.cpu_count(),
        type=int,
        help="Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0.",
    )
    parser.add_argument(
        "-lb",
        "--loss-beta",
        default=0.01,
        type=float,
        help="Factor by which to multiply the regularization loss term",
    )
    parser.add_argument(
        "-lr", "--learning-rate", default=3e-4, type=float, help="Learning rate."
    )

    args = parser.parse_args()
    main(args)
