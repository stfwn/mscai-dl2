# stdlib
import os
from argparse import ArgumentParser

# third party
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities import seed

# first party
import datamodules
import models


def main(args):
    seed.seed_everything(420)
    # datamodule = datamodules.FashionMNISTDataModule(
    #     data_dir="./data", num_workers=os.cpu_count(), batch_size=256
    # )
    datamodule = datamodules.ParityDatamodule(
        path="./data/parity/",
        num_problems=(100000, 10000, 10000),
        num_workers=os.cpu_count(),
        batch_size=128,
        vector_size=20,
    )
    model = models.PonderNet(
        encoder=None,
        step_function="rnn",
        step_function_args=dict(
            in_dim=torch.tensor(datamodule.dims).prod(),  # 1
            out_dim=datamodule.num_classes,
            state_dim=128,
            rnn_type="gru",
            # hidden_dims=[300, 200],
        ),
        max_ponder_steps=20,
        preds_reduction_method="ponder",
        task="classification",
        learning_rate=0.001,
        loss_beta=0.01,
        lambda_prior=0.2,
        ponder_epsilon=0.05,
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
        gradient_clip_val=0.5,
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
                name=None,
                project="mscai-dl2",
                log_model=True,
            ),
        ],
        max_epochs=50,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(datamodule=datamodule)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-m", "--model-name", type=str, choices=["pondermlp"], default="pondermlp"
    )
    args = parser.parse_args()
    main(args)
