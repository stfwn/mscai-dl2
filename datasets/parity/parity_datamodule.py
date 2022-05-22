import inspect
import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from parity import ParityDataset, save_parity_data


class ParityDatamodule(pl.LightningDataModule):
    def __init__(self, num_problems, vector_size, extrapolate, path, batch_size):
        super().__init__()

        for item in inspect.signature(ParityDatamodule).parameters:
            setattr(self, item, eval(item))

        self.num_classes = 2
        self.dims = vector_size
        self.problem_str = f"{vector_size}{'_extrapolate' if extrapolate else ''}"

    def prepare_data(self):
        save_parity_data(
            vector_size=self.vector_size,
            num_problems=self.num_problems,
            path=self.path,
            extrapolate=self.extrapolate,
        )

    def setup(self, stage=None):
        self.train_dataset = ParityDataset(
            os.path.join(self.path, f"train_{self.problem_str}.pt")
        )
        self.valid_dataset = ParityDataset(
            os.path.join(self.path, f"valid_{self.problem_str}.pt")
        )
        self.test_dataset = ParityDataset(
            os.path.join(self.path, f"test_{self.problem_str}.pt")
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
