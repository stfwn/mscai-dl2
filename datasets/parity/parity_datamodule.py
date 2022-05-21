import inspect
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from parity import generate_parity_data, ParityDataset

class Parity_Datamodule(pl.LightningDataModule):
    def __init__(self,
                num_problems, 
                vector_size,
                extrapolate,
                path,
                batch_size):
        super().__init__()
        
        for item in inspect.signature(DataModuleClass).parameters:
            setattr(self, item, eval(item))
        
        self.num_classes = 2
        self.dims = vector_size
        self.problem_str = f"{vector_size}" + ("_extrapolate" if extrapolate else "")
    
    def prepare_data(self):
        
        if self.extrapolate:
            train_data = generate_parity_data(self.vector_size, self.num_problems[0], 1, self.vector_size // 2)
            valid_data = generate_parity_data(self.vector_size, self.num_problems[1], self.vector_size // 2 + 1, self.vector_size)
            test_data = generate_parity_data(self.vector_size, self.num_problems[2],  self.vector_size // 2 + 1, self.vector_size)
            
        else:
            train_data = generate_parity_data(self.vector_size, self.num_problems[0])
            valid_data = generate_parity_data(self.vector_size, self.num_problems[1])
            test_data = generate_parity_data(self.vector_size, self.num_problems[2])

        torch.save(train_data, os.path.join(self.path, f"train_{self.problem_str}.pt"))
        torch.save(valid_data, os.path.join(self.path, f"valid_{self.problem_str}.pt"))
        torch.save(test_data, os.path.join(self.path, f"test_{self.problem_str}.pt"))
    
    def setup(self, stage=None):
        self.train_dataset = ParityDataset(os.path.join(self.path, f"train_{self.problem_str}.pt"))
        self.valid_dataset = ParityDataset(os.path.join(self.path, f"valid_{self.problem_str}.pt"))
        self.test_dataset = ParityDataset(os.path.join(self.path, f"test_{self.problem_str}.pt"))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size = self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size)
