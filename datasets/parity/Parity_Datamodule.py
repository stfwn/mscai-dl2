import inspect
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split

def generate_parity_data(vector_size, num_problems, min_integer_change=None, max_integer_change=None):
    """
    Generates a dataset of parity problems.
    :param vector_size: The size of the vectors to be generated.
    :param num_problems: The number of problems to be generated.
    :param min_integer_change: The minimum number of integers that should be changed to a -1 or 1.
    :param max_integer_change: The maximum number of integers that should be changed to a -1 or 1.
    :return: A tuple containing the problems of size (num_problems, vector_size) and the labels of size (num_problems,).
    """

    # If no min/max integer change is specified, assume we want the non extrapolation case.
    if min_integer_change is None or max_integer_change is None:
        min_integer_change = 1
        max_integer_change = vector_size

    problems = torch.zeros((num_problems, vector_size))
    labels = torch.zeros(num_problems)

    for index, problem in enumerate(problems):
        # Sample which indices should be changed from 0 to -1 or 1.
        change_indices = np.random.choice(np.arange(vector_size),
                                          size=np.random.randint(min_integer_change, max_integer_change + 1),
                                          replace=False)
        # Change the indices to -1 or 1.
        problem[change_indices] = torch.Tensor(np.random.choice([-1, 1], len(change_indices), replace=True))

        # The label is 0 if the number of 1s is even, otherwise it is 1.
        labels[index] = (problem == 1).sum() % 2

    return problems, labels.long()


class ParityDataset(Dataset):
    """
    PyTorch Datset for the parity problem as introduced in the original ACT paper (Graves, 2016).
    """

    def __init__(self, filepath):
        self.problems, self.labels = torch.load(filepath)

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        return self.problems[idx], self.labels[idx]


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
