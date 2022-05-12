import torch
from torch.utils.data import Dataset
import numpy as np
import os


class ParityDataset(Dataset):

    def __init__(self, filepath):
        self.problems, self.labels = torch.load(filepath)

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        return self.problems[idx], self.labels[idx]


def generate_parity_data(vector_size, num_problems):
    # TODO: Add extrapolation ability.

    problems = torch.zeros((num_problems, vector_size))
    labels = torch.zeros(num_problems)

    for index, problem in enumerate(problems):
        change_indices = torch.randint(vector_size, size=(vector_size,))
        problem[change_indices] = torch.Tensor(np.random.choice([-1, 1], len(change_indices), replace=True))

        # The label is 0 if the number of 1s is even, otherwise it is 1
        labels[index] = (problem == 1).sum() % 2

    return problems, labels


def save_parity_data(vector_size: int, num_problems: list, path: str):
    train_data = generate_parity_data(vector_size, num_problems[0])
    valid_data = generate_parity_data(vector_size, num_problems[1])
    test_data = generate_parity_data(vector_size, num_problems[2])

    torch.save(train_data, os.path.join(path, 'train.pt'))
    torch.save(valid_data, os.path.join(path, 'valid.pt'))
    torch.save(test_data, os.path.join(path, 'test.pt'))


def create_parity_dataloaders(path, batch_size, num_workers):
    train_dataset = ParityDataset(os.path.join(path, 'train.pt'))
    valid_dataset = ParityDataset(os.path.join(path, 'valid.pt'))
    test_dataset = ParityDataset(os.path.join(path, 'test.pt'))

    return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers), \
           torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers), \
           torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


if __name__ == '__main__':
    save_parity_data(vector_size=10, num_problems=[100000, 1000, 1000], path="./")
    train_loader, valid_loader, test_loader = create_parity_dataloaders("./", batch_size=32, num_workers=4)

    # Print the first batch of the training set
    for index, (problems, labels) in enumerate(train_loader):
        for problem, label in zip(problems, labels):
            print(problem, label)

        break
