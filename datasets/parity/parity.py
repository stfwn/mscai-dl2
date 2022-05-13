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


def generate_parity_data(vector_size, num_problems, min_integer_change=None, max_integer_change=None):

    # If no min/max integer change is specified, assume we want the non extrapolation case
    if min_integer_change is None or max_integer_change is None:
        min_integer_change = 1
        max_integer_change = vector_size

    problems = torch.zeros((num_problems, vector_size))
    labels = torch.zeros(num_problems)

    for index, problem in enumerate(problems):
        change_indices = np.random.choice(np.arange(vector_size),
                                          size=np.random.randint(min_integer_change, max_integer_change + 1),
                                          replace=False)
        problem[change_indices] = torch.Tensor(np.random.choice([-1, 1], len(change_indices), replace=True))

        # The label is 0 if the number of 1s is even, otherwise it is 1
        labels[index] = (problem == 1).sum() % 2

    return problems, labels


def save_parity_data(vector_size: int, num_problems: list, path: str, extrapolate: bool = False):
    if extrapolate:
        train_data = generate_parity_data(vector_size, num_problems[0], 1, vector_size // 2)
        valid_data = generate_parity_data(vector_size, num_problems[1], vector_size // 2 + 1, vector_size)
        test_data = generate_parity_data(vector_size, num_problems[2],  vector_size // 2 + 1, vector_size)
    else:
        train_data = generate_parity_data(vector_size, num_problems[0])
        valid_data = generate_parity_data(vector_size, num_problems[1])
        test_data = generate_parity_data(vector_size, num_problems[2])

    problem_str = f"{vector_size}" + ("_extrapolate" if extrapolate else "")

    torch.save(train_data, os.path.join(path, f"train_{problem_str}.pt"))
    torch.save(valid_data, os.path.join(path, f"valid_{problem_str}.pt"))
    torch.save(test_data, os.path.join(path, f"test_{problem_str}.pt"))


def create_parity_dataloaders(path, batch_size, num_workers, vector_size=48, extrapolate=False):
    problem_str = f"{vector_size}" + ("_extrapolate" if extrapolate else "")

    train_dataset = ParityDataset(os.path.join(path, f"train_{problem_str}.pt"))
    valid_dataset = ParityDataset(os.path.join(path, f"valid_{problem_str}.pt"))
    test_dataset = ParityDataset(os.path.join(path, f"test_{problem_str}.pt"))

    return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers), \
           torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers), \
           torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


if __name__ == '__main__':
    size_vector = 10
    num_probs = [100000, 10000, 10000]
    save_dir = "./"
    extrap = False

    save_parity_data(vector_size=size_vector, num_problems=num_probs, path=save_dir, extrapolate=extrap)
    train_loader, valid_loader, test_loader = create_parity_dataloaders("./", vector_size=size_vector, batch_size=32, num_workers=4, extrapolate=extrap)

    # Print the first batch of the training set
    print("Training set:")
    for index, (problems, labels) in enumerate(train_loader):
        for problem, label in zip(problems, labels):
            print(problem, label)

        break

    # Print the first batch of the validation set (to test if extrapolation is working)
    print("\n\nValidation set:")
    for index, (problems, labels) in enumerate(valid_loader):
        for problem, label in zip(problems, labels):
            print(problem, label)

        break

