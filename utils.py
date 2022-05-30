import torch


def calculate_beta_std(alphas: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
    """
    Calculate the standard deviation of the beta distribution.

    :return: Standard deviation of the beta distribution.
    """
    variance = (alphas * betas) / ((alphas + betas) ** 2 * (alphas + betas + 1))
    return torch.sqrt(variance)


def mode_agreement_metric(samples_mode, samples):
    """
    Calculate the mode agreement metric.

    :param samples_mode: The mode of each sample. (batch_size)
    :param samples: The samples. (batch_size, samples)
    :return: The normalized agreement between the samples and the mode. (float)
    """
    return (
        (samples == samples_mode.unsqueeze(1)).float().sum(1) / samples.size(1)
    ).mean()
