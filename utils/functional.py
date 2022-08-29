import torch
from torch import Tensor


def random_choices(array_or_length: Tensor | int, count: int, repeat=False) -> Tensor:
    """Create a random sample from array_or_length with count.

    Args:
        array_or_length (Tensor | int): Tensor randomly chosen from array or length for it.
        count (int): Number of samples to draw.
        repeat (bool, optional): Whether allow multiple occurrences of the same element. Defaults to False.

    Returns:
        Tensor: Index of samples if array_or_length is int, otherwise Tensor of samples.
    """

    def _random_choices(length=0, count=0, repeat=False):
        assert count >= 0 and length >= 0
        if repeat:
            return torch.randint(0, length, count, dtype=torch.long)
        return torch.randperm(length, dtype=torch.long)[:count]

    if isinstance(array_or_length, torch.Tensor):
        return array_or_length[_random_choices(array_or_length.shape[0], count=count, repeat=repeat).to(array_or_length.device)]
    return _random_choices(array_or_length, count=count, repeat=repeat)
