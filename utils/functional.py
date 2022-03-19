import torch


def random_choices(array_or_length: torch.Tensor | int, count: int, repeat=False) -> torch.Tensor:
    '''
    Args:
        array_or_length:Tensor randomly chosen from  or length for it
        count:The count of return indexes
        repeat: Whether the indexes can be repeated
    Return:
        Tensor of indexes with type long and same device with array_or_length if array_or_length is int \\
        Tensor of data chosen from array_or_length if array_or_length is Tensor
    '''

    def _random_choices(length=0, count=0, repeat=False):
        assert count >= 0 and length >= 0
        if repeat:
            return torch.randint(0, length, count, dtype=torch.long)
        return torch.randperm(length, dtype=torch.long)[:count]

    if isinstance(array_or_length, torch.Tensor):
        return array_or_length[_random_choices(array_or_length.shape[0], count=count, repeat=repeat).to(array_or_length.device)]
    return _random_choices(array_or_length, count=count, repeat=repeat)
