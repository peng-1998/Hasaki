import torch
from torch import Tensor
import torch.nn.functional as F


def dilate(map: Tensor, kernel: Tensor) -> Tensor:
    assert len(kernel.shape) == 2 and kernel.size(0) == kernel.size(1) \
        and kernel.max() == 1 and kernel.size(0) % 2 == 1
    map = map.unsqueeze(1)
    kernel = kernel.to(map.device)
    return (1.0*(F.conv2d(map, kernel.unsqueeze(0).unsqueeze(0), stride=1, padding=int(kernel.size(0)/2)) > 0)).square(1)


def erosion(map: Tensor, kernel: Tensor) -> Tensor:
    return 1 - dilate(1 - map, kernel)
