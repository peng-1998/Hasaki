import torch.nn.functional as F
from torch import Tensor


def dilate(map: Tensor, kernel: Tensor) -> Tensor:
    assert len(kernel.shape) == 2 and kernel.size(0) == kernel.size(1) and kernel.max() == 1 and kernel.size(0) % 2 == 1
    map = map.unsqueeze(1)
    kernel = kernel.to(map.device)
    return (1.0 * (F.conv2d(map.float(), kernel.unsqueeze(0).unsqueeze(0), stride=1, padding=int(kernel.size(0) / 2)) > 0)).squeeze(1)


def erosion(map: Tensor, kernel: Tensor) -> Tensor:
    return 1 - dilate(1 - map, kernel)
