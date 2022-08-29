import math
from typing import List, Tuple

import numpy
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms.functional import InterpolationMode


def _get_image_size(img: Tensor) -> List[int]:
    return img.shape[-3:]


def crop(img: Tensor, z, deep, y, height, x, width) -> Tensor:
    return img[..., z:z + deep, y:y + height, x:x + width]


def to_numpy(img: Tensor) -> numpy.ndarray:
    return img.permute(3, 2, 1, 0).cpu().numpy()


def to_tensor(img: numpy.ndarray) -> Tensor:
    if len(img.shape) == 4:
        return torch.tensor(img).permute(3, 2, 1, 0).contiguous()
    return to_tensor(torch.tensor(img).unsqueeze(3))


def resize(img: Tensor, size: Tuple[int, int, int], interpolation: str = InterpolationMode.BILINEAR) -> Tensor:
    """Resize an 3D image to the given size.

    Args:
        img (Tensor): Image to be resized. Expected to be of shape (C, D, H, W) or (D, H, W).
        size (Tuple[int, int, int]): Desired output size. Expected to be of Tuple (D, H, W).
        interpolation (str, optional): Interpolation method. Defaults to InterpolationMode.BILINEAR.

    Returns:
        Tensor: Resized image.
    """
    assert len(img.shape) == 4 and len(size) == 3
    deep, height, width = _get_image_size(img)
    m = torch.tensor([[[size[0] / deep, 0, 0, 0], [0, size[1] / height, 0, 0], [0, 0, size[2] / width, 0]]])
    grid = F.affine_grid(m, size, True)
    if interpolation == "bilinear":
        mode_enum = 0
    elif interpolation == "nearest":
        mode_enum = 1
    else:  # mode == 'bicubic'
        mode_enum = 2
    return torch.grid_sampler(img.unsqueeze(0), grid, mode_enum, 0, True)[0]


def rotate(img: Tensor, angles: Tuple[float, float, float], interpolation: str = InterpolationMode.BILINEAR, expand: bool = False) -> Tensor:
    """Rotate an 3D image by the given angles.

    Args:
        img (Tensor): Image to be rotated. Expected to be of shape (C, D, H, W) or (D, H, W).
        angles (Tuple[float, float, float]): Rotate angles for axis D,H,W. angle % 360.
        interpolation (str, optional): Interpolation method. Defaults to InterpolationMode.BILINEAR.
        expand (bool, optional): Expand image to include all all pix. Defaults to False.

    Returns:
        Tensor: Rotated image.
    """    

    s, c = math.sin(angles[0] * math.pi / 180), math.cos(angles[0] * math.pi / 180)
    m1 = torch.tensor([[0, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])
    s, c = math.sin(angles[1] * math.pi / 180), math.cos(angles[1] * math.pi / 180)
    m2 = torch.tensor([[c, 0, -s, 0], [0, 0, 0, 0], [s, 0, c, 0], [0, 0, 0, 1]])
    s, c = math.sin(angles[2] * math.pi / 180), math.cos(angles[2] * math.pi / 180)
    m3 = torch.tensor([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
    m = torch.mm(m1, torch.mm(m2, m3))
    v = _get_image_size(img)
    if expand:
        d, h, w = _get_image_size(img)
        v = torch.tensor([[w / 2], [h / 2], [d / 2], 1])
        v = torch.floor(torch.mm(m, v)) + 1
        v = torch.stack([v[2], v[1], v[0]]) * 2

    if interpolation == InterpolationMode.BILINEAR:
        mode = 0
    elif interpolation == InterpolationMode.NEAREST:
        mode = 1
    else:
        mode = 2
    m = m[0:3].unsqueeze(0)
    grid = F.affine_grid(m, v, True)
    return torch.grid_sampler(img.unsqueeze(0), grid, mode, 0, True)[0]
