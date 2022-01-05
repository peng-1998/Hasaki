from typing import List, Sequence, Tuple, Union
import warnings
import torch
from torch.functional import Tensor
from torch.nn import Module
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int
from . import functional3D as TF3D
import torchvision.transforms as T
import random

'for all 3D images dim = (channel,deep,height,width)'


class RandomCrop(Module):
    def __init__(self, size: Tuple[int, int, int], padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()
        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, *args):
        result = [_ for _ in args]
        if self.padding is not None:
            result = [F.pad(_, self.padding, self.fill, self.padding_mode) for _ in result]

        deep, height, width = TF3D._get_image_size(result[0])
        # pad the width if needed
        if self.pad_if_needed and width < self.size[2]:
            padding = [0, 0, 0, 0, int((self.size[2] - width) / 2), self.size[2] - width - int((self.size[2] - width) / 2)]
            result = [F.pad(_, padding, self.fill, self.padding_mode) for _ in result]
        # pad the height if needed
        if self.pad_if_needed and height < self.size[1]:
            padding = [0, 0, int((self.size[1] - height) / 2), self.size[1] - height - int((self.size[1] - height) / 2), 0, 0]
            result = [F.pad(_, padding, self.fill, self.padding_mode) for _ in result]
        # pad the deep if needed
        if self.pad_if_needed and deep < self.size[0]:
            padding = [int((self.size[0] - deep) / 2), self.size[0] - deep - int((self.size[0] - deep) / 2), 0, 0, 0, 0]
            result = [F.pad(_, padding, self.fill, self.padding_mode) for _ in result]
        d, h, w = self.size
        x = random.randint(0, width - w)
        y = random.randint(0, height - h)
        z = random.randint(0, deep - d)
        return [TF3D.crop(_, z, d, y, h, x, w) for _ in result]


class RandomVerticalFlip(T.RandomHorizontalFlip):
    def forward(self, *args) -> Union[tuple, List]:
        if torch.rand(1) < self.p:
            return [_.flip(-1) for _ in args]
        return args


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, *args) -> Union[tuple, List]:
        if torch.rand(1) < self.p:
            return [_.flip(-1) for _ in args]
        return args


class RandomDeepFlip(T.RandomHorizontalFlip):
    def forward(self, *args) -> Union[tuple, List]:
        if torch.rand(1) < self.p:
            return [_.flip(-3) for _ in args]
        return args


class RandomResizedCrop(Module):
    def __init__(self, size, scale=(0.08, 1.0), interpolation=InterpolationMode.BILINEAR,interpolations=[]):
        super().__init__()
        self.size = size
        self.interpolations = interpolations

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn("Argument interpolation should be of type InterpolationMode instead of int. " "Please, use InterpolationMode enum.")
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.scale = scale

    def forward(self, *args) -> list:
        deep, height, width = TF3D._get_image_size(args[0])
        w = random.randint(int(self.scale[0] * width) + 1, int(self.scale[1] * width))
        h = random.randint(int(self.scale[0] * height) + 1, int(self.scale[1] * height))
        d = random.randint(int(self.scale[0] * deep) + 1, int(self.scale[1] * deep))
        x = random.randint(0, width - w)
        y = random.randint(0, height - h)
        z = random.randint(0, deep - d)
        if len(self.interpolations) == 0:
            return [TF3D.resize(TF3D.crop(_, z, d, y, h, x, w), self.size,self.interpolation) for _ in args]
        else:
            return [TF3D.resize(TF3D.crop(_, z, d, y, h, x, w), self.size,interpolation) for _,interpolation in zip(args,self.interpolations)]


class RandomRotation(Module):
    """
    Random rotate 3D objects at 3 axis:x,y,z
    """
    def __init__(self, degrees: Tuple[int, int, int], interpolation=InterpolationMode.BILINEAR, expand=False,interpolations=[]) -> None:
        """
        Args::
            degrees:The max value for rotate angel in x,y,z axis.Use [360,360,360] for any angle.
            interpolation:Interpolation mode.
            expand:It will expand the size of tensor to contains the image after rotated.
        """

        super().__init__()
        self.degrees       = degrees
        self.interpolation = interpolation
        self.expand        = expand
        self.interpolations = interpolations

    def forward(self, *args) -> list[Tensor]:
        x_theta = random.random() * self.degrees[0]
        y_theta = random.random() * self.degrees[0]
        z_theta = random.random() * self.degrees[0]
        if len(self.interpolations):
            return [TF3D.rotate(_, (x_theta, y_theta, z_theta), self.interpolation, expand=self.expand) for _ in args]
        else:
            return [TF3D.rotate(_, (x_theta, y_theta, z_theta), interpolation, expand=self.expand) for _,interpolation in zip(args,self.interpolations)]

class ElasticDeformation(Module):
    """
    A random elastic deformation transformation which applied to the image.
    """
    def __init__(self, grids_size: Tuple[int, int, int], sigma: float,interpolation=InterpolationMode.BILINEAR,interpolations=[]):
        """
        Args::
            grids_size:The shape of grid which guide elastic deformation transformation.The smaller grid there is the smoother the deformation changes.
            sigma:The sigma of an normal distribution.The bigger sigma is the more intense change will be.
        """
        super().__init__()
        self.grids_size = list(grids_size)
        self.sigma = sigma
        self.interpolation = interpolation
        self.interpolations = interpolations

    def forward(self, *args) -> list:
        d, h, w = TF3D._get_image_size(args[0])
        grid = self._getgrid(d, h, w)
        if len(self.interpolations)==0:
            return [self._elasticdeformation(_, grid,self.interpolation) for _ in args]    
        else:
            return [self._elasticdeformation(_, grid,interpolation) for _,interpolation in zip(args,self.interpolations)]

    def _getgrid(self, d, h, w):
        grid = self.sigma * torch.randn([3] + self.grids_size)
        grid = TF3D.resize(grid, (d, h, w)).permute(1, 2, 3, 0).unsqueeze(0)
        grid[..., 0] = grid[..., 0] * 2 / w
        grid[..., 1] = grid[..., 1] * 2 / h
        grid[..., 1] = grid[..., 1] * 2 / d
        x, y, z = torch.linspace(-1, 1, w), torch.linspace(-1, 1, h), torch.linspace(-1, 1, d)
        xy = torch.stack(torch.meshgrid(x, y, z)).permute(3, 2, 1, 0).unsqueeze(0)
        return grid + xy

    def _elasticdeformation(self, img: Tensor, grid: Tensor,interpolation):
        if interpolation == "bilinear":
            mode_enum = 0
        elif interpolation == "nearest":
            mode_enum = 1
        else:  # mode == 'bicubic'
            mode_enum = 2
        return torch.grid_sampler(img.unsqueeze(0), grid, mode_enum, 0, True)[0]
