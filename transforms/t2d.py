from typing import List, Tuple, Union

import numpy
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.functional import Tensor
from torch.nn import Module

from .functional2D import dilate, erosion


class RandomCrop(T.RandomCrop):

    def forward(self, *args) -> list:
        result = [_ for _ in args]
        if self.padding is not None:
            result = [TF.pad(_, self.padding, self.fill, self.padding_mode) for _ in result]

        width, height = TF._get_image_size(result[0])
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            result = [TF.pad(_, padding, self.fill, self.padding_mode) for _ in result]
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            result = [TF.pad(_, padding, self.fill, self.padding_mode) for _ in result]

        i, j, h, w = self.get_params(result[0], self.size)

        return [TF.crop(_, i, j, h, w) for _ in result]


class RandomVerticalFlip(T.RandomVerticalFlip):

    def forward(self, *args) -> Union[tuple, List]:
        if torch.rand(1) < self.p:
            return [TF.vflip(_) for _ in args]
        return args


class RandomHorizontalFlip(T.RandomHorizontalFlip):

    def forward(self, *args) -> Union[tuple, List]:
        if torch.rand(1) < self.p:
            return [TF.hflip(_) for _ in args]
        return args


class RandomResizedCrop(T.RandomResizedCrop):

    def __init__(self, size, scale=..., ratio=..., interpolation=..., interpolations: List = []):
        '''
        The only parameter different with torchvision.transforms.RandomResizedCrop is we use \'interpolations\' apply in input images use same indexes.
        If \'interpolations\' is an empty array,it will apply parameter \'interpolation\' for each input image.
        If not,the parameter \'interpolations\''s should have the same lenght with input image.
        '''
        super().__init__(size, scale=scale, ratio=ratio, interpolation=interpolation)
        self.interpolations = interpolations

    def forward(self, *args) -> list:
        i, j, h, w = self.get_params(args[0], self.scale, self.ratio)
        if len(self.interpolations) == 0:
            return [TF.resized_crop(_, i, j, h, w, self.size, self.interpolation) for _ in args]
        else:
            return [TF.resized_crop(_, i, j, h, w, self.size, interpolation) for _, interpolation in zip(args, self.interpolations)]


class RandomAffine(T.RandomAffine):

    def __init__(self, degrees, translate=None, scale=None, shear=None, interpolation=..., fill=0, fillcolor=None, resample=None, interpolations=[]):
        '''
        The only parameter different with torchvision.transforms.RandomAffine is we use \'interpolations\' apply in input images use same indexes.
        If \'interpolations\' is an empty array,it will apply parameter \'interpolation\' for each input image.
        If not,the parameter \'interpolations\''s should have the same lenght with input image.
        '''
        super().__init__(degrees, translate=translate, scale=scale, shear=shear, interpolation=interpolation, fill=fill, fillcolor=fillcolor, resample=resample)
        assert isinstance(interpolations, List)
        self.interpolations = interpolations

    def forward(self, *args) -> list:
        fill = self.fill
        fills = []
        for img in args:
            if isinstance(img, torch.Tensor):
                if isinstance(fill, (int, float)):
                    fills.append([float(fill)] * TF._get_image_num_channels(img))
                else:
                    fills.append([float(f) for f in fill])

        img_size = TF._get_image_size(args[0])

        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)
        if len(self.interpolations) == 0:
            return [TF.affine(args[i], *ret, interpolation=self.interpolation, fill=fills[i]) for i in range(len(args))]
        else:
            return [TF.affine(args[i], *ret, interpolation=self.interpolations[i], fill=fills[i]) for i in range(len(args))]


class RandomRotation(T.RandomRotation):

    def __init__(self, degrees, interpolation=..., expand=False, center=None, fill=0, resample=None, interpolations=[]):
        '''
        The only parameter different with torchvision.transforms.RandomRotation is we use \'interpolations\' apply in input images use same indexes.
        If \'interpolations\' is an empty array,it will apply parameter \'interpolation\' for each input image.
        If not,the parameter \'interpolations\''s should have the same lenght with input image.
        '''
        super().__init__(degrees, interpolation=interpolation, expand=expand, center=center, fill=fill, resample=resample)
        self.interpolations = interpolations

    def forward(self, *args) -> list:
        fill = self.fill
        fills = []
        for img in args:
            if isinstance(img, torch.Tensor):
                if isinstance(fill, (int, float)):
                    fills.append([float(fill)] * TF._get_image_num_channels(img))
                else:
                    fills.append([float(f) for f in fill])

        angle = self.get_params(self.degrees)
        if len(self.interpolations) == 0:
            return [TF.rotate(args[i], angle, self.resample, self.expand, self.center, fills[i], interpolation=self.interpolation) for i in range(len(args))]
        else:
            return [TF.rotate(args[i], angle, self.resample, self.expand, self.center, fills[i], interpolation=self.interpolations[i]) for i in range(len(args))]


class ElasticDeformation(Module):

    def __init__(self, grids_size: Tuple[int, int], sigma: float, interpolation=TF.InterpolationMode.BILINEAR, interpolations=[]):
        '''
        If \'interpolations\' is an empty array,it will apply parameter \'interpolation\' for each input image.
        If not,the parameter \'interpolations\''s should have the same lenght with input image.
        We recommend bilinear interpolation for the original image and nearest neighbor sampling for the label.
        '''
        super().__init__()
        self.grids_size = list(grids_size)
        self.sigma = sigma
        self.interpolations = interpolations
        self.interpolation = interpolation

    def forward(self, *args) -> list:
        w, h = TF._get_image_size(args[0])
        grid = self._getgrid(w, h)
        if len(self.interpolations) == 0:
            return [self._elasticdeformation(_, grid, self.interpolation) for _ in args]
        else:
            return [self._elasticdeformation(_, grid, interpolation) for _, interpolation in zip(args, self.interpolations)]

    def _getgrid(self, w, h):
        grid = self.sigma * torch.randn([2] + self.grids_size).unsqueeze(0)
        grid = TF.resize(grid, (h, w)).permute(0, 2, 3, 1)
        grid[..., 0] = grid[..., 0] * 2 / w
        grid[..., 1] = grid[..., 1] * 2 / h
        x, y = torch.linspace(-1, 1, w), torch.linspace(-1, 1, h)
        xy = torch.stack(torch.meshgrid(x, y)).permute(2, 1, 0).unsqueeze(0)
        return grid + xy

    def _elasticdeformation(self, img, grid: Tensor, interpolation):
        if isinstance(img, (Image.Image, numpy.ndarray)):
            image = TF.to_tensor(img)
        else:
            image = img
        if interpolation == "bilinear":
            mode_enum = 0
        elif interpolation == "nearest":
            mode_enum = 1
        else:  # mode == 'bicubic'
            mode_enum = 2
        image = torch.grid_sampler(image.unsqueeze(0), grid, mode_enum, 0, True)[0]
        if isinstance(img, torch.Tensor):
            return image
        if isinstance(img, numpy.ndarray):
            return image.cpu().numpy()
        if isinstance(img, Image.Image):
            return TF.to_pil_image(image)


class Dilate(Module):
    r'It is suitable for dilate operation of binary map.The input should has size (N,H,W) or (H,W). \
    The kernel indicates the connected region and will be [[0,1,0],[1,1,1],[0,1,0]] if kernel is None. \
    The size of kernel should be (2k+1,2k+1).'

    def __init__(self, kernel: Tensor = None) -> None:
        super().__init__()
        self.kernel = kernel
        if kernel is None:
            self.kernel = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float32)

    def forward(self, x):
        if len(x.shape) == 2:
            return dilate(x.unsqueeze(0), self.kernel)[0]
        else:
            return dilate(x, self.kernel)


class Erosion(Module):
    r'It is suitable for erosion operation of binary map.The input should has size (N,H,W) or (H,W). \
    The kernel indicates the connected region and will be [[0,1,0],[1,1,1],[0,1,0]] if kernel is None. \
    The size of kernel should be (2k+1,2k+1).'

    def __init__(self, kernel: Tensor = None) -> None:
        super().__init__()
        self.kernel = kernel
        if kernel is None:
            self.kernel = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float32)

    def forward(self, x: Tensor):
        if len(x.shape) == 2:
            return erosion(x.unsqueeze(0), self.kernel)[0]
        else:
            return erosion(x, self.kernel)
