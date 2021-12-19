from typing import List, Tuple, Union
import numpy
from torch.functional import Tensor
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch
import torch.nn.functional as F
from torch.nn import Module
from torchvision.transforms.transforms import RandomRotation, RandomVerticalFlip
from PIL import Image
from .functional2D import erosion, dilate


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
    def forward(self, *args) -> list:
        i, j, h, w = self.get_params(args[0], self.scale, self.ratio)
        return [TF.resized_crop(_, i, j, h, w, self.size, self.interpolation) for _ in args]


class RandomAffine(T.RandomAffine):
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

        return [TF.affine(args[i], *ret, interpolation=self.interpolation, fill=fills[i]) for i in range(len(args))]


class RandomRotation(T.RandomRotation):
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

        angle = self.get_params(self.degrees)

        return [TF.rotate(args[i], angle, self.resample, self.expand, self.center, fills[i]) for i in range(len(args))]


class ElasticDeformation(Module):
    def __init__(self, grids_size: Tuple[int, int], sigma: float):
        super().__init__()
        self.grids_size = list(grids_size)
        self.sigma = sigma

    def forward(self, *args) -> list:
        w, h = TF._get_image_size(args[0])
        grid = self._getgrid(w, h)
        return [self._elasticdeformation(_, grid) for _ in args]

    def _getgrid(self, w, h):
        grid = self.sigma * torch.randn([2] + self.grids_size).unsqueeze(0)
        grid = TF.resize(grid, (h, w)).permute(0, 2, 3, 1)
        grid[..., 0] = grid[..., 0] * 2 / w
        grid[..., 1] = grid[..., 1] * 2 / h
        x, y = torch.linspace(-1, 1, w), torch.linspace(-1, 1, h)
        xy = torch.stack(torch.meshgrid(x, y)).permute(2, 1, 0).unsqueeze(0)
        return grid + xy

    def _elasticdeformation(self, img, grid: Tensor):
        if isinstance(img, (Image.Image, numpy.ndarray)):
            image = TF.to_tensor(img)
        else:
            image = img
        image = torch.grid_sampler(image.unsqueeze(0), grid, 2, 0, True)[0]
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
