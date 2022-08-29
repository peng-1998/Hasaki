from typing import List, Tuple, Union

import numpy
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor
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

    def __init__(self, size: Tuple[int, int] | int, scale: Tuple[float, float], ratio: Tuple[float], interpolation: List[str] | str = T.InterpolationMode.BILINEAR):
        """Crop a random portion of image and resize it to a given size.

        Args:
            size (Tuple[int, int] | int): expected output size of the crop, for each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made.
            scale (Tuple[float,float]): Specifies the lower and upper bounds for the random area of the crop,
            before resizing. The scale is defined with respect to the area of the original image.
            ratio (Tuple[float]):lower and upper bounds for the random aspect ratio of the crop, before
            resizing.
            interpolation (List[str] | str, optional): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.If a list type is passed in, the corresponding interpolation method will be applied to each image to be transformed. Defaults to T.InterpolationMode.BILINEAR.
        """

        if isinstance(interpolation, str):
            super().__init__(size, scale=scale, ratio=ratio, interpolation=interpolation)
        else:
            super().__init__(size, scale=scale, ratio=ratio, interpolation=T.InterpolationMode.BILINEAR)
        self.interpolation = interpolation

    def forward(self, *args) -> list:
        i, j, h, w = self.get_params(args[0], self.scale, self.ratio)
        if isinstance(self.interpolation, str):
            return [TF.resized_crop(_, i, j, h, w, self.size, self.interpolation) for _ in args]
        else:
            return [TF.resized_crop(_, i, j, h, w, self.size, interpolation) for _, interpolation in zip(args, self.interpolation)]


class RandomAffine(T.RandomAffine):

    def __init__(self, degrees: float | Tuple[float, float], translate: Tuple[float, float] = None, scale: Tuple[float, float] = None, shear: List[float] | float = None, interpolation: List[str] | str = T.InterpolationMode.BILINEAR, fill=0, fillcolor=None, resample=None):
        """Random affine transformation of the image keeping center invariant.

        Args:
            degrees (float | Tuple[float, float]): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
            translate (Tuple[float, float], optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
            scale (Tuple[float, float], optional):scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
            shear (List[float] | float, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be applied. Else if shear is a sequence of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a sequence of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default.
            interpolation (List[str] | str, optional): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.If a list type is passed in, the corresponding interpolation method will be applied to each image to be transformed. Defaults to T.InterpolationMode.BILINEAR.
            fill (int, optional): Pixel fill value for the area outside the transformed
            image. Default is ``0``.
        """
        if isinstance(interpolation, str):
            super().__init__(degrees, translate=translate, scale=scale, shear=shear, interpolation=interpolation, fill=fill, fillcolor=fillcolor, resample=resample)
        else:
            super().__init__(degrees, translate=translate, scale=scale, shear=shear, interpolation=T.InterpolationMode.BILINEAR, fill=fill, fillcolor=fillcolor, resample=resample)
        self.interpolation = interpolation

    def forward(self, *args) -> list[Tensor]:
        fill = self.fill
        fills = []
        for img in args:
            if isinstance(img, Tensor):
                if isinstance(fill, int | float):
                    fills.append([float(fill)] * TF._get_image_num_channels(img))
                else:
                    fills.append([float(f) for f in fill])

        img_size = TF._get_image_size(args[0])

        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)
        if isinstance(self.interpolation, str):
            return [TF.affine(args[i], *ret, interpolation=self.interpolation, fill=fills[i]) for i in range(len(args))]
        else:
            return [TF.affine(args[i], *ret, interpolation=self.interpolation[i], fill=fills[i]) for i in range(len(args))]


class RandomRotation(T.RandomRotation):

    def __init__(self, degrees, interpolation: List[str] | str = TF.InterpolationMode.BILINEAR, expand=False, center=None, fill=0, resample=None):

        if isinstance(interpolation, str):
            super().__init__(degrees, interpolation=interpolation, expand=expand, center=center, fill=fill, resample=resample)
        else:
            super().__init__(degrees, interpolation=T.InterpolationMode.BILINEAR, expand=expand, center=center, fill=fill, resample=resample)
        self.interpolation = interpolation

    def forward(self, *args) -> list:
        fill = self.fill
        fills = []
        for img in args:
            if isinstance(img, Tensor):
                if isinstance(fill, int | float):
                    fills.append([float(fill)] * TF._get_image_num_channels(img))
                else:
                    fills.append([float(f) for f in fill])

        angle = self.get_params(self.degrees)
        if len(self.interpolations) == 0:
            return [TF.rotate(args[i], angle, self.resample, self.expand, self.center, fills[i], interpolation=self.interpolation) for i in range(len(args))]
        else:
            return [TF.rotate(args[i], angle, self.resample, self.expand, self.center, fills[i], interpolation=self.interpolations[i]) for i in range(len(args))]


class ElasticDeformation(Module):

    def __init__(self, grids_size: Tuple[int, int], sigma: float, interpolation: List[str] | str = TF.InterpolationMode.BILINEAR):
        """

        Args:
            grids_size (Tuple[int, int]): Size of the grid.
            sigma (float): Sigma of the gaussian filter.
            interpolation (List[str] | str, optional): _description_. Defaults to TF.InterpolationMode.BILINEAR.
        """
        '''
        If \'interpolations\' is an empty array,it will apply parameter \'interpolation\' for each input image.
        If not,the parameter \'interpolations\''s should have the same lenght with input image.
        '''
        super().__init__()
        self.grids_size = list(grids_size)
        self.sigma = sigma
        self.interpolation = interpolation

    def forward(self, *args) -> list:
        w, h = TF._get_image_size(args[0])
        grid = self._getgrid(w, h)
        if isinstance(self.interpolation, str):
            return [self._elasticdeformation(_, grid, self.interpolation) for _ in args]
        else:
            return [self._elasticdeformation(_, grid, interpolation) for _, interpolation in zip(args, self.interpolation)]

    def _getgrid(self, w, h):
        grid = self.sigma * torch.randn([2] + self.grids_size).unsqueeze(0)
        grid = TF.resize(grid, (h, w)).permute(0, 2, 3, 1)
        grid[..., 0] = grid[..., 0] * 2 / w
        grid[..., 1] = grid[..., 1] * 2 / h
        x, y = torch.linspace(-1, 1, w), torch.linspace(-1, 1, h)
        xy = torch.stack(torch.meshgrid(x, y)).permute(2, 1, 0).unsqueeze(0)
        return grid + xy

    def _elasticdeformation(self, img: Image.Image | numpy.ndarray | Tensor, grid: Tensor, interpolation: str = T.InterpolationMode.BILINEAR):
        if isinstance(img, Image.Image | numpy.ndarray):
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
    """
    It is suitable for dilate operation of binary map.The input should has size (N,H,W) or (H,W). 
    The kernel indicates the connected region and will be [[0,1,0],[1,1,1],[0,1,0]] for default. 
    The size of kernel should be (2k+1,2k+1).
    """

    def __init__(self, kernel: Tensor = None) -> None:
        """

        Args:
            kernel (Tensor, optional): Kernel of dilate operation. Defaults to None.
        """        
        super().__init__()
        self.kernel = kernel if kernel is not None else torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float32)

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) == 2:
            return dilate(x.unsqueeze(0), self.kernel)[0]
        else:
            return dilate(x, self.kernel)


class Erosion(Module):
    """
    It is suitable for erosion operation of binary map.The input should has size (N,H,W) or (H,W). 
    The kernel indicates the connected region and will be [[0,1,0],[1,1,1],[0,1,0]] for default. 
    The size of kernel should be (2k+1,2k+1).
    """

    def __init__(self, kernel: Tensor = None) -> None:
        """

        Args:
            kernel (Tensor, optional): Kernel of erosion operation. Defaults to None.
        """        
        super().__init__()
        self.kernel = kernel if kernel is not None else torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float32)

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) == 2:
            return erosion(x.unsqueeze(0), self.kernel)[0]
        else:
            return erosion(x, self.kernel)
