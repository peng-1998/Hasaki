import torch
from torch.functional import Tensor
import torch.nn as nn
from .utils import unet_like
from torch.nn import Module


class Unet_2D_with_any_layer(unet_like):
    def __init__(self, image_channels: int, out_channels: int, layers: int, base_channels: int = 64) -> None:
        encoders = [self._make_layers(image_channels, base_channels)]
        encoders += [self._make_layers(base_channels * (2**i), base_channels * (2**(i + 1))) for i in range(layers - 1)]
        decoders = [self._make_layers(base_channels * (2**(i + 1)), base_channels * (2**i)) for i in range(layers)]
        skips = [lambda x: x for i in range(layers)]
        fuses = [lambda x, y: torch.cat([x, y], dim=1) for i in range(layers)]
        upsamples = [nn.ConvTranspose2d(64 * (2**i), 64 * (2**(i - 1)), 2, 2) for i in range(1, layers + 1)]
        downsmpler = [nn.MaxPool2d(2, 2) for i in range(layers)]
        bridge = self._make_layers(base_channels * (2**(layers - 1)), base_channels * (2**layers))
        after_unet = nn.Conv2d(base_channels, out_channels, 1, 1)
        super().__init__(encoders=encoders, decoders=decoders, downsamples=downsmpler, upsamples=upsamples, skips=skips, fuses=fuses, bridge=bridge, after_unet=after_unet)

    def _make_layers(self, in_channels, out_channels) -> nn.Sequential:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.BatchNorm2d(out_channels), nn.ReLU(True), nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.BatchNorm2d(out_channels), nn.ReLU(True))


class Unet_in_paper(unet_like):
    def __init__(self, image_channels: int, out_channels: int) -> None:
        encoders = [
            nn.Sequential(nn.Conv2d(image_channels, 64, 3), nn.ReLU(True), nn.Conv2d(64, 64, 3), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(64, 64 * 2, 3), nn.ReLU(True), nn.Conv2d(64 * 2, 64 * 2, 3), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(64 * 2, 64 * 4, 3), nn.ReLU(True), nn.Conv2d(64 * 4, 64 * 4, 3), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(64 * 4, 64 * 8, 3), nn.ReLU(True), nn.Conv2d(64 * 8, 64 * 8, 3), nn.ReLU(True))
        ]
        decoders = [
            nn.Sequential(nn.Conv2d(64 * 2, 64, 3), nn.ReLU(True), nn.Conv2d(64, 64, 3), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(64 * 4, 64 * 2, 3), nn.ReLU(True), nn.Conv2d(64 * 2, 64 * 2, 3), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(64 * 8, 64 * 4, 3), nn.ReLU(True), nn.Conv2d(64 * 4, 64 * 4, 3), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(64 * 16, 64 * 8, 3), nn.ReLU(True), nn.Conv2d(64 * 8, 64 * 8, 3), nn.ReLU(True), nn.Dropout())
        ]
        skips = [lambda x: x[..., 88:480, 88:480], lambda x: x[..., 40:240, 40:240], lambda x: x[..., 16:120, 16:120], lambda x: x[..., 4:60, 4:60]]
        fuses = [lambda x, y: torch.cat([x, y], dim=1) for i in range(4)]
        downsamples = [nn.MaxPool2d(2, 2) for i in range(4)]
        upsamples = [nn.ConvTranspose2d(64 * (2**i), 64 * (2**(i - 1)),2,2) for i in range(1, 5)]
        bridge = nn.Sequential(nn.Conv2d(64 * 8, 64 * 16, 3), nn.ReLU(True), nn.Conv2d(64 * 16, 64 * 16, 3), nn.ReLU(True), nn.Dropout())
        after_unet = nn.Conv2d(64, out_channels, 1, 1)
        super().__init__(encoders, decoders, downsamples, upsamples, skips, fuses, bridge, None, after_unet)


class Res_Base_block(Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.F = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.BatchNorm2d(out_channels), nn.ReLU(), nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.H = lambda x: x
        if in_channels != out_channels:
            self.H = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))

    def forward(self, x: Tensor) -> Tensor:
        return self.F(x) + self.H(x)


class Dense_block(Module):
    def __init__(self, base_channels) -> None:
        super().__init__()
        self.den_con_layer1 = self._make_layer(base_channels, base_channels)
        self.den_con_layer2 = self._make_layer(base_channels * 2, base_channels)
        self.den_con_layer3 = self._make_layer(base_channels * 3, base_channels)
        self.den_con_layer4 = self._make_layer(base_channels * 4, base_channels)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.den_con_layer1(x)
        x2 = self.den_con_layer2(torch.cat([x, x1], dim=1))
        x3 = self.den_con_layer3(torch.cat([x, x1, x2], dim=1))
        x4 = self.den_con_layer4(torch.cat([x, x1, x2, x3], dim=1))
        return x4

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(nn.BatchNorm2d(in_channels), nn.Conv2d(in_channels, out_channels, 1), nn.ReLU(True), nn.BatchNorm2d(out_channels), nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.ReLU(True), nn.Dropout())


class Transition_block(Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.BatchNorm2d(channels), nn.Conv2d(channels, channels*2, 1), nn.MaxPool2d(2, 2))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Merge_layer(Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.conv(torch.cat([x, y], dim=1))


class Dense_Unet_with_any_layer(unet_like):
    def __init__(self, image_channels: int, out_channels: int, layers: int, base_channels: int = 64) -> None:
        encoders = [Dense_block(base_channels * (2**i)) for i in range(layers)]
        decoders = [Dense_block(base_channels * (2**i)) for i in range(layers)]
        skips = [lambda x: x for i in range(layers)]
        fuses = [Merge_layer(base_channels * (2**i)) for i in range(layers)]
        downsmpler = [Transition_block(base_channels * (2**i)) for i in range(layers)]
        upsamples = [nn.ConvTranspose2d(64 * (2**i), 64 * (2**(i - 1)), 2, 2) for i in range(1, layers + 1)]
        bridge = Dense_block(base_channels * (2**layers))
        before_unet = nn.Sequential(nn.Conv2d(image_channels, base_channels, 3, 1, 1), nn.BatchNorm2d(base_channels))
        after_unet = nn.Conv2d(base_channels, out_channels, 1, 1)
        super().__init__(encoders=encoders, decoders=decoders, downsamples=downsmpler, upsamples=upsamples, skips=skips, fuses=fuses, bridge=bridge, before_unet=before_unet, after_unet=after_unet)


class Res_Unet_with_any_layer(unet_like):
    def __init__(self, image_channels: int, out_channels: int, layers: int, base_channels: int = 64) -> None:
        encoders = [Res_Base_block(image_channels, base_channels)]
        encoders += [Res_Base_block(base_channels * (2**i), base_channels * (2**(i + 1))) for i in range(layers - 1)]
        decoders = [Res_Base_block(base_channels * (2**(i + 1)), base_channels * (2**i)) for i in range(layers)]
        skips = [lambda x: x for i in range(layers)]
        fuses = [lambda x, y: torch.cat([x, y], dim=1) for i in range(layers)]
        downsmpler = [nn.MaxPool2d(2, 2) for i in range(layers)]
        upsamples = [nn.ConvTranspose2d(64 * (2**i), 64 * (2**(i - 1)), 2, 2) for i in range(1, layers + 1)]
        bridge = Res_Base_block(base_channels * (2**(layers - 1)), base_channels * (2**layers))
        after_unet = nn.Conv2d(base_channels, out_channels, 1, 1)
        super().__init__(encoders=encoders, decoders=decoders, downsamples=downsmpler, upsamples=upsamples, skips=skips, fuses=fuses, bridge=bridge, after_unet=after_unet)
