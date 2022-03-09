import torch
import torch.nn as nn
from typing import List
from torch import Tensor
from torch.nn import Module


class Unet_pp_base_block(Module):
    def __init__(self, net: Module, nexts: List = [], num_maps: int = 0, down_sample: Module = None, up_sample: Module = None) -> None:
        super().__init__()
        self.nexts = nexts
        self.maps = []
        self.next_up = []
        self.next_down = []
        self.num_maps = num_maps
        self.network = net
        self.down_sample = down_sample
        self.up_sample = up_sample

    def set_map(self, x: Tensor) -> None:
        self.maps.append(x)
        if len(self.maps) == self.num_maps:
            self.forward(torch.cat(self.maps, dim=1))

    def forward(self, x: Tensor) -> None:
        map = self.network(x)
        for unit in self.nexts:
            unit.set_map(map)
        if len(self.next_down):
            self.next_down[0].set_map(self.down_sample(map))
        if len(self.next_up):
            self.next_up[0].set_map(self.up_sample(map))
        self.maps = []


class Unet_pp(Module):
    def __init__(self, in_channel: int, out_channel: int, layers: int, base_channel: int = 64) -> None:
        super().__init__()
        make_layer = lambda in_c, out_c: nn.Sequential(nn.Conv2d(in_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(), nn.Conv2d(out_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU())
        for l in range(layers):
            for i in range(l + 1):
                in_ch = (i + 1) * base_channel * (2**(layers - 1 - l))
                out_ch = base_channel * (2**(layers - 1 - l))
                if i == 0:
                    in_ch = int(in_ch / 2)
                if l == layers - 1 and i == 0:
                    in_ch = in_channel
                down_sample = nn.MaxPool2d(2, 2) if i == 0 and l != 0 else None
                up_sample = nn.ConvTranspose2d(out_ch, int(out_ch / 2), 2, 2) if l != layers - 1 else None
                self.add_module(f'x_{l}_{i}', Unet_pp_base_block(num_maps=(i + 1), nexts=[], net=make_layer(in_ch, out_ch), down_sample=down_sample, up_sample=up_sample))

        for l in range(layers):
            for i in range(l + 1):
                for k in range(i + 1, l + 1):
                    self._modules[f'x_{l}_{i}'].nexts.append(self._modules[f'x_{l}_{k}'])
                if i == 0 and l != 0:
                    self._modules[f'x_{l}_{i}'].next_down.append(self._modules[f'x_{l-1}_{0}'])
                if l != layers - 1:
                    self._modules[f'x_{l}_{i}'].next_up.append(self._modules[f'x_{l+1}_{i+1}'])
        for i in range(1, layers):
            self._modules[f'x_{layers-1}_{i}'].nexts.append(self)
        self.layers = layers
        self.result = []
        self.adjust = nn.ModuleList(nn.Conv2d(base_channel, out_channel, 1) for i in range(layers - 1))

    def set_map(self, x: Tensor) -> None:
        self.result.append(x)

    def forward(self, x: Tensor) -> List[Tensor]:
        self._modules[f'x_{self.layers-1}_0'](x)
        result = self.result
        self.result = []
        return [self.adjust[i](result[i]) for i in range(len(result))]
