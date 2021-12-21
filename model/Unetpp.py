from typing import List
import torch.nn as nn
import torch
from torch.nn import Module
from torch import Tensor


class Unet_pp_base_block(Module):
    def __init__(self, nexts: List = [], next_up: Module = None, next_down: Module = None, num_maps: int = 0, down_sample: Module = None, up_sample: Module = None) -> None:
        super().__init__()
        assert (next_down and down_sample) or (not (next_down or down_sample))
        assert (next_up and up_sample) or (not (next_up or up_sample))        
        self.nexts = nexts
        self.maps = []
        self.next_up = next_up
        self.next_down = next_down
        self.num_maps = 0
        self.network = num_maps
        self.down_sample = down_sample
        self.up_sample = up_sample

    def set_map(self, x: Tensor) -> None:
        self.maps.append(x)
        if len(self.maps) == self.num_maps:
            self.forward(torch.cat(self.maps), dim=1)

    def forward(self, x: Tensor) -> None:
        map = self.network(x)
        for unit in self.nexts:
            unit.set_map(map)
        if self.next_down:
            self.next_down.set_map(self.down_sample(map))
        if self.next_up:
            self.next_up.set_map(self.up_sample(map))

