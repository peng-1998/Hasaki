from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module


class unet_like(Module):
    def __init__(self, encoders: List[Module], decoders: List[Module], downsamples: List[Module], upsamples: List[Module], skips: List[Module], fuses: List[Module], bridge: Module = None, before_unet: Module = None, after_unet: Module = None) -> None:
        super().__init__()

        assert isinstance(encoders, List)
        for i, m in enumerate(encoders):
            self.add_module(f'encoder{i}', m)
        self.encoders = encoders

        assert isinstance(decoders, List)
        for i, m in enumerate(decoders):
            self.add_module(f'decoder{i}', m)
        self.decoders = decoders

        assert len(encoders) == len(decoders)

        assert isinstance(skips, List)
        for i, m in enumerate(skips):
            if isinstance(m, Module):
                self.add_module(f'skip{i}', m)
        self.skips = skips

        assert isinstance(downsamples, List)
        for i, m in enumerate(downsamples):
            if isinstance(m, Module):
                self.add_module(f'downsamples{i}', m)
        self.downsamples = downsamples

        assert isinstance(upsamples, List)
        for i, m in enumerate(upsamples):
            if isinstance(m, Module):
                self.add_module(f'upsamples{i}', m)
        self.upsamples = upsamples

        assert isinstance(fuses, List)
        for i, m in enumerate(fuses):
            if isinstance(m, Module):
                self.add_module(f'fuse{i}', m)
        self.fuses = fuses

        self.bridge = bridge
        if before_unet is not None:
            self.before_unet = before_unet
        else:
            self.before_unet = lambda x: x
        if after_unet is not None:
            self.after_unet = after_unet
        else:
            self.after_unet = lambda x: x

    def forward(self, x):
        x = self.before_unet(x)
        x = self._forward(0, x)
        x = self.after_unet(x)
        return x

    def _forward(self, layer: int, x):
        if layer == len(self.encoders):
            if self.bridge is not None:
                return self.bridge(x)
            else:
                return x
        else:
            x = self.encoders[layer](x)
            x_ = self._forward(layer + 1, self.downsamples[layer](x))
            x_ = self.upsamples[layer](x_)
            x = self.skips[layer](x)
            x = self.fuses[layer](x_, x)
            x = self.decoders[layer](x)
            return x
