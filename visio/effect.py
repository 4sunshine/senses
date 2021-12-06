from visio.source import *
from dataclasses import dataclass
from visio.filters import ColorConverterCUDA
from visio.region import Region

import numpy as np
import torch


__all__ = ['ColorGridCUDA', 'ColorGridCUDAConfig', 'EFFECT_MAPPING']


class EffectSource(Source):
    def process_stream(self, stream):
        pass


class Effect(Enum):
    Grid = 0
    Colorize = 1
    RandomLines = 2
    ChannelShift = 3


@dataclass
class ColorGridCUDAConfig:
    solution = Effect.Grid
    type = SourceType.effect
    name = 'color_grid'
    device = Device.cuda
    threshold = 0.05
    apply_x = True
    apply_y = True
    step_x = 40 #4
    step_y = 40 #4
    color = (0, 255, 0)


class ColorGridCUDA(EffectSource):
    def __init__(self, cfg=None):
        super(ColorGridCUDA, self).__init__(cfg)
        self.color = torch.tensor(self.cfg.color, dtype=torch.float32, device='cuda') / 255.

    def default_config(self):
        return ColorGridCUDAConfig()

    def process_stream(self, stream):
        # image = stream['rgb_buffer_cuda']
        # if len(image.shape) > 2:
        if not stream['new_ready']:
            return
        if self.cfg.apply_x:
            stream['rgb_buffer_cuda'][:3, :, ::self.cfg.step_x] = self.color[:, None, None]
        if self.cfg.apply_y:
            stream['rgb_buffer_cuda'][:3, ::self.cfg.step_y, :] = self.color[:, None, None]


@dataclass
class GradientColorizeCUDAConfig:
    solution = Effect.Colorize
    type = SourceType.effect
    name = 'colorize'
    device = Device.cuda
    apply_x = True
    invert = False
    sqrt = False
    flip = False
    colormap = 'tab20c'  #'plasma'#'nipy_spectral'
    target = Region.Body  # Could be face, body, or None


class GradientColorizeCUDA(EffectSource):
    def __init__(self, cfg=None):
        super(GradientColorizeCUDA, self).__init__(cfg)
        self.color_model = ColorConverterCUDA(self.cfg.colormap)

    def default_config(self):
        return GradientColorizeCUDAConfig()

    def process_stream(self, stream):
        if not stream['new_ready']:
            return
        image = stream['rgb_buffer_cuda'].clone()
        if self.cfg.target == Region.Body:
            bbox = stream['rois'].get('person_region', None)
        elif self.cfg.target == Region.Face:
            bbox = stream['rois'].get('face', None)
        else:
            bbox = None
        if bbox is None or np.all(bbox < 0):
            endpoints = None
        else:
            if self.cfg.apply_x:
                endpoints = bbox[0], bbox[2]
            else:
                endpoints = bbox[1], bbox[3]

        stream['rgb_buffer_cuda'] = self.color_model.process_grad_grayscale(image, endpoints, self.cfg.apply_x,
                                                                            self.cfg.invert, self.cfg.sqrt,
                                                                            self.cfg.flip)


@dataclass
class RandomLinesCUDAConfig:
    solution = Effect.RandomLines
    type = SourceType.effect
    name = 'random_lines'
    device = Device.cuda
    apply_x = True
    count_x = 290
    apply_y = True
    count_y = 245
    same = True
    change_every = 1


class RandomLinesCUDA(EffectSource):
    def __init__(self, cfg=None):
        """COLOR IS THE SAME AT THE MOMENT"""
        super(RandomLinesCUDA, self).__init__(cfg)
        if self.cfg.apply_x:
            self._index_x = torch.randint(0, 1, (self.cfg.count_x,), dtype=torch.int64)
        if self.cfg.apply_y:
            self._index_y = torch.randint(0, 1, (self.cfg.count_y,), dtype=torch.int64)

    def default_config(self):
        return RandomLinesCUDAConfig()

    def process_stream(self, stream):
        if not stream['new_ready']:
            return
        h, w = stream['rgb_buffer_cuda'].shape[-2:]
        if self.tick() % self.cfg.change_every == 0:
            if self.cfg.apply_x:
                self._index_x = torch.randint(0, w, (self.cfg.count_x,), dtype=torch.int64)
            if self.cfg.apply_y:
                self._index_y = torch.randint(0, h, (self.cfg.count_y,), dtype=torch.int64)
        if self.cfg.same:
            if self.cfg.apply_x:
                stream['rgb_buffer_cuda'][:3, :, self._index_x] =\
                    torch.rand((3,), dtype=torch.float32, device='cuda')[:, None, None]
            if self.cfg.apply_y:
                stream['rgb_buffer_cuda'][:3, self._index_y, :] =\
                    torch.rand((3,), dtype=torch.float32, device='cuda')[:, None, None]
        else:
            if self.cfg.apply_x:
                stream['rgb_buffer_cuda'][:3, :, self._index_x] =\
                    torch.rand((3, self.cfg.count_x), dtype=torch.float32, device='cuda')[:, None, :]
            if self.cfg.apply_y:
                stream['rgb_buffer_cuda'][:3, self._index_y, :] =\
                    torch.rand((3, self.cfg.count_y), dtype=torch.float32, device='cuda')[:, :, None]


@dataclass
class ChannelShiftConfig:
    solution = Effect.ChannelShift
    type = SourceType.effect
    name = 'channel_shift'
    device = Device.cuda
    apply_x = True
    shift_x = 4
    apply_y = False
    shift_y = 4


class ChannelShift(EffectSource):
    def __init__(self, cfg=None):
        super().__init__(cfg)

    def default_config(self):
        return ChannelShiftConfig()

    def process_stream(self, stream):
        if not stream['new_ready']:
            return
        # CHECK LATER WITH TORCH ROLL
        if self.cfg.apply_x:
            stream['rgb_buffer_cuda'][0, :, self.cfg.shift_x:] =\
                stream['rgb_buffer_cuda'][0, :, : -self.cfg.shift_x]
            stream['rgb_buffer_cuda'][2, :, : -self.cfg.shift_x] =\
                stream['rgb_buffer_cuda'][2, :, self.cfg.shift_x:]

        if self.cfg.apply_y:
            stream['rgb_buffer_cuda'][0, self.cfg.shift_y:, :] =\
                stream['rgb_buffer_cuda'][0, : -self.cfg.shift_y, :]
            stream['rgb_buffer_cuda'][2, : -self.cfg.shift_y, :] =\
                stream['rgb_buffer_cuda'][2, self.cfg.shift_y:, :]


EFFECT_MAPPING = {
    Effect.Grid: ColorGridCUDA,
    Effect.Colorize: GradientColorizeCUDA,
    Effect.RandomLines: RandomLinesCUDA,
    Effect.ChannelShift: ChannelShift,
}
