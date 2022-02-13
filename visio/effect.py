from visio.source import *
from dataclasses import dataclass
from visio.filters import ColorConverterCUDA
from visio.region import Region

import numpy as np
import torch
import kornia


__all__ = ['ColorGridCUDA', 'ColorGridCUDAConfig', 'EFFECT_MAPPING']


class EffectSource(Source):
    def process_stream(self, stream):
        pass


class Effect(Enum):
    Grid = 0
    Colorize = 1
    RandomLines = 2
    ChannelShift = 3
    Ghost = 4
    RollShift = 5


@dataclass
class ColorGridCUDAConfig:
    solution = Effect.Grid
    type = SourceType.effect
    name = 'color_grid'
    device = Device.cuda
    threshold = 0.05
    apply_x = True
    apply_y = True
    step_x = 4
    step_y = 4
    color = (0, 0, 0)


class ColorGridCUDA(EffectSource):
    def __init__(self, cfg=None, data=None):
        super(ColorGridCUDA, self).__init__(cfg, data)
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
class GhostCUDAConfig:
    solution = Effect.Grid
    type = SourceType.effect
    name = 'color_grid'
    device = Device.cuda
    threshold = 0.05
    apply_x = True
    apply_y = True
    step_x = 4
    step_y = 4
    color = (255, 255, 255)


class GhostCUDA(EffectSource):
    def __init__(self, cfg=None, data=None):
        super(GhostCUDA, self).__init__(cfg, data)
        self.color = torch.tensor(self.cfg.color, dtype=torch.float32, device='cuda') / 255.
        filter = [[2, 4, 5, 4, 2],
                  [4, 9, 12, 9, 4],
                  [5, 12, 15, 12, 5],
                  [4, 9, 12, 9, 4],
                  [2, 4, 5, 4, 2]]
        weight_smooth = (torch.tensor(filter, dtype=torch.float32, device='cuda') / 159.).unsqueeze_(0).repeat(1, 1, 1).unsqueeze_(0)
        self.smooth = torch.nn.Conv2d(1, 1, 5, bias=False, dtype=torch.float32, padding=2, device='cuda').requires_grad_(False)
        self.smooth.weight = torch.nn.Parameter(weight_smooth, requires_grad=False)

        canny_x = [[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]]
        weight_cx = (torch.tensor(canny_x, dtype=torch.float32, device='cuda')).unsqueeze_(0).repeat(1, 1, 1).unsqueeze_(0)

        canny_y = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        weight_cy = (torch.tensor(canny_y, dtype=torch.float32, device='cuda')).unsqueeze_(0).repeat(1, 1, 1).unsqueeze_(0)

        weight_canny = torch.cat([weight_cx, weight_cy], dim=0)
        print(weight_canny.shape)

        self.canny = torch.nn.Conv2d(1, 2, 3, bias=False, dtype=torch.float32, padding=1, device='cuda').requires_grad_(False)
        self.canny.weight = torch.nn.Parameter(weight_canny, requires_grad=False)
        #
        # self.c_y = torch.nn.Conv2d(3, 1, 3, bias=False, dtype=torch.float32, padding=1, device='cuda').requires_grad_(False)
        # self.c_y.weight = weight_cy
        #

    @staticmethod
    def rgb_to_gray(img):
        """https://github.com/pytorch/vision/blob/main/torchvision/transforms/functional_tensor.py"""
        r, g, b = img.unbind(dim=-3)
        # This implementation closely follows the TF one:
        # https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/image_ops_impl.py#L2105-L2138
        l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
        return l_img.unsqueeze(dim=-3)

    def default_config(self):
        return GhostCUDAConfig()

    def process_stream(self, stream):
        # image = stream['rgb_buffer_cuda']
        # if len(image.shape) > 2:
        gray = self.rgb_to_gray(stream['rgb_buffer_cuda'])
        img = self.smooth(gray.unsqueeze(0))
        img = torch.norm(self.canny(img), dim=1, keepdim=True)  #torch.sum(self.canny(img).pow_(2), dim=1, keepdim=True).pow_(0.5)
        stream['alpha_cuda'] = img[0] * stream['alpha_cuda']
        # if not stream['new_ready']:
        #     return
        # if self.cfg.apply_x:
        #     stream['rgb_buffer_cuda'][:3, :, ::self.cfg.step_x] = self.color[:, None, None]
        # if self.cfg.apply_y:
        #     stream['rgb_buffer_cuda'][:3, ::self.cfg.step_y, :] = self.color[:, None, None]


@dataclass
class GradientColorizeCUDAConfig:
    solution = Effect.Colorize
    type = SourceType.effect
    name = 'colorize'
    device = Device.cuda
    apply_x = True
    invert = False
    sqrt = True
    flip = False
    colormap = 'nipy_spectral'  #'plasma'#'tab20c'  #'plasma'#'nipy_spectral'
    target = Region.Body  # Could be face, body, or None


class GradientColorizeCUDA(EffectSource):
    def __init__(self, cfg=None, data=None):
        super(GradientColorizeCUDA, self).__init__(cfg, data)
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
class RollShiftCUDAConfig:
    solution = Effect.RollShift
    type = SourceType.effect
    name = 'colorize'
    device = Device.cuda
    width = 1280
    height = 720
    apply_x = True
    invert = False
    sqrt = True
    flip = False
    colormap = 'nipy_spectral'  #'plasma'#'tab20c'  #'plasma'#'nipy_spectral'
    target = Region.Body  # Could be face, body, or None


class RollShiftCUDA(EffectSource):
    def __init__(self, cfg=None, data=None):
        super(RollShiftCUDA, self).__init__(cfg, data)
        x = torch.arange(self.cfg.width, dtype=torch.int32)
        y = torch.arange(self.cfg.height, dtype=torch.int32)
        self.index_x, self.index_y = torch.meshgrid(x, y, indexing='xy')
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
    change_every = 2


class RandomLinesCUDA(EffectSource):
    def __init__(self, cfg=None, data=None):
        """COLOR IS THE SAME AT THE MOMENT"""
        super(RandomLinesCUDA, self).__init__(cfg, data)
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
            if stream['alpha_cuda'] is not None:
                stream['alpha_cuda'] = torch.maximum(torch.roll(stream['alpha_cuda'], self.cfg.shift_x, dims=2),
                                                     torch.roll(stream['alpha_cuda'], -self.cfg.shift_x, dims=2))
            # stream['alpha_cuda'] = stream['alpha_cuda'][:, :, ]

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
    Effect.Ghost: GhostCUDA,
}
