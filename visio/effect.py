from visio.source import *
from dataclasses import dataclass

import torch


__all__ = ['ColorGridCUDA', 'ColorGridCUDAConfig', 'EFFECT_MAPPING']


class EffectSource(Source):
    def process_stream(self, stream):
        pass


class Effect(Enum):
    Grid = 0


@dataclass
class ColorGridCUDAConfig:
    solution = Effect.Grid
    type = SourceType.effect
    name = 'effect'
    device = Device.cuda
    threshold = 0.05
    apply_x = True
    apply_y = True
    step_x = 4
    step_y = 4
    color = (0, 0, 0)


class ColorGridCUDA(EffectSource):
    def __init__(self, cfg):
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


EFFECT_MAPPING = {
    Effect.Grid: ColorGridCUDA,
}
