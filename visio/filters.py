import numpy as np

from dataclasses import dataclass
from typing import Union

from visio.utils import ColorMap


class BaseFilter(object):
    def __init__(self, cfg=None, content=None):
        self.cfg = cfg
        self.content = self.init_content(content)
        self.tick = 0

    def init_content(self, content):
        return content

    def transform(self, image, *args, **kwargs):
        return image

    def reload(self):
        self.tick = 0


@dataclass
class ColorGridFilterDefaults:
    color: Union[int, int, int] = (0, 0, 0)
    step_x: int = 4
    step_y: int = 4
    apply_x: bool = True
    apply_y: bool = True


class ColorGridFilter(BaseFilter):
    def __init__(self, cfg=ColorGridFilterDefaults()):
        super().__init__(cfg)

    def transform(self, image, *args, **kwargs):
        if len(image.shape) > 2:
            if self.cfg.apply_x:
                image[:, ::self.cfg.step_x, :3] = self.cfg.color
            if self.cfg.apply_y:
                image[::self.cfg.step_y, :, :3] = self.cfg.color
        elif len(image.shape) == 2:
            if self.cfg.apply_x:
                image[:, ::self.cfg.step_x] = self.cfg.color[0]
            if self.cfg.apply_y:
                image[::self.cfg.step_y, :] = self.cfg.color[0]
        return image


@dataclass
class GrayScaleGradientColorizeFilterDefaults:
    colormap: str = 'nipy_spectral'


class GrayScaleGradientColorizeFilter(BaseFilter):
    def __init__(self, cfg=GrayScaleGradientColorizeFilterDefaults()):
        super(GrayScaleGradientColorizeFilter, self).__init__(cfg)

    def init_content(self, content):
        return ColorMap(self.cfg.colormap)

    def transform(self, image, *args, **kwargs):
        return self.content.process_grad_grayscale(image, *args, **kwargs)


@dataclass
class RandomVerticalLinesDefaults:
    count: int = 15
    change_every: int = 1


class RandomVerticalLines(BaseFilter):
    def __init__(self, cfg=RandomVerticalLinesDefaults()):
        super(RandomVerticalLines, self).__init__(cfg)

    def transform(self, image, *args, **kwargs):
        w = image.shape[1]
        if self.tick % self.cfg.change_every == 0:
            index = np.array(np.random.randint(0, w, self.cfg.count, dtype=np.int32))
        self.tick += 1
        if len(image.shape) == 2:
            image[:, index] = np.random.randint(0, 255, (1,), dtype=np.uint8)
        else:
            image[:, index, :3] = np.random.randint(0, 255, (3,), dtype=np.uint8)
        return image
