import time

import numpy as np
import torch
import cv2

import matplotlib.pyplot as plt
import torchvision.transforms.functional_tensor as F

from dataclasses import dataclass
from typing import Union
from PIL import Image


class ColorMap:
    def __init__(self, cmap='bwr'):
        """TABLE SHAPE IS 256 x 3"""
        self.table, self.cmap, self.palette, self.hsv = self._get_table(cmap)

    @staticmethod
    def _get_table(cmap):
        image = np.arange(256, dtype=np.uint8)[:, np.newaxis]
        cm = plt.get_cmap(cmap)
        colored_image = cm(image)
        colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)
        colored_hsv = cv2.cvtColor(colored_image, cv2.COLOR_RGB2HSV)
        table_hsv = colored_hsv[:, 0]
        table = colored_image[:, 0]
        table = [tuple(col) for col in table]
        return table, cmap, cm, table_hsv

    def set_color_map(self, cmap):
        self.table, self.cmap, self.palette, self.hsv = self._get_table(cmap)

    def process_grad_grayscale(self, gray, endpoints=None, axis=1, invert=False, sqrt=False, flip=False):
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        h, w = np.shape(gray)
        dim_size = w if axis else h
        if not endpoints:
            min_x, max_x = 0, dim_size - 1
        else:
            min_x, max_x = endpoints
            min_x = min(max(0, min_x), dim_size - 1)
            max_x = min(max(0, max_x), dim_size - 1)
        if sqrt:
            gray = np.sqrt(gray / 255.)  #[:, min_x: max_x + 1]
        else:
            gray = gray / 255.
        all_grad_points = np.linspace(0, 255, (max_x - min_x + 1), dtype=np.uint8)
        left_border = np.zeros((min_x, ), dtype=np.uint8)
        right_border = 255 * np.ones((max(dim_size - max_x - 1, 0), ), dtype=np.uint8)
        all_points = np.concatenate([left_border, all_grad_points, right_border])
        if flip:
            all_points = np.flip(all_points)
        if axis == 1:
            all_points = all_points[np.newaxis, :]
        else:
            all_points = all_points[:, np.newaxis]

        colored_image = self.palette(all_points)
        colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)
        colored_hsv = cv2.cvtColor(colored_image, cv2.COLOR_RGB2HSV)
        new_v = (colored_hsv[..., 2] * gray).astype(np.uint8)
        result = np.zeros(gray.shape + (3,), dtype=np.uint8)

        if not invert:
            result[..., 2] = new_v
            result[:, :, :2] = colored_hsv[:, :, :2][:, None, :]
        else:
            result[..., 1] = new_v
            result[:, :, [0, 2]] = colored_hsv[:, :, [0, 2]][:, None]

        return cv2.cvtColor(result, cv2.COLOR_HSV2RGB)

    def __getitem__(self, item):
        if isinstance(item, np.ndarray):
            colored_image = self.palette(item)
            return (colored_image[:, :, :3] * 255).astype(np.uint8)
        elif isinstance(item, Image.Image):
            img = item.convert('L')
            img = np.array(img.getchannel(0))
            colored_image = self.palette(img)
            return Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8))
        else:
            item = max(0, min(255, item))
            return self.table[item]


class ColorConverterCUDA:
    def __init__(self, colormap='nipy_spectral'):
        self.rgb_lookup, self.hsv_lookup = self._get_table(colormap)

    @staticmethod
    def rgb_to_gray(img):
        """https://github.com/pytorch/vision/blob/main/torchvision/transforms/functional_tensor.py"""
        r, g, b = img.unbind(dim=-3)
        # This implementation closely follows the TF one:
        # https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/image_ops_impl.py#L2105-L2138
        l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
        return l_img.unsqueeze(dim=-3)

    @staticmethod
    def rgb_to_hsv(img):
        return F._rgb2hsv(img)

    @staticmethod
    def hsv_to_rgb(image):
        """https://kornia.readthedocs.io/en/latest/_modules/kornia/color/hsv.html#rgb_to_hsv"""
        h = image[..., 0, :, :]
        s = image[..., 1, :, :]
        v = image[..., 2, :, :]

        hi = torch.floor(h * 6) % 6
        f = ((h * 6) % 6) - hi
        one = torch.tensor(1.0, device=image.device, dtype=image.dtype)
        p = v * (one - s)
        q = v * (one - f * s)
        t = v * (one - (one - f) * s)

        hi = hi.long()
        indices = torch.stack([hi, hi + 6, hi + 12], dim=-3)
        out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
        return torch.gather(out, -3, indices)

    @staticmethod
    def float_to_numpy(img):
        if len(img.shape) == 3:
            return img.mul(255).byte().cpu().permute(1, 2, 0).numpy()
        else:
            return img.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()

    def _get_table(self, cmap):
        image = np.arange(256, dtype=np.uint8)[:, np.newaxis]
        cm = plt.get_cmap(cmap)
        colored_image = cm(image)
        colored_image = torch.from_numpy(colored_image[:, :, :3]).permute(2, 0, 1).float()
        hsv_image = self.rgb_to_hsv(colored_image)
        rgb_table = torch.nn.Embedding(256, 3, _weight=colored_image[..., 0].T).requires_grad_(False).cuda()
        hsv_table = torch.nn.Embedding(256, 3, _weight=hsv_image[..., 0].T).requires_grad_(False).cuda()
        return rgb_table, hsv_table

    @torch.no_grad()
    def process_grad_grayscale(self, gray, endpoints=None, apply_x=True, invert=False, sqrt=False, flip=False):
        if len(gray.shape) >= 3:
            gray = self.rgb_to_gray(gray)
        _, h, w = gray.shape
        dim_size = w if apply_x else h
        if not endpoints:
            min_x, max_x = 0, dim_size - 1
        else:
            min_x, max_x = endpoints
            min_x = min(max(0, min_x), dim_size - 1)
            max_x = min(max(0, max_x), dim_size - 1)
        if sqrt:
            gray = torch.sqrt(gray)  #[:, min_x: max_x + 1]
            #gray = torch.pow(gray, 2)
        all_grad_points = torch.linspace(0, 255, (max_x - min_x + 1), dtype=torch.float32, device='cuda')
        left_border = torch.zeros((min_x, ), dtype=torch.float32, device='cuda')
        right_border = torch.ones((max(dim_size - max_x - 1, 0), ), dtype=torch.float32, device='cuda').mul_(255)
        all_points = torch.cat([left_border, all_grad_points, right_border]).int()
        if flip:
            all_points = all_points[::-1]
        if apply_x:
            all_points = all_points[None, :]
        else:
            all_points = all_points[:, None]

        colored_hsv = self.hsv_lookup(all_points).permute(2, 0, 1)

        new_v = colored_hsv[2, ...] * gray[0]
        result = torch.empty((3, ) + gray.shape[-2:], dtype=torch.float32, device='cuda')

        if not invert:
            result[2, ...] = new_v
            result[:2, :, :] = colored_hsv[:2, :, :]
        else:
            result[1, ...] = new_v
            result[[0, 2], :, :] = colored_hsv[[0, 2], :, :][:, :, None]

        return self.hsv_to_rgb(result)


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


class ComposeFilter(BaseFilter):
    def transform(self, image, *args, **kwargs):
        for c in self.content:
            image = c.transform(image)
        return image


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


class ColorGridCUDAFilter(BaseFilter):
    def __init__(self, cfg=ColorGridFilterDefaults()):
        super().__init__(cfg)
        self.color = torch.tensor(self.cfg.color, dtype=torch.float32, device='cuda') / 255.

    def transform(self, image, *args, **kwargs):
        if len(image.shape) > 2:
            if self.cfg.apply_x:
                image[:3, :, ::self.cfg.step_x] = self.color[:, None, None]
            if self.cfg.apply_y:
                image[:3, ::self.cfg.step_y, :] = self.color[:, None, None]
        elif len(image.shape) == 2:
            if self.cfg.apply_x:
                image[:, ::self.cfg.step_x] = self.color[0]
            if self.cfg.apply_y:
                image[::self.cfg.step_y, :] = self.color[0]
        return image


@dataclass
class GradientColorizeFilterDefaults:
    colormap: str = 'nipy_spectral'


class GradientColorizeFilter(BaseFilter):
    def __init__(self, cfg=GradientColorizeFilterDefaults()):
        super(GradientColorizeFilter, self).__init__(cfg)

    def init_content(self, content):
        return ColorMap(self.cfg.colormap)

    def transform(self, image, *args, **kwargs):
        return self.content.process_grad_grayscale(image, *args, **kwargs)


class GradientColorizeCUDAFilter(BaseFilter):
    def __init__(self, cfg=GradientColorizeFilterDefaults()):
        super(GradientColorizeCUDAFilter, self).__init__(cfg)

    def init_content(self, content):
        return ColorConverterCUDA(self.cfg.colormap)

    def transform(self, image, *args, **kwargs):
        return self.content.process_grad_grayscale(image, *args, **kwargs)


@dataclass
class RandomVerticalLinesFilterDefaults:
    count: int = 15
    change_every: int = 1


class RandomVerticalLinesFilter(BaseFilter):
    def __init__(self, cfg=RandomVerticalLinesFilterDefaults()):
        super(RandomVerticalLinesFilter, self).__init__(cfg)

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


class RandomVerticalLinesCUDAFilter(BaseFilter):
    def __init__(self, cfg=RandomVerticalLinesFilterDefaults()):
        super(RandomVerticalLinesCUDAFilter, self).__init__(cfg)
        self._index = torch.randint(0, 1, (self.cfg.count,), dtype=torch.int64)

    def transform(self, image, *args, **kwargs):
        w = image.shape[-1]
        if self.tick % self.cfg.change_every == 0:
            self._index = torch.randint(0, w, (self.cfg.count,), dtype=torch.int64)
        self.tick += 1
        if len(image.shape) == 2:
            image[:, self._index] = torch.rand((1,), dtype=torch.float32)  # TODO: FIX THIS CASE
        else:
            image[..., :3, :, self._index] = torch.rand((3,), dtype=torch.float32, device='cuda')[None, :, None, None]
        return image
