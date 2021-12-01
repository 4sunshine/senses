import numpy as np
import time

from dataclasses import dataclass
from enum import Enum

from typing import List


class BidirectionalIterator(object):
    """https://stackoverflow.com/a/2777223"""
    def __init__(self, collection):
        self.collection = collection
        self.index = 0

    def next(self):
        try:
            result = self.collection[self.index]
            self.index += 1
        except IndexError:
            raise StopIteration
        return result

    def prev(self):
        self.index -= 1
        if self.index < 0:
            raise StopIteration
        return self.collection[self.index]

    def __iter__(self):
        return self


class SourceType(Enum):
    transparency = 0
    region = 1
    keypoint = 2
    event = 3
    effect = 4


class DeviceType(Enum):
    cpu = 10
    cuda = 11


class RegionType(Enum):
    face_box = 20
    body_box = 21
    text_box = 22
    picture_box = 23
    video_box = 24
    drawing_box = 25


class EventType(Enum):
    keypress = 30
    born = 31
    time_passed = 32
    destroyed = 33


@dataclass
class SourceDefaultConfig:
    name: str = 'default_source'
    type: SourceType = 0
    device: DeviceType = 0
    url: list = None  # SOURCE OF URLS
    birth: EventType = 33  # EVENT CLASS
    appear: EventType = 32
    disappear: EventType = 30
    destroy: EventType = 32
    fade_in_time: float = 1  # time in seconds
    fade_out_time: float = 1


@dataclass
class TransparencyConfig(SourceDefaultConfig):
    type = 0
    name = 'transparency'
    device = 11

@dataclass
class LayerConfig:
    name: str = 'default_layer'
    transparency: List[SourceDefaultConfig] = None
    region: List[SourceDefaultConfig] = None
    effect: List[SourceDefaultConfig] = None
    event: List[SourceDefaultConfig] = None


class MediaDataLayer_EXP(object):
    def __init__(self, cfg, global_start_time=None):
        self.cfg = cfg
        self.alpha_source = self.init_alpha()  # LIST OF ALPHA SOURCES
        self.roi_source = self.init_roi()  # LIST OF ROI SOURCES
        self.effect_source = self.init_effect()  # LIST OF EFFECT SOURCES
        self.event_tracker = self.init_event_tracker()  # LIST OF EVENT TRACKERS SOURCES
        self._buffer = np.zeros(cfg.size[::-1] + (3,), dtype=np.uint8)
        self._alpha_buffer = np.zeros(cfg.size[::-1] + (1,), dtype=np.float32)
        self._alpha_compose = None

        self._stream = {
            'rgb_buffer_cpu': np.zeros(cfg.size[::-1] + (3,), dtype=np.uint8),
            'alpha_cpu': np.ones(cfg.size[::-1] + (1,), dtype=np.float32),
            'rgb_buffer_cuda': None,  # USE FLOAT RGB REPRESENTATION
            'alpha_cuda': None,
            'global_time': 0.,
            'rois': dict(),
            'keypoints': dict(),
            'events': dict(),
            'shared_rois': dict(),
            'shared_keypoints': dict(),
            'shared_events': dict(),
            'data': dict(),
            'shared_data': dict(),
            'local_start': time.time(),
            'global_start': time.time() if global_start_time is None else global_start_time,
            'current_index': 0,
            'tick': 0,
        }
        self._local_start = self._stream['local_start']
        self._global_start = self._stream['global_start']

    def start_time(self):
        return self._stream['local_start']

    def time_since_local_start(self):
        return time.time() - self._local_start

    def time_since_global_start(self):
        return time.time() - self._global_start

    def tick(self):
        return self._stream['tick']

    def update_alpha(self):
        self._alpha_id
        # POOR ARCHITECTURE
        config = self.cfg.alpha_source
        if config is None:
            return None
        else:
            return self.cfg.alpha_source(self.cfg.alpha_source_cfg)

    def init_roi(self):
        return None

    def init_effect(self):
        return None

    def init_event_tracker(self):
        return None

    def alpha(self, buffer):
        if self.alpha_source is not None:
            return self.alpha_source.process_image(buffer)
        else:
            return None, None

    def roi(self, buffer, alpha):
        if self.roi_source is not None:
            self.roi_source(buffer, alpha)
        else:
            return dict(), dict()

    def effect(self, buffer, alpha, roi, keypoints):
        if self.effect_source is not None:
            return self.effect_source(buffer, alpha, roi, keypoints)
        else:
            return buffer, alpha

    def read(self):
        return False, None

    def is_refreshed(self):
        success, data = self.read()
        if success:
            self._buffer = data
            self._alpha_buffer, self._alpha_compose = self.alpha(self._buffer)
        return success

    def float_render(self, roi, keypoints):
        if self.effect is not None:
            buffer, alpha = self.effect(self._buffer, self._alpha_buffer, roi, keypoints)
            return buffer, alpha, self._alpha_compose
        else:
            return self._buffer, self._alpha_buffer, self._alpha_compose

    def salient_regions(self):
        roi, keypoints = self.roi(self._buffer, self._alpha_buffer)
        return roi, keypoints

    def update(self, events):
        pass

    def events(self, roi, keypoints):
        if self.event_tracker is not None:
            return self.event_tracker(roi, keypoints)
        else:
            return []