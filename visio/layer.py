import numpy as np
import time
import torch
import cv2

from typing import Union
from dataclasses import dataclass

from visio.factory import SourceFactory
from visio.source import Device, BidirectionalIterator


@dataclass
class MediaLayerConfig:
    name: str = 'default_layer'
    stream: list = None
    transparency: list = None
    region: list = None
    effect: list = None
    event: list = None


class MediaLayer(object):
    def __init__(self, cfg, global_start_time=None, source_factory=SourceFactory()):
        self.cfg = cfg
        self.SF = source_factory
        assert isinstance(cfg.transparency, list)
        self.stream_source, self.stream_config = self.init_source(cfg.stream)
        self.summary = None  # SHORT DESCRIPTION OR EMBEDDING BERT FOR EXAMPLE
        self.transparency_source, self.transparency_configs = self.init_source(cfg.transparency)  # LIST OF ALPHA SOURCES
        self.region_source, self.region_configs = self.init_source(cfg.region)
        self.effect_source, self.effect_configs = self.init_source(cfg.effect)
        self.event_source, self.event_configs = self.init_source(cfg.event)

        self._stream = {
            'rgb_buffer_cpu': np.zeros(self.stream_source.cfg.size[::-1] + (3,), dtype=np.uint8),
            'alpha_cpu': np.ones(self.stream_source.cfg.size[::-1] + (1,), dtype=np.float32),
            'rgb_buffer_cuda': None,  # USE FLOAT RGB REPRESENTATION
            'alpha_cuda': None,
            'global_time': 0.,
            'rois': {
                'person_region': np.array([-1, -1, -1, -1], dtype=np.int32),
                'face': np.array([-1, -1, -1, -1], dtype=np.int32),
            },
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
            'new_ready': True,
        }

        self._local_start = self._stream['local_start']
        self._global_start = self._stream['global_start']

    def _do_stream(self):
        self.stream_source.read_stream(self._stream)
        self.transparency_source.process_stream(self._stream)
        self.region_source.process_stream(self._stream)
        self.event_source.process_stream(self._stream)
        self.effect_source.process_stream(self._stream)
        self._stream['tick'] += 1

    def render_cuda(self):
        self._do_stream()
        if self._stream['alpha_cuda'] is None:
            return self._stream['rgb_buffer_cuda'].clone(), None
        else:
            return self._stream['rgb_buffer_cuda'].clone(), self._stream['alpha_cuda'].clone()

    def world_act(self, regions, keypoints, events):
        pass

    def world_feedback(self, feedback):
        pass

    def start_time(self):
        return self._stream['local_start']

    def time_since_local_start(self):
        return time.time() - self._local_start

    def time_since_global_start(self):
        return time.time() - self._global_start

    def close(self):
        self.stream_source.close()
        self.transparency_source.close()
        self.effect_source.close()
        self.event_source.close()
        self.region_source.close()

    def send_events(self, events):
        self.stream_source.send_events(events)

    def listen_events(self, events):
        self.stream_source.listen_events(events)

    def tick(self):
        return self._stream['tick']

    def init_source(self, cfg):
        if cfg:
            configs = BidirectionalIterator(cfg)
            source_cfg = configs.get()
            configs.next()
            return self.SF.init_source(source_cfg), configs
        else:
            return self.SF.empty_source(), BidirectionalIterator([])

    def update_source(self, attribute, next=True):
        assert attribute in ('transparency', 'region', 'effect', 'event')  # ATTRIBUTE SHOULD BE ENUM OF SOURCE_TYPE
        attribute_configs = getattr(self, '_'.join([attribute, 'configs']))
        try:
            config = attribute_configs.next() if next else attribute_configs.prev()
            return self.SF.init_source(config)
        except IndexError:
            return self.SF.empty_source()



