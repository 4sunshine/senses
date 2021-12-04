from dataclasses import dataclass
from enum import Enum


__all__ = ['SourceType', 'DeviceType', 'Enum', 'Source']


class SourceType(Enum):
    transparency = 0
    region = 1
    keypoint = 2
    event = 3
    effect = 4
    stream = 5
    dummy = 6


class DeviceType(Enum):
    cpu = 10
    cuda = 11


@dataclass
class SourceDefaultConfig:
    name: str
    type: SourceType
    device: DeviceType
    url: None  # SOURCE OF URLS
    # birth: EventType = None # EVENT CLASS
    # appear: EventType = None
    # disappear: EventType = None
    # destroy: EventType = None
    # fade_in_time: float = None # time in seconds
    # fade_out_time: float = None


class Source(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else self.default_config()
        self.init_source()

    def default_solution(self):
        return self

    def default_config(self):
        return SourceDefaultConfig(name='default', type=SourceType.dummy, device=DeviceType.cpu, url=None)

    def init_source(self):
        pass

    def process_stream(self, stream):
        pass

    def close(self):
        pass


class Composition(Source):
    def __init__(self, cfg):
        assert isinstance(cfg, list)
        types = [c.type for c in cfg]
        assert types.count(types[0]) == len(types)
        # sources = list(map(lambda x: ))
        # source = globals()[attribute.upper()][source_cfg.solution](source_cfg)
        # super(Composition, self).__init__(cfg)
        # self.sources
        # assert len(set())

    def init_source(self):
        pass

    def process_stream(self, stream):
        pass

    def close(self):
        pass

