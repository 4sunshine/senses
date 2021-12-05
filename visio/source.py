from dataclasses import dataclass
from enum import Enum


__all__ = ['SourceType', 'Device', 'Enum', 'Source', 'SOURCE_MAPPING', 'Composition']


class SourceType(Enum):
    transparency = 0
    region = 1
    keypoint = 2
    event = 3
    effect = 4
    stream = 5
    dummy = 6


class Device(Enum):
    cpu = 10
    cuda = 11


@dataclass
class SourceDefaultConfig:
    name: str
    type: SourceType
    device: Device
    url: None  # SOURCE OF URLS
    # birth: EventType = None # EVENT CLASS
    # appear: EventType = None
    # disappear: EventType = None
    # destroy: EventType = None
    # fade_in_time: float = None # time in seconds
    # fade_out_time: float = None


class Source(object):
    def __init__(self, cfg=None):
        self.cfg = self.default_config() if cfg is None else cfg
        self.init_source()

    def default_solution(self):
        return self

    def default_config(self):
        return SourceDefaultConfig(name='default', type=SourceType.dummy, device=Device.cpu, url=None)

    def init_source(self):
        pass

    def process_stream(self, stream):
        pass

    def close(self):
        pass


class Composition(Source):
    def __init__(self, sources):
        assert isinstance(sources, list)
        configs = [s.cfg for s in sources]
        types = [cfg.type for cfg in configs]
        assert types.count(types[0]) == len(types)
        super(Composition, self).__init__(cfg=configs)
        self.sources = sources

    def init_source(self):
        pass

    def process_stream(self, stream):
        for s in self.sources:
            s.process_stream(stream)

    def close(self):
        for s in self.sources:
            s.close()


SOURCE_MAPPING = {
    SourceType.dummy: Source,
}

