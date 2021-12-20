from dataclasses import dataclass
from enum import Enum

import torch
from torchvision.transforms.functional import to_tensor


__all__ = ['SourceType', 'Device', 'Enum', 'Source', 'SOURCE_MAPPING', 'Composition', 'BidirectionalIterator', 'Event',
           'SourceInterface']


@dataclass
class SourceInterface:
    config: None
    data: None
    events: None


class BidirectionalIterator(object):
    """https://stackoverflow.com/a/2777223"""
    def __init__(self, collection):
        self.collection = collection
        self._index = 0

    def next(self):
        result = self.collection[self._index]
        self._index += 1
        if self._index > len(self) - 1:
            self._index = len(self) - 1
        return result

    def prev(self):
        self._index -= 1
        if self._index < 0:
            self._index = 0
        return self.collection[self._index]

    def get(self, item=None):
        if isinstance(item, str):
            return getattr(self.collection[self.id()], item)
        else:
            return self.collection[max(0, min(self.id(), len(self)))]

    def id(self):
        return self._index #- 1

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.collection)


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


class Event(Enum):
    started = 0
    finished = 1
    appear = 2
    fade_out = 3
    Keypress = 4
    keypoint = 5
    openhand = 6
    finger_at = 7
    face_at = 8
    next_element = 9
    KeyPressRight = 10
    KeyPressLeft = 11
    Escape = 12


@dataclass
class WaitingEvents:
    next_data = Event.KeyPressRight
    prev_data = Event.KeyPressLeft


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
    def __init__(self, cfg=None, data=None, events=None):
        self.cfg = self.default_config() if cfg is None else cfg
        if data is None:
            data = []
        self.data = BidirectionalIterator(self.init_source(data))  #.next()
        # self.data.next()
        self.l_events = self.init_events(events['l_events'] if events else None)
        self.s_events = self.init_events(events['s_events'] if events else None)
        self.new_data_ready = True
        #self.waiting_events = self.cfg.events or None
        self._tick = -1

    def init_events(self, events):
        return events

    def default_solution(self):
        return self

    def rgb_cpu_to_cuda(self, image):
        return torch.from_numpy(image).type(torch.float32).permute(2, 0, 1).cuda() / 255. #o_tensor(image) / 255.

    def rgb_cuda_to_cpu(self, image):
        if len(image.shape) == 3:
            return image.mul(255).byte().cpu().permute(1, 2, 0).numpy()
        else:
            return image[0].mul(255).byte().cpu().permute(1, 2, 0).numpy()

    def tick(self):
        self._tick += 1
        return self._tick

    def default_config(self):
        return SourceDefaultConfig(name='default', type=SourceType.dummy, device=Device.cpu, url=None)

    def init_source(self, data):
        return data

    def process_stream(self, stream):
        pass

    def close(self):
        pass

    def listen_events(self, events):
        pass
        # if self.l_events:
        #     if self.l_events.next_data in events:
        #         self.next()
        #     elif self.l_events.prev_data in events:
        #         self.prev()

    def send_events(self, events):
        pass

    def next(self):
        self.data.next()
        self.new_data_ready = True

    def prev(self):
        self.data.prev()
        self.new_data_ready = True

    def __len__(self):
        return 1  # TODO: CHECK IF LEN(DATA) CAN BE USED


class Composition(Source):
    def __init__(self, sources, data=None):
        assert isinstance(sources, list)
        print('COMP_HERE')
        print(sources)
        configs = [s.cfg for s in sources]
        types = [cfg.type for cfg in configs]
        assert types.count(types[0]) == len(types)
        super(Composition, self).__init__(cfg=configs, data=data)
        self.sources = sources

    def process_stream(self, stream):
        for s in self.sources:
            s.process_stream(stream)

    def close(self):
        for s in self.sources:
            s.close()


SOURCE_MAPPING = {
    SourceType.dummy: Source,
}

