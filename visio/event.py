import cv2.cv2

from visio.source import *
from dataclasses import dataclass


__all__ = ['Event', 'Keypress', 'EVENT_MAPPING']


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


class EventSource(Source):
    def process_stream(self, stream):
        pass

    def listen(self):
        pass


@dataclass
class SpaceButtonConfig:
    solution = Event.Keypress
    type = SourceType.event
    name = 'keypress'
    device = Device.cpu
    ff_code = 32


class Keypress(EventSource):
    def process_stream(self, stream):
        if cv2.waitKey(1) & 0xFF == self.cfg.ff_code:
            print('LOL')

    def default_config(self):
        return SpaceButtonConfig()


class EmptyStreamProcessor:
    def process_stream(self, stream):
        pass


EVENT_MAPPING = {
    Event.Keypress: Keypress,
}
