import cv2

from visio.source import *
from dataclasses import dataclass


__all__ = ['Keypress', 'EVENT_MAPPING']


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
        k = cv2.waitKey(1)
        print(k)

    def default_config(self):
        return SpaceButtonConfig()


class EmptyStreamProcessor:
    def process_stream(self, stream):
        pass


EVENT_MAPPING = {
    Event.Keypress: Keypress,
}
