from visio.source_head import *
from dataclasses import dataclass


__all__ = ['EventType', 'Keypress']


class EventType(Enum):
    started = 0
    finished = 1
    appear = 2
    fade_out = 3
    keypress = 4
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


class Keypress(EventSource):
    def process_stream(self, stream):
        pass


class EmptyStreamProcessor:
    def process_stream(self, stream):
        pass
