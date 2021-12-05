from visio.source import *
from dataclasses import dataclass
from typing import Union

import cv2
import torch


__all__ = ['Stream', 'WebCamCV2', 'WebCamCV2Config', 'STREAM_MAPPING']


class Stream(Enum):
    Video = 0
    Presentation = 1
    Text = 2
    Image = 3
    WebCam = 4
    Evolution = 5


class StreamReader(object):
    def __init__(self, cfg=None):
        self.cfg = self.default_config() if cfg is None else cfg
        self._tick = -1

    def tick(self):
        self._tick += 1
        return self._tick

    def default_config(self):
        return None

    def build_config(self, prompt=None):
        pass

    def read_stream(self, stream):
        stream['new_ready'] = False

    def close(self):
        pass


@dataclass
class WebCamCV2Config:
    input_id: int = 0
    solution = Stream.WebCam
    type = SourceType.stream
    name = 'webcam_cv2'
    size: Union[int, int] = (1280, 720)
    flip: bool = True
    fps: int = 25
    rgb: bool = True


class WebCamCV2(StreamReader):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.cap = cv2.VideoCapture(self.cfg.input_id)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.size[1])

    def default_config(self):
        return WebCamCV2Config()

    def read(self):
        success, frame = self.cap.read()
        if success:
            if not self.cfg.flip and self.cfg.rgb:
                return success, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif self.cfg.flip and self.cfg.rgb:
                return success, cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            elif self.cfg.flip and not self.cfg.rgb:
                return success, cv2.flip(frame, 1)
            else:
                return success, frame
        else:
            return success, frame

    def read_stream(self, stream):
        success, frame = self.cap.read()
        stream['new_ready'] = success
        if self.cfg.flip:
            stream['rgb_buffer_cpu'] = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        else:
            stream['rgb_buffer_cpu'] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # if self.cfg.device == DeviceType.cuda:
        #     stream['cuda_buffer_gpu'] = to_tensor(stream['rgb_buffer_cpu']) / 255.

    def close(self):
        self.cap.release()


@dataclass
class EvolutionConfig:
    input_id = 0
    solution = Stream.Evolution
    type = SourceType.stream
    url = ''
    device = Device.cuda
    name = 'evolution'
    size = (1280, 720)
    initial_state = 'white'  # LATER CONSIDER OTHER STATES ('white', 'noise', 'black')


class Evolution(StreamReader):
    def __init__(self, cfg=None):
        super().__init__(cfg)

    def default_config(self):
        return EvolutionConfig()

    def initial_state(self):
        if self.cfg.device == Device.cuda:
            if self.cfg.initial_state == 'white':
                state = torch.ones((3,) + self.cfg.size[::-1], dtype=torch.float32, device='cuda')
            elif self.cfg.initial_state == 'noise':
                state = torch.rand((3,) + self.cfg.size[::-1], dtype=torch.float32, device='cuda')
            else:
                state = torch.zeros((3,) + self.cfg.size[::-1], dtype=torch.float32, device='cuda')
        else:
            state = None
        return state, None

    def read_stream(self, stream):
        if self.tick() == 0:
            stream['rgb_buffer_cuda'], stream['alpha_cuda'] = self.initial_state()
        stream['new_ready'] = True
        # if self.cfg.device == DeviceType.cuda:
        #     stream['cuda_buffer_gpu'] = to_tensor(stream['rgb_buffer_cpu']) / 255.


STREAM_MAPPING = {
    Stream.WebCam: WebCamCV2,
    Stream.Evolution: Evolution,
}
