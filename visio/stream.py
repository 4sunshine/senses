import PIL.Image
import numpy as np

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
    Images = 3
    WebCam = 4
    Evolution = 5


class StreamReader(Source):
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
    def __init__(self, cfg=None, data=None):
        super().__init__(cfg, data)
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
        if state is not None:
            alpha = torch.ones(state.shape[1:], dtype=torch.float32, device='cuda')
        else:
            alpha = None
        return state, alpha

    def read_stream(self, stream):
        if self.tick() == 0:
            stream['rgb_buffer_cuda'], stream['alpha_cuda'] = self.initial_state()
        stream['new_ready'] = True
        # if self.cfg.device == DeviceType.cuda:
        #     stream['cuda_buffer_gpu'] = to_tensor(stream['rgb_buffer_cpu']) / 255.


@dataclass
class ImagesCUDAConfig:
    input_id = 0
    solution = Stream.Images
    type = SourceType.stream
    url = []
    device = Device.cuda
    name = 'evolution'
    size = (1280, 720)
    initial_state = 'white'  # LATER CONSIDER OTHER STATES ('white', 'noise', 'black')


class ImagesCUDA(StreamReader):
    def __init__(self, cfg=None, data=None):
        data = data or cfg.url
        super(ImagesCUDA, self).__init__(cfg, data)

    def init_source(self, images):
        # LATER CHECK EQUAL IMAGES SIZES
        buffer = []
        for img in images:
            if isinstance(img, torch.Tensor):
                assert img.shape[-2:] == self.cfg.size[::-1]
                buffer.append(img.cuda())
            elif isinstance(img, PIL.Image.Image):
                assert img.size == self.cfg.size
                buffer.append(self.rgb_cpu_to_cuda(np.array(img)))
            elif isinstance(img, np.ndarray):
                assert img.shape[:2] == self.cfg.size[::-1]
                buffer.append(self.rgb_cpu_to_cuda(img))
            elif isinstance(img, str):
                img = PIL.Image.open(img).convert('RGBA')  #cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
                assert img.size == self.cfg.size
                buffer.append(self.rgb_cpu_to_cuda(np.array(img)))
            else:
                print(f'invalid img {img}')
                continue
        if buffer:
            buffer = torch.stack(buffer).cuda()
        return buffer

    def default_config(self):
        return ImagesCUDAConfig()

    def listen_events(self, events):
        if Event.KeyPressLeft in events:
            self.prev()
        elif Event.KeyPressRight in events:
            self.next()

    def read_stream(self, stream):
        if self.new_data_ready:
            stream['new_ready'] = True
            if self.cfg.device == Device.cuda:
                data = self.data.get()
                stream['rgb_buffer_cuda'] = data[:3]
                if len(data) == 4:
                    stream['alpha_cuda'] = data[3:, :, :]
            else:
                stream['rgb_buffer_cpu'] = self.data.get()
            self.new_data_ready = False
        else:
            stream['new_ready'] = False

    def __len__(self):
        return len(self.data)


STREAM_MAPPING = {
    Stream.WebCam: WebCamCV2,
    Stream.Evolution: Evolution,
    Stream.Images: ImagesCUDA,
}

