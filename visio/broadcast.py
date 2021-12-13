import cv2
import torch
import time
import subprocess

from visio.source import Device, Source, Event
from visio.event import Keypress
from typing import Union
from dataclasses import dataclass
from enum import Enum


class BroadcastType(Enum):
    window = 0
    av = 1
    fake_cam = 2


class Broadcast(Source):
    def default_config(self):
        return None

    def close(self):
        for layer in self.data:
            layer.close()

    def send(self, image):
        pass

    def broadcast(self):
        if self.cfg.device == Device.cuda:
            with torch.no_grad():
                self.broadcast_cuda()
        else:
            self.broadcast_cpu()

    def listen_events(self, events):
        pass

    def send_events(self, events):
        self.s_events.update_events(events)

    def broadcast_cuda(self):
        while True:
            start = time.time()  # LATER ADD TIMING OPT
            result = self.layers_process_cuda()
            self.send(result)
            need_break = self.handle_events()
            print((1 / (time.time() - start)))

            if need_break:
                self.close()
                break

    def broadcast_cpu(self):
        pass

    def layers_process_cuda(self):
        result = None
        for l in self.data:
            layer, alpha = l.render_cuda()
            # self.events += shared_events
            # result = layer * alpha
            if result is None:
                if alpha is None:
                    result = layer
                else:
                    result = layer * alpha
            else:
                result = layer * alpha + result * (1 - alpha)
        return self.rgb_cuda_to_cpu(result)  # result.mul(255).byte().cpu().permute(1, 2, 0).numpy()

    def handle_events(self):
        events = set()
        self.send_events(events)
        for l in self.data:
            l.send_events(events)
        for l in self.data:
            l.listen_events(events)
        self.listen_events(events)
        return Event.Escape in events


@dataclass
class BroadcastWindowConfig:
    name = 'broadcast_window'
    device = Device.cuda
    solution = BroadcastType.window
    window_name: str = 'video_input'
    window_position: Union[int, int] = (600, 0)
    fps: int = 25
    rgb: bool = True


class BroadcastWindow(Broadcast):
    def __init__(self, cfg=None, data=None):
        super(BroadcastWindow, self).__init__(cfg, data)
        self.s_events = Keypress()
        self.data = list(self.data.collection)
        self.window_name = str(self.cfg.window_name)
        cv2.namedWindow(self.window_name)
        self._x, self._y = self.cfg.window_position
        cv2.moveWindow(self.window_name, self._x, self._y)

    def default_config(self):
        return BroadcastWindowConfig()

    def close(self):
        super(BroadcastWindow, self).close()
        cv2.destroyAllWindows()

    def send(self, image):
        cv2.imshow(self.window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


@dataclass
class AVBroadcastConfig:
    name = 'avbroadcast'
    device = Device.cuda
    url = []  # SOURCE OF URLS
    input_id: int = 0
    solution = BroadcastType.av
    window_name: str = 'video_input'
    size: Union[int, int] = (1280, 720)
    window_position: Union[int, int] = (600, 0)
    fps: int = 25
    audio_device: str = 'hw:1,0'
    output_filename: str = 'out.mp4'
    window_show = True


class AVBroadcast(BroadcastWindow):
    def __init__(self, cfg=None, data=None):
        super(AVBroadcast, self).__init__(cfg, data)
        command = ['ffmpeg',
                   '-y',
                   # INPUT VIDEO STREAM
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-s', f'{self.cfg.size[0]}x{self.cfg.size[1]}',
                   '-pix_fmt', 'rgb24',  # 'bgr24'
                   '-use_wallclock_as_timestamps', '1',
                   '-i', '-',
                   # INPUT AUDIO STREAM
                   '-f', 'alsa',
                   '-ac', '1',
                   # '-channels', '1',
                   # '-sample_rate', '44100',
                   '-i', self.cfg.audio_device,
                   '-acodec', 'aac',
                   # OUTPUT VIDEO OPTIONS
                   '-vcodec', 'libx264',
                   '-preset', 'ultrafast',
                   # '-tune', 'film',
                   '-qp', '0',
                   # OUTPUT AUDIO OPTIONS
                   '-acodec', 'mp2',
                   self.cfg.output_filename]

        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def default_config(self):
        return AVBroadcastConfig()

    def send(self, image):
        self.process.stdin.write(image.tobytes())
        if self.cfg.window_show:
            super(AVBroadcast, self).send(image)

    def close(self):
        self.process.stdin.close()
        self.process.terminate()
        self.process.wait()
        super(AVBroadcast, self).close()
        print(self.process.poll())
