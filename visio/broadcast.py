import cv2
import numpy as np
import torch
import time
import subprocess

from visio.source import Device, Source, Event
from visio.event import Keypress
from typing import Union
from dataclasses import dataclass
from enum import Enum
import threading


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
            start_write = time.time()
            self.send(result)
            print(f'WRITING: {time.time()-start_write}')
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

# import threading
#
# c = threading.Condition()
#
# class Thread_A(threading.Thread):
#     def __init__(self, name):
#         threading.Thread.__init__(self)
#         self.name = name
#
#     def run(self):
#         global flag
#         global val     #made global here
#         while True:
#             c.acquire()
#             if flag == 0:
#                 print "A: val=" + str(val)
#                 time.sleep(0.1)
#                 flag = 1
#                 val = 30
#                 c.notify_all()
#             else:
#                 c.wait()
#             c.release()

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
    audio_device: str = 'hw:2,0'
    output_filename: str = 'out.mp4'
    window_show = True


class AVBroadcast(BroadcastWindow):
    def __init__(self, cfg=None, data=None):
        super(AVBroadcast, self).__init__(cfg, data)
        self.command = ['ffmpeg',
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
                        '-thread_queue_size', '1024',
                        # '-channels', '1',
                        # '-sample_rate', '44100',
                        '-i', self.cfg.audio_device,
                        #'-acodec', 'aac',
                        # OUTPUT VIDEO OPTIONS
                        '-vcodec', 'h264', #'h264_nvenc',#'libx264',
                        '-f', 'mp4',
                        '-preset', 'fast', #'ultrafast',
                        '-threads', '6',
                        # '-crf', '0',
                        # '-preset', 'ultrafast',
                        # # '-tune', 'film',
                        '-qp', '0',
                        # OUTPUT AUDIO OPTIONS
                        '-acodec', 'aac',
                        self.cfg.output_filename]
        self.process = None
        # self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def default_config(self):
        return AVBroadcastConfig()

    def send(self, image):
        bytes_image = image.tobytes()
        self.process.stdin.write(bytes_image)
        if self.cfg.window_show:
            cv2.imshow(self.window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def close(self):
        self.process.stdin.close()
        self.process.terminate()
        self.process.wait()
        super(AVBroadcast, self).close()
        print(self.process.poll())

    def broadcast_cuda(self):
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE)
        result = None
        while True:
            start = time.time()  # LATER ADD TIMING OPT
            if result is not None:
                #start_write = time.time()
                result_bytes = result.tobytes()
                self.process.stdin.write(result_bytes)  #.stdin.write(result_bytes)
                #print(f'WRITING: {time.time() - start_write}')
            new_ready = False
            result = self.layers_process_cuda()
            new_ready = True
            #start_write = time.time()
            # self.send(result)
            #print(f'WRITING: {time.time()-start_write}')
            if self.cfg.window_show:
                cv2.imshow(self.window_name, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            need_break = self.handle_events()
            print((1 / (time.time() - start)))

            if need_break:
                self.close()
                break

    def broadcast(self):
        if self.cfg.device == Device.cuda:
            with torch.no_grad():
                self.broadcast_cuda()
        else:
            self.broadcast_cpu()


import socket

@dataclass
class AVBroadcastServerConfig:
    name = 'avbroadcast'
    device = Device.cuda
    url = []  # SOURCE OF URLS
    input_id: int = 0
    solution = BroadcastType.av
    window_name: str = 'video_input'
    size: Union[int, int] = (1280, 720)
    window_position: Union[int, int] = (600, 0)
    fps: int = 25
    audio_device: str = 'hw:2,0'
    output_filename: str = 'out.mp4'
    window_show = True


class AVBroadcastServer(object):
    HOST = '127.0.0.1'
    PORT = 65432

    def __init__(self):
        self.command = ['ffmpeg',
                        '-y',
                        # INPUT VIDEO STREAM
                        '-f', 'rawvideo',
                        '-vcodec', 'rawvideo',
                        '-s', f'{1280}x{720}',
                        '-pix_fmt', 'rgb24',  # 'bgr24'
                        '-use_wallclock_as_timestamps', '1',
                        '-i', '-',
                        # INPUT AUDIO STREAM
                        '-f', 'alsa',
                        '-ac', '1',
                        '-thread_queue_size', '1024',
                        # '-channels', '1',
                        # '-sample_rate', '44100',
                        '-i', 'hw:2,0',
                        #'-acodec', 'aac',
                        # OUTPUT VIDEO OPTIONS
                        '-vcodec', 'libx264',
                        '-f', 'mp4',
                        '-preset', 'ultrafast',
                        '-threads', '6',
                        # '-crf', '0',
                        # '-preset', 'ultrafast',
                        # # '-tune', 'film',
                        '-qp', '0',
                        # OUTPUT AUDIO OPTIONS
                        '-acodec', 'aac',
                        'out_serv.mp4']


    def broadcast(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((self.HOST, self.PORT))
        s.listen()
        s.settimeout(10)
        conn, addr = s.accept()
        print('Connection established')
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE)
        i = 0
        size = 3 * 1280 * 720
        full_data = np.empty((3 * size,), dtype=np.uint8)
        chunkie = np.empty((size,), dtype=np.uint8)
        cur_len = 0
        cur_pos = 0
        while True:
            serv_start = time.time()
            data = conn.recv(size)
            result = np.frombuffer(data, dtype=np.uint8)
            res_len = len(result)
            full_data[cur_pos: cur_pos + res_len] = result
            cur_pos += res_len
            conn.sendall(b'ok')
            print(f'SERVER TIME: {time.time() - serv_start:.04f}')
            if cur_pos >= size:
                buffer = full_data[:size]
                frame_bytes = buffer.copy().tobytes()
                full_data = np.roll(full_data, -size)
                cur_pos %= size
                self.process.stdin.write(frame_bytes)
                buffer = buffer.reshape((720, 1280, 3))
                cv2.imshow('F', buffer)
                cv2.waitKey(1)
            # result = np.frombuffer(data, dtype=np.uint8)
            # print(result.shape)
            # assert result.shape[0] == size
            # result = result.reshape((720, 1280, 3))
            # if True:  #self.cfg.window_show:
            #     #img = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            #     cv2.imshow('F', result)
            #     cv2.waitkey(1)
            i += 1
            if i == 500:
                print('BREAKING SERVER')
                s.close()
                conn.close()
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait()
                print(self.process.poll())
                break


@dataclass
class AVBroadcastCliConfig:
    name = 'avbroadcast'
    device = Device.cuda
    url = []  # SOURCE OF URLS
    input_id: int = 0
    solution = BroadcastType.av
    window_name: str = 'video_input'
    size: Union[int, int] = (1280, 720)
    window_position: Union[int, int] = (600, 0)
    fps: int = 25
    audio_device: str = 'hw:2,0'
    output_filename: str = 'out.mp4'
    window_show = True


class AVBroadcastCli(BroadcastWindow):
    def __init__(self, cfg=None, data=None):
        super(AVBroadcastCli, self).__init__(cfg, data)
        self.process = None
        # self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def default_config(self):
        return AVBroadcastConfig()

    def send(self, image):
        bytes_image = image.tobytes()
        self.process.stdin.write(bytes_image)
        if self.cfg.window_show:
            cv2.imshow(self.window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def close(self):
        self.process.stdin.close()
        self.process.terminate()
        self.process.wait()
        super(AVBroadcast, self).close()
        print(self.process.poll())

    def broadcast_cuda(self):
        # self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE)
        HOST = '127.0.0.1'  # The server's hostname or IP address
        PORT = 65432  # The port used by the server

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST, PORT))

        result = None
        while True:
            start = time.time()  # LATER ADD TIMING OPT
            if result is not None:
                start_write = time.time()
                result_bytes = result.tobytes()
                s.sendall(result_bytes)
                print(f'WRITING: {time.time() - start_write}')
                data = s.recv(2)
                print(repr(data))
            result = self.layers_process_cuda()
            #start_write = time.time()
            # self.send(result)
            #print(f'WRITING: {time.time()-start_write}')
            if False:  #self.cfg.window_show:
                img = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                cv2.imshow(self.window_name, img)
            need_break = self.handle_events()
            print((1 / (time.time() - start)))

            if need_break:
                self.close()
                s.close()
                break

    def broadcast(self):
        if self.cfg.device == Device.cuda:
            with torch.no_grad():
                self.broadcast_cuda()
        else:
            self.broadcast_cpu()



if __name__ == '__main__':
    server = AVBroadcastServer()
    server.broadcast()
