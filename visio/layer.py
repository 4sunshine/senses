import numpy as np
import time
import torch
import cv2

from types import MappingProxyType
from typing import Union

from dataclasses import dataclass
from enum import Enum

from typing import List

from detect.face import MPSimpleFaceDetector


def defaults_mapping(cfg):
    return MappingProxyType(cfg)


class BidirectionalIterator(object):
    """https://stackoverflow.com/a/2777223"""
    def __init__(self, collection):
        self.collection = collection
        self._index = 0

    def next(self):
        try:
            result = self.collection[self._index]
            self._index += 1
        except IndexError:
            raise StopIteration
        return result

    def prev(self):
        self._index -= 1
        if self._index < 0:
            raise StopIteration
        return self.collection[self._index]

    def get(self, item=None):
        if isinstance(item, str):
            return getattr(self.collection[self.id()], item)
        else:
            return self.collection[self.id()]

    def id(self):
        return self._index - 1

    def __iter__(self):
        return self


# STREAM_INPUT


class StreamInput(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def read_stream(self, stream):
        stream['new_ready'] = False

    def close(self):
        pass


class CV2WebCam(StreamInput):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cap = cv2.VideoCapture(self.cfg.input_id)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.size[1])

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


class StreamType(Enum):
    Video = 0
    Presentation = 1
    Text = 2
    Image = 3
    WebCam = 4




STREAM = defaults_mapping(
    {
        StreamType.Video: CV2WebCam,
    }
)


@dataclass
class VideoDefaults:
    input_id: int = 0
    solution = StreamType.WebCam
    type = SourceType.stream
    window_name: str = 'video_input'
    size: Union[int, int] = (1280, 720)
    origin: Union[int, int] = (0, 0)
    margins: Union[int, int] = (0, 0)
    window_position: Union[int, int] = (600, 0)
    flip: bool = True
    file: str = ''
    fps: int = 25
    rgb: bool = True
    audio_device: str = 'hw:2,0'
    output_filename: str = 'out.mp4'
    #alpha_source: AlphaSource = RVMAlpha



class MediaDataLayer_EXP(object):
    def __init__(self, cfg, global_start_time=None, source_factory=SourceFactory()):
        self.cfg = cfg
        self.SF = source_factory
        assert isinstance(cfg.transparency, list)
        self.stream_source, self.stream_config = self.init_source(cfg.stream)
        self.summary = None  # SHORT DESCRIPTION OR EMBEDDING BERT FOR EXAMPLE
        self.transparency_source, self.transparency_configs = self.init_source(cfg.transparency)  # LIST OF ALPHA SOURCES
        self.region_source, self.region_configs = self.init_source(cfg.region)
        self.effect_source, self.effect_configs = self.init_source(cfg.effect)
        self.event_source, self.event_configs = self.init_source(cfg.event)

        self._stream = {
            'rgb_buffer_cpu': np.zeros(self.stream_config.get().size[::-1] + (3,), dtype=np.uint8),
            'alpha_cpu': np.ones(self.stream_config.get().size[::-1] + (1,), dtype=np.float32),
            'rgb_buffer_cuda': None,  # USE FLOAT RGB REPRESENTATION
            'alpha_cuda': None,
            'global_time': 0.,
            'rois': {
                'person_region': np.array([-1, -1, -1, -1], dtype=np.int32),
                'face': np.array([-1, -1, -1, -1], dtype=np.int32),
            },
            'keypoints': dict(),
            'events': dict(),
            'shared_rois': dict(),
            'shared_keypoints': dict(),
            'shared_events': dict(),
            'data': dict(),
            'shared_data': dict(),
            'local_start': time.time(),
            'global_start': time.time() if global_start_time is None else global_start_time,
            'current_index': 0,
            'tick': 0,
            'new_ready': True,
        }

        self._local_start = self._stream['local_start']
        self._global_start = self._stream['global_start']

    def _do_stream(self):
        self.stream_source.read_stream(self._stream)
        self.transparency_source.process_stream(self._stream)
        self.region_source.process_stream(self._stream)
        self.event_source.process_stream(self._stream)
        self.effect_source.process_stream(self._stream)

    def render_cuda(self):
        self._do_stream()
        return self._stream['rgb_buffer_cuda'].clone(), self._stream['alpha_cuda'].clone()

    def world_act(self, regions, keypoints, events):
        pass

    def world_feedback(self, feedback):
        pass

    def start_time(self):
        return self._stream['local_start']

    def time_since_local_start(self):
        return time.time() - self._local_start

    def time_since_global_start(self):
        return time.time() - self._global_start

    def tick(self):
        return self._stream['tick']

    def init_source(self, cfg):
        if cfg:
            configs = BidirectionalIterator(cfg)
            source_cfg = configs.next()
            return self.SF.init_source(source_cfg), configs
        else:
            return self.SF.empty_source(), BidirectionalIterator([])

    def update_source(self, attribute, next=True):
        assert attribute in ('transparency', 'region', 'effect', 'event')  # ATTRIBUTE SHOULD BE ENUM OF SOURCE_TYPE
        attribute_configs = getattr(self, '_'.join([attribute, 'configs']))
        try:
            config = attribute_configs.next() if next else attribute_configs.prev()
            return self.SF.init_source(config)
        except IndexError:
            return self.SF.empty_source()


class StreamOutput(object):
    def __init__(self, cfg, layers):
        self.cfg = cfg
        self.layers = self.init_layers(layers)
        self.window_name = self.cfg.window_name
        cv2.namedWindow(self.window_name)
        self._x, self._y = self.cfg.window_position
        # OWN EVENTS SHOULD BE LIKE KEY PRESSED
        self.events = None

    def init_layers(self, layers):
        return layers

    def close(self):
        for l in self.layers:
            l.close()
        cv2.destroyAllWindows()

    def show(self, image):
        cv2.moveWindow(self.window_name, self._x, self._y)
        cv2.imshow(self.window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def stream(self):
        if self.cfg.device == DeviceType.cuda:
            with torch.no_grad():
                self.stream_cuda()
        else:
            self.stream_cpu()

    def stream_cpu(self):
        pass

    def layers_process_cuda(self):
        result = None
        for l in self.layers:
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
        return result.mul(255).byte().cpu().permute(1, 2, 0).numpy()

    def stream_cuda(self):
        #while not False: #StopToken:
        while True:
            # if self.events:
            #     for l in self.layers:
            #         l.world_act(self.events)
            # CLEAN EVENTS
            start = time.time()
            result = self.layers_process_cuda()
            print((1 / (time.time() - start)))
            self.show(result)

            if cv2.waitKey(1) & 0xFF == 27:
                break


@dataclass
class LayerConfig:
    name: str = 'default_layer'
    stream: List[SourceDefaultConfig] = None
    transparency: List[SourceDefaultConfig] = None
    region: List[SourceDefaultConfig] = None
    effect: List[SourceDefaultConfig] = None
    event: List[SourceDefaultConfig] = None

@dataclass
class StreamOutputConfig:
    name = ''
    device = DeviceType.cuda
    url = []  # SOURCE OF URLS
    input_id: int = 0
    solution = StreamType.WebCam
    window_name: str = 'video_input'
    size: Union[int, int] = (1280, 720)
    origin: Union[int, int] = (0, 0)
    margins: Union[int, int] = (0, 0)
    window_position: Union[int, int] = (600, 0)
    flip: bool = True
    file: str = ''
    fps: int = 25
    rgb: bool = True
    audio_device: str = 'hw:2,0'
    output_filename: str = 'out.mp4'


def test_layers():
    stream_cfg = VideoDefaults()
    transp_cfg = TransparencyConfig()
    region_cfg = RegionConfig()
    effect_cfg = EffectConfig()
    print(str(SourceType.transparency))
    effect_cfg.color = (0, 255, 0)
    effect_cfg.step_x = 20
    effect_cfg.step_y = 20
    print(effect_cfg)
    event_cfg = None  # WAIT FOR TO ITERATE OVER
    cfg = LayerConfig(stream=[stream_cfg],
                      transparency=[transp_cfg],
                      region=[region_cfg],
                      effect=[effect_cfg],
                      event=event_cfg)
    layer_1 = MediaDataLayer_EXP(cfg)
    out_cfg = StreamOutputConfig()
    broad = StreamOutput(out_cfg, [layer_1])
    broad.stream()
    # layer_1._do_stream()


