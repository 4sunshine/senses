import numpy as np
import time
import torch
import cv2

from torchvision.transforms.functional import to_tensor
from types import MappingProxyType

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

    def id(self):
        return self._index - 1

    def __iter__(self):
        return self


class SourceType(Enum):
    transparency = 0
    region = 1
    keypoint = 2
    event = 3
    effect = 4


class DeviceType(Enum):
    cpu = 10
    cuda = 11


class RegionType(Enum):
    face_box = 20
    body_box = 21
    text_box = 22
    picture_box = 23
    video_box = 24
    drawing_box = 25


class EventType(Enum):
    keypress = 30
    born = 31
    time_passed = 32
    destroyed = 33
    slide_about = 34  # CONTENT TYPE


@dataclass
class SourceDefaultConfig:
    name: str
    type: SourceType
    device: DeviceType
    url: list  # SOURCE OF URLS
    # birth: EventType = None # EVENT CLASS
    # appear: EventType = None
    # disappear: EventType = None
    # destroy: EventType = None
    # fade_in_time: float = None # time in seconds
    # fade_out_time: float = None

# ******* TRANSPARENCY BLOCK ******** #


class AlphaSource(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.init_source()

    def init_source(self):
        pass

    def process_stream(self, stream):
        if self.cfg.device == DeviceType.cuda:
            if (stream['rgb_buffer_cuda'] is not None) and (stream['rgb_buffer_cuda'].shape[0] == 4):
                stream['alpha_cuda'] = stream['rgb_buffer_cuda'][3:, ...]
        else:
            if stream['rgb_buffer_cpu'].shape[-1] == 4:
                stream['alpha_cpu'] = stream['rgb_buffer'][..., -1:]


class RVMAlpha(AlphaSource):
    def __init__(self, cfg):
        super(RVMAlpha, self).__init__(cfg)
        self.model = torch.jit.load(self.cfg.url).cuda().eval()
        # self.model.backbone_scale = 1 / 4
        self._rec = [None] * 4
        self._downsample_ratio = self.cfg.downsample_ratio

    @torch.no_grad()
    def process_stream(self, stream):  # CHECK DEVICE LATER
        if stream['rgb_buffer_cuda'] is not None:
            source = stream['rgb_buffer_cuda'].unsqueeze(0)
            fgr, pha, *self._rec = self.model(source, *self._rec, self._downsample_ratio)
        else:
            if stream['rgb_buffer'] is not None:
                source = to_tensor(stream['rgb_buffer']).unsqueeze(0).cuda()
                fgr, pha, *self._rec = self.model(source, *self._rec, self._downsample_ratio)
                stream['rgb_buffer_cuda'] = fgr[0]
            else:
                pha = [None]
        stream['alpha_cuda'] = pha[0]


class TransparencyType(Enum):
    Opaque = 0
    RVMAlpha = 1


TRANSPARENCY = defaults_mapping(
    {
        TransparencyType.Opaque: AlphaSource,
        TransparencyType.RVMAlpha: RVMAlpha,

    }
)


@dataclass
class TransparencyConfig:
    solution = TransparencyType.RVMAlpha
    type = SourceType.transparency
    name = 'transparency'
    device = DeviceType.cuda
    url = 'visio/segmentation/rvm_mobilenetv3_fp32.torchscript'
    downsample_ratio: float = 0.25


# ******* REGION BLOCK ******** #

class RegionSource(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.init_source()

    def init_source(self):
        pass

    def process_stream(self, stream):
        pass

    def close(self):
        pass
        # if stream['rgb_buffer_cpu'] is not None:
        #     h, w = stream['rgb_buffer_cpu'].shape[: 2]
        #     stream['rois']['frame_bbox'] = [0, 0, w - 1, h - 1]
        # elif stream['rgb_buffer_cuda'] is not None:
        #     h, w = stream['rgb_buffer_cuda'].shape[1: 3]
        #     stream['rois']['frame_bbox'] = [0, 0, w - 1, h - 1]


# LATER COMPOSITION


class MPFaceDetector(RegionSource):
    def __init__(self, cfg):
        super(MPFaceDetector, self).__init__(cfg)
        self.model = MPSimpleFaceDetector(cfg.model_type,
                                          cfg.det_conf,
                                          cfg.max_detections)

    def process_stream(self, stream):
        stream['rois']['face'] = self.model.get_face_bbox(stream['rgb_buffer_cpu'])[0]

    def close(self):
        self.model.close()


class PersonRegionCUDA(RegionSource):
    def __init__(self, cfg):
        super(PersonRegionCUDA, self).__init__(cfg)

    def process_stream(self, stream):
        y, x = torch.where(stream['alpha_cuda'][0] > self.cfg.threshold)
        x_min = torch.min(x)
        x_max = torch.max(x)
        y_min = torch.min(y)
        y_max = torch.max(y)
        stream['rois']['person_region'][0] = x_min
        stream['rois']['person_region'][1] = y_min
        stream['rois']['person_region'][2] = x_max
        stream['rois']['person_region'][3] = y_max


class RegionType(Enum):
    Face = 0
    Body = 1


REGION = defaults_mapping(
    {
        RegionType.Face: MPFaceDetector,
        RegionType.Body: PersonRegionCUDA,
    }
)


@dataclass
class RegionConfig:
    solution = RegionType.Body
    type = SourceType.region
    name = 'region'
    device = DeviceType.cuda
    threshold = 0.05


# ******** EFFECT SOURCE ******* #

class EffectSource(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def process_stream(self, stream):
        pass

    def close(self):
        pass


class ColorGridCUDAEffect(EffectSource):
    def __init__(self, cfg):
        super(ColorGridCUDAEffect, self).__init__(cfg)
        self.color = torch.tensor(self.cfg.color, dtype=torch.float32, device='cuda') / 255.

    def process_stream(self, stream):
        image = stream['rgb_buffer_cuda']
        if len(image.shape) > 2:
            if self.cfg.apply_x:
                image[:3, :, ::self.cfg.step_x] = self.color[:, None, None]
            if self.cfg.apply_y:
                image[:3, ::self.cfg.step_y, :] = self.color[:, None, None]


class EffectType(Enum):
    Grid = 0


EFFECT = defaults_mapping(
    {
        EffectType.Grid: ColorGridCUDAEffect,
    }
)


@dataclass
class EffectConfig:
    solution = EffectType.Grid
    type = SourceType.effect
    name = 'effect'
    device = DeviceType.cuda
    threshold = 0.05
    apply_x = True
    apply_y = True
    step_x = 4
    step_y = 4
    color = (0, 0, 0)

# ******* EVENTS BLOCK ******** #


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


class Event(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def process_stream(self, stream):
        pass

    def listen(self):
        pass


class Keypress(Event):
    def process_stream(self, stream):
        pass


class LayerType(Enum):
    video = 0
    presentation = 1
    background = 2
    image = 3


class EmptyStreamProcessor:
    def process_stream(self):
        pass


class MediaDataLayer_EXP(object):
    def __init__(self, cfg, global_start_time=None):
        self.cfg = cfg
        assert isinstance(cfg.transparency, list)
        self.stream_source = None
        self.summary = None  # SHORT DESCRIPTION OR EMBEDDING BERT FOR EXAMPLE
        self.transparency_source, self.transparency_configs = self.init_source('transparency', cfg.transparency)  # LIST OF ALPHA SOURCES
        self.region_source, self.region_configs = self.init_source('region', cfg.region)
        self.effect_source, self.effect_configs = self.init_source('effect', cfg.effect)
        self.event_source, self.event_configs = self.init_source('event', cfg.event)

        self._stream = {
            'rgb_buffer_cpu': np.zeros(cfg.size[::-1] + (3,), dtype=np.uint8),
            'alpha_cpu': np.ones(cfg.size[::-1] + (1,), dtype=np.float32),
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

    def cuda_render(self):
        self._do_stream()
        return self._stream['rgb_buffer_cuda'], self._stream['alpha_cuda'],\
               self._stream['shared_rois'], self._stream['shared_keypoints'], self._stream['shared_events']

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

    @staticmethod
    def init_alpha(cfg):
        alpha_configs = BidirectionalIterator(cfg)
        alpha_source_cfg = next(alpha_configs)
        alpha_source = TRANSPARENCY[alpha_source_cfg.solution](alpha_source_cfg)
        return alpha_source, alpha_configs

    @staticmethod
    def init_source(attribute, cfg):
        assert attribute in ('transparency', 'region', 'effect', 'event')
        if len(cfg) > 0:
            configs = BidirectionalIterator(cfg)
            source_cfg = next(configs)
            source = globals()[attribute.upper()][source_cfg.solution](source_cfg)
            return source, configs
        else:
            return EmptyStreamProcessor(), BidirectionalIterator([])

    def update_source(self, attribute, next=True):
        assert attribute in ('transparency', 'region', 'effect', 'event')
        attribute_configs = getattr(self, '_'.join([attribute, 'configs']))
        config = attribute_configs.next() if next else attribute_configs.prev()
        return globals()[attribute.upper()][config.solution](config)


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
        success, frame = self.read()
        stream['new_ready'] = success
        if self.cfg.flip:
            stream['rgb_buffer_cpu'] = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        else:
            stream['rgb_buffer_cpu'] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # if self.cfg.device == DeviceType.cuda:
        #     stream['cuda_buffer_gpu'] = to_tensor(stream['rgb_buffer_cpu']) / 255.

    def close(self):
        self.cap.release()


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
            self.stream_cuda()
        else:
            self.stream_cpu()

    def stream_cpu(self):
        pass

    def stream_cuda(self):
        #while not False: #StopToken:
        if self.events:
            for l in self.layers:
                l.world_act(self.events)
        # CLEAN EVENTS
        result = None
        for l in self.layers:
            layer, alpha, shared_rois, shared_keypoints, shared_events = l.render()
            self.events += shared_events
            if result is None or alpha is None:
                result = layer
            else:
                result = layer * alpha + result * (1 - alpha)


@dataclass
class LayerConfig:
    name: str = 'default_layer'
    stream: str = None
    transparency: List[SourceDefaultConfig] = None
    region: List[SourceDefaultConfig] = None
    effect: List[SourceDefaultConfig] = None
    event: List[SourceDefaultConfig] = None

