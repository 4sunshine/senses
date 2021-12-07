from visio.source import *
from dataclasses import dataclass

import torch
from detect.face import MPSimpleFaceDetector
from detect.hands import MPHandsDetector


__all__ = ['MPFace', 'MPFaceConfig', 'PersonRegionCUDA', 'PersonRegionCUDAConfig', 'Region', 'REGION_MAPPING']


class Region(Enum):
    Face = 0
    Body = 1
    Text = 2
    Hands = 3


class RegionSource(Source):
    def __init__(self, cfg=None):
        super(RegionSource, self).__init__(cfg)

    def process_stream(self, stream):
        pass
        # if stream['rgb_buffer_cpu'] is not None:
        #     h, w = stream['rgb_buffer_cpu'].shape[: 2]
        #     stream['rois']['frame_bbox'] = [0, 0, w - 1, h - 1]
        # elif stream['rgb_buffer_cuda'] is not None:
        #     h, w = stream['rgb_buffer_cuda'].shape[1: 3]
        #     stream['rois']['frame_bbox'] = [0, 0, w - 1, h - 1]


# LATER COMPOSITION


@dataclass
class MPFaceConfig:
    solution = Region.Face
    type = SourceType.region
    name = 'face'
    device = Device.cpu
    max_detections = 1
    det_conf = 0.5
    model_type = 1


class MPFace(RegionSource):
    def __init__(self, cfg=None):
        super(MPFace, self).__init__(cfg)
        self.model = MPSimpleFaceDetector(self.cfg.model_type,
                                          self.cfg.det_conf,
                                          self.cfg.max_detections)

    def default_config(self):
        return MPFaceConfig()

    def process_stream(self, stream):
        if not stream['new_ready']:
            return
        stream['rgb_buffer_cpu'].flags.writeable = False
        bbox = self.model.get_face_bbox(stream['rgb_buffer_cpu'])[0]
        stream['rgb_buffer_cpu'].flags.writeable = True
        stream['rois']['face'][0] = bbox[0]
        stream['rois']['face'][1] = bbox[1]
        stream['rois']['face'][2] = bbox[2]
        stream['rois']['face'][3] = bbox[3]

    def close(self):
        self.model.close()


@dataclass
class MPHandsConfig:
    solution = Region.Hands
    type = SourceType.region
    name = 'hands'
    device = Device.cpu
    max_detections = 1
    det_conf = 0.5
    model_complexity = 0
    min_tracking_conf = 0.5
    max_num_hands = 1


class MPHands(RegionSource):
    def __init__(self, cfg=None):
        super(MPHands, self).__init__(cfg)
        self.model = MPHandsDetector(
            self.cfg.model_complexity,
            self.cfg.det_conf,
            self.cfg.min_tracking_conf,
            self.cfg.max_num_hands,
        )

    def default_config(self):
        return MPHandsConfig()

    def process_stream(self, stream):
        if not stream['new_ready']:
            return
        stream['rgb_buffer_cpu'].flags.writeable = False
        self.model.detect(stream['rgb_buffer_cpu'])
        stream['rgb_buffer_cpu'].flags.writeable = True

    def close(self):
        self.model.close()


@dataclass
class PersonRegionCUDAConfig:
    solution = Region.Body
    type = SourceType.region
    name = 'person'
    device = Device.cuda
    threshold = 0.05


class PersonRegionCUDA(RegionSource):
    def default_config(self):
        return PersonRegionCUDAConfig()

    def process_stream(self, stream):
        if not stream['new_ready']:
            return
        y, x = torch.where(stream['alpha_cuda'][0] > self.cfg.threshold)
        if len(y) == 0:
            return
        stream['rois']['person_region'][0] = torch.min(x)
        stream['rois']['person_region'][1] = torch.min(y)
        stream['rois']['person_region'][2] = torch.max(x)
        stream['rois']['person_region'][3] = torch.max(y)


REGION_MAPPING = {
    Region.Face: MPFace,
    Region.Body: PersonRegionCUDA,
    Region.Hands: MPHands,
}
