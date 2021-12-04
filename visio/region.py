from visio.source_head import *
from dataclasses import dataclass

import torch
from detect.face import MPSimpleFaceDetector


__all__ = ['MPFace', 'MPFaceConfig', 'PersonRegionCUDA', 'PersonRegionCUDAConfig', 'RegionType']


class RegionType(Enum):
    Face = 0
    Body = 1
    Text = 2


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
    solution = RegionType.Face
    type = SourceType.region
    name = 'face'
    device = DeviceType.cpu
    max_detections = 1
    det_conf = 0.5
    model_type = 1


class MPFace(RegionSource):
    def __init__(self, cfg=None):
        super(MPFace, self).__init__(cfg)
        self.model = MPSimpleFaceDetector(cfg.model_type,
                                          cfg.det_conf,
                                          cfg.max_detections)

    def default_config(self):
        return MPFaceConfig()

    def process_stream(self, stream):
        bbox = self.model.get_face_bbox(stream['rgb_buffer_cpu'])[0]
        stream['rois']['face'][0] = bbox[0]
        stream['rois']['face'][1] = bbox[1]
        stream['rois']['face'][2] = bbox[2]
        stream['rois']['face'][3] = bbox[3]

    def close(self):
        self.model.close()


@dataclass
class PersonRegionCUDAConfig:
    solution = RegionType.Body
    type = SourceType.region
    name = 'person'
    device = DeviceType.cuda
    threshold = 0.05


class PersonRegionCUDA(RegionSource):
    def default_config(self):
        return PersonRegionCUDAConfig()

    def process_stream(self, stream):
        y, x = torch.where(stream['alpha_cuda'][0] > self.cfg.threshold)
        stream['rois']['person_region'][0] = torch.min(x)
        stream['rois']['person_region'][1] = torch.max(x)
        stream['rois']['person_region'][2] = torch.min(y)
        stream['rois']['person_region'][3] = torch.max(y)

#
# REGION = defaults_mapping(
#     {
#         RegionType.Face: MPFaceDetector,
#         RegionType.Body: PersonRegionCUDA,
#     }
# )
#

