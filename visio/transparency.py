from visio.source_head import *
from dataclasses import dataclass

import torch
from torchvision.transforms.functional import to_tensor


__all__ = ['RVMAlphaCUDA', 'RVMAlphaCUDAConfig', 'TransparencyType']


class AlphaSource(Source):
    def process_stream(self, stream):
        if self.cfg.device == DeviceType.cuda:
            if (stream['rgb_buffer_cuda'] is not None) and (stream['rgb_buffer_cuda'].shape[0] == 4):
                stream['alpha_cuda'] = stream['rgb_buffer_cuda'][3:, ...]
        else:
            if stream['rgb_buffer_cpu'].shape[-1] == 4:
                stream['alpha_cpu'] = stream['rgb_buffer'][..., -1:]


class TransparencyType(Enum):
    Opaque = 0
    RVMAlpha = 1


@dataclass
class RVMAlphaCUDAConfig:
    solution = TransparencyType.RVMAlpha
    type = SourceType.transparency
    name = 'transparency'
    device = DeviceType.cuda
    url = 'visio/segmentation/rvm_mobilenetv3_fp32.torchscript'
    downsample_ratio: float = 0.25


class RVMAlphaCUDA(AlphaSource):
    def __init__(self, cfg):
        super(RVMAlphaCUDA, self).__init__(cfg)
        self.model = torch.jit.load(self.cfg.url).cuda().eval()
        # self.model.backbone_scale = 1 / 4
        self._rec = [None] * 4
        self._downsample_ratio = self.cfg.downsample_ratio

    def default_config(self):
        return RVMAlphaCUDAConfig()

    @torch.no_grad()
    def process_stream(self, stream):  # CHECK DEVICE LATER
        # if stream['rgb_buffer_cuda'] is not None:
        #     source = stream['rgb_buffer_cuda'].unsqueeze(0)
        #     fgr, pha, *self._rec = self.model(source, *self._rec, self._downsample_ratio)
        # else:
        # if stream['rgb_buffer_cpu'] is not None:
        source = to_tensor(stream['rgb_buffer_cpu']).unsqueeze_(0).cuda()
        fgr, pha, *self._rec = self.model(source, *self._rec, self._downsample_ratio)
        stream['rgb_buffer_cuda'] = fgr[0]
        # else:
        #     pha = [None]
        stream['alpha_cuda'] = pha[0]

