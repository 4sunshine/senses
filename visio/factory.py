from types import MappingProxyType

from visio.source_head import *
from visio.effect import *
from visio.transparency import *
from visio.event import *


class SourceFactory:
    def __init__(self, prompt=None):
        self.all_sources = self.prepare_all(prompt)

    @staticmethod
    def defaults_mapping(config):
        return MappingProxyType(config)

    def prepare_all(self, prompt=None):
        all_sources = self.defaults_mapping(
            {
                SourceType.transparency:
                    self.defaults_mapping(  # DEFINE DICT IN TRANSPARENCY
                        {
                            TransparencyType.Opaque: AlphaSource,
                            TransparencyType.RVMAlpha: RVMAlphaCUDA,
                        },
                    ),
                SourceType.region:
                    self.defaults_mapping(
                        {
                            RegionType.Face: MPFaceDetector,
                            RegionType.Body: PersonRegionCUDA,
                        },
                    ),
                SourceType.effect:
                    self.defaults_mapping(
                        {
                            EffectType.Grid: ColorGridCUDAEffect,
                        }
                    ),
                SourceType.stream:
                    self.defaults_mapping(
                        {
                            StreamType.WebCam: CV2WebCam,
                        }
                    ),
                SourceType.empty:
                    self.defaults_mapping(
                        {
                            EmptyType.Empty: EmptyStreamProcessor,
                        }
                    )
            }
        )
        return all_sources

    def init_source(self, config):
        return self.all_sources[config.type][config.solution](config)

    def empty_source(self):
        return self.all_sources[SourceType.empty][EmptyType.Empty]()
