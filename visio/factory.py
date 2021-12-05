from types import MappingProxyType

from visio.source import *
from visio.effect import *
from visio.transparency import *
from visio.event import *
from visio.region import *
from visio.stream import *


class SourceFactory:
    def __init__(self, prompt=None):
        self.all_sources = self.prepare_all(prompt)

    @staticmethod
    def defaults_mapping(config):
        return MappingProxyType(config)

    def prepare_all(self, prompt=None):
        all_sources = self.defaults_mapping(
            {
                SourceType.transparency: TRANSPARENCY_MAPPING,
                SourceType.region: REGION_MAPPING,
                SourceType.effect: EFFECT_MAPPING,
                SourceType.event: EVENT_MAPPING,
                SourceType.stream: STREAM_MAPPING,
                SourceType.dummy: SOURCE_MAPPING,
            }
        )
        return all_sources


    # CHECK DEFAULTS
    def init_source(self, config):
        if config is None:
            return self.empty_source()
        elif isinstance(config, list):
            try:
                assert not isinstance(config[0], list)
                sources = [self.init_source(cfg) for cfg in config]
                return Composition(sources)
            except Exception as e:
                print(f'Incorrect config:')
                print(config)
                print('* * *')
                print(e)
                return self.empty_source()
        elif isinstance(config, tuple):
            try:
                assert len(config) == 2
                source_type, solution = config
                return self.all_sources[source_type][solution](None)
            except Exception as e:
                print(f'Incorrect config:')
                print(config)
                print('* * *')
                print(e)
                return self.empty_source()
        else:
            return self.all_sources[config.type][config.solution](config)

    def empty_source(self):
        return self.all_sources[SourceType.dummy][SourceType.dummy]()
