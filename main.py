from visio.example import best_photos
from visio.video import rvm_test
from visio.slide import test_ppt_class, test_layered_video
from visio.utils import test_cuda_cmap
from visio.factory import SourceFactory
from visio.source import SourceType
from visio.stream import Stream
from visio.effect import Effect
from visio.transparency import Transparency
from visio.region import Region
from visio.layer import MediaLayerConfig, MediaLayer
from visio.broadcast import BroadcastWindow, AVBroadcast


if __name__ == '__main__':
    stream = [(SourceType.stream, Stream.Evolution)]
    transp = [None]
    region = [None]
    effect = [[(SourceType.effect, Effect.RandomLines), (SourceType.effect, Effect.Colorize)]]
    event = [None]
    cfg = MediaLayerConfig(stream=stream, transparency=transp, region=region, effect=effect, event=event)
    layer = MediaLayer(cfg)

    stream = [(SourceType.stream, Stream.WebCam)]
    transp = [(SourceType.transparency, Transparency.RVMAlpha)]
    region = [None]
    effect = [None]
    event = [None]
    cfg_2 = MediaLayerConfig(stream=stream, transparency=transp, region=region, effect=effect, event=event)
    layer_2 = MediaLayer(cfg_2)
    layers = [layer, layer_2]

    broad = AVBroadcast(layers)  #BroadcastWindow([layer, layer_2])
    broad.broadcast()

    # import sys
    # path = sys.argv[1]
    # test_ppt_class(path)
    # test_layered_video()
    # rvm_test()
    # test_cuda_cmap()
    # test_layers()
