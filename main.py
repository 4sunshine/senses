from visio.example import best_photos
from visio.video import rvm_test
from visio.slide import test_ppt_class
from visio.utils import test_cuda_cmap
from visio.factory import SourceFactory
from visio.source import SourceType, SourceInterface as SI, WaitingEvents, Event
from visio.stream import Stream, STREAM_MAPPING
from visio.effect import Effect
from visio.transparency import Transparency
#from visio.event import Event
from visio.region import Region
from visio.layer import MediaLayerConfig, MediaLayer
from visio.broadcast import BroadcastWindow, AVBroadcast


if __name__ == '__main__':
    # stream = [(SourceType.stream, Stream.Evolution)]
    # transp = [None]
    # region = [None]
    # effect = [(SourceType.effect, Effect.Colorize)] #[[(SourceType.effect, Effect.RandomLines), (SourceType.effect, Effect.Colorize)]]
    # event = [None]#[(SourceType.event, Event.Keypress)]
    # cfg = MediaLayerConfig(stream=stream, transparency=transp, region=region, effect=effect, event=event)
    # layer = MediaLayer(cfg)
    #
    stream = [SI((SourceType.stream, Stream.WebCam), None, None)]
    transp = [SI((SourceType.transparency, Transparency.RVMAlpha), None, None)]
    region = [SI((SourceType.region, Region.Body), None, None)] # (SourceType.region, Region.Hands)
    effect = [SI([SI((SourceType.effect, Effect.Colorize), None, None), SI((SourceType.effect, Effect.Grid), None, None)], None, None)]
    event = [SI(None, None, None)]
    cfg_2 = MediaLayerConfig(stream=stream, transparency=transp, region=region, effect=effect, event=event)
    layer_2 = MediaLayer(cfg_2)
    # layers = [layer, layer_2]
    #
    # broad = BroadcastWindow([layer, layer_2])
    # broad.broadcast()

    import sys
    path = sys.argv[1]
    s = test_ppt_class(path)
    print(s.slides)
    SF = SourceFactory()

    url = ['']

    #ims_slides = SF.init_source(SourceInterface((SourceType.stream, Stream.Images), s.slides, None))
    stream = [SI((SourceType.stream, Stream.Images), s.slides, None)]
    #stream_2 = [SI((SourceType.stream, Stream.Images), url, None)]
    transp = [SI(None, None, None)]
    region = [SI(None, None, None)]
    effect = [SI([SI((SourceType.effect, Effect.Colorize), None, None), SI((SourceType.effect, Effect.Grid), None, None)], None, None)] #[[(SourceType.effect, Effect.RandomLines), (SourceType.effect, Effect.Colorize)]]
    event = [SI(None, None, None)]#[(SourceType.event, Event.Keypress)]
    cfg = MediaLayerConfig(stream=stream, transparency=transp, region=region, effect=effect, event=event)
    layer = MediaLayer(cfg)
    effect = [SI(None, None, None)]
    #cfg = MediaLayerConfig(stream=stream_2, transparency=transp, region=region, effect=effect, event=event)
    layer_3 = MediaLayer(cfg)
    layers = (layer_2, layer) #[::-1]

    broad = AVBroadcast(None, layers)  #BroadcastWindow(None, layers)
    broad.broadcast()

    #print(len(is_s))
    #stream = [(SourceType.stream, Stream.Images)]
    # test_layered_video()
    # rvm_test()
    # test_cuda_cmap()
    # test_layers()
