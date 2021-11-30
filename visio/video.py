import pyvirtualcam
import torch
import cv2
import numpy as np
import subprocess
import time
import mediapipe as mp

from typing import Union
from torchvision.transforms.functional import to_tensor
from dataclasses import dataclass

from detect.face import MPSimpleFaceDetector
from detect.hands import MPHandsDetector, MPPoseDetector
from visio.filters import ColorMap

from visio.text import draw_text_image
from visio.filters import *
import pyfakewebcam


@dataclass
class AlphaSourceDefaults:
    model_path: str = 'visio/segmentation/rvm_mobilenetv3_fp32.torchscript'
    downsample_ratio: float = 0.25
    alpha_compose: bool = True


class EventTracker(object):
    def __init__(self):
        pass

    def __getitem__(self, roi, keypoints):
        return None


class MediaDataLayer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.alpha_source = self.init_alpha()
        self.roi_source = self.init_roi()
        self.effect_source = self.init_effect()
        self.event_tracker = self.init_event_tracker()
        self._buffer = np.zeros(cfg.size[::-1] + (3,), dtype=np.uint8)
        self._alpha_buffer = np.zeros(cfg.size[::-1] + (1,), dtype=np.float32)
        self._alpha_compose = None

    def init_alpha(self):
        # POOR ARCHITECTURE
        config = self.cfg.alpha_source
        if config is None:
            return None
        else:
            return self.cfg.alpha_source(self.cfg.alpha_source_cfg)

    def init_roi(self):
        return None

    def init_effect(self):
        return None

    def init_event_tracker(self):
        return None

    def alpha(self, buffer):
        if self.alpha_source is not None:
            return self.alpha_source.process_image(buffer)
        else:
            return None, None

    def roi(self, buffer, alpha):
        if self.roi_source is not None:
            self.roi_source(buffer, alpha)
        else:
            return dict(), dict()

    def effect(self, buffer, alpha, roi, keypoints):
        if self.effect_source is not None:
            return self.effect_source(buffer, alpha, roi, keypoints)
        else:
            return buffer, alpha

    def read(self):
        return False, None

    def is_refreshed(self):
        success, data = self.read()
        if success:
            self._buffer = data
            self._alpha_buffer, self._alpha_compose = self.alpha(self._buffer)
        return success

    def float_render(self, roi, keypoints):
        if self.effect is not None:
            buffer, alpha = self.effect(self._buffer, self._alpha_buffer, roi, keypoints)
            return buffer, alpha, self._alpha_compose
        else:
            return self._buffer, self._alpha_buffer, self._alpha_compose

    def salient_regions(self):
        roi, keypoints = self.roi(self._buffer, self._alpha_buffer)
        return roi, keypoints

    def update(self, events):
        pass

    def events(self, roi, keypoints):
        if self.event_tracker is not None:
            return self.event_tracker(roi, keypoints)
        else:
            return []


class AlphaSource(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.init_source()

    def init_source(self):
        pass

    def process_image(self, image):
        h, w = image.shape[:2]
        return np.ones((h, w, 1), dtype=np.float32), None

    def __getitem__(self, image):
        return self.process_image(image), None


class RVMAlpha(AlphaSource):
    def init_source(self):
        self.model = torch.jit.load(self.cfg.model_path).cuda().eval()
        # self.model.backbone_scale = 1 / 4
        self._rec = [None] * 4
        self._downsample_ratio = self.cfg.downsample_ratio

    @torch.no_grad()
    def process_image(self, image):
        source = to_tensor(image).unsqueeze_(0) #.permute(0, 3, 1, 2)
        fgr, pha, *self._rec = self.model(source.cuda(), *self._rec, self._downsample_ratio)
        # print(fgr)
        if not self.cfg.alpha_compose:
            return pha[0, 0, ...].unsqueeze_(-1).cpu().numpy(), None  # com.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()
        else:
            return pha[0, 0, ...].unsqueeze_(-1).cpu().numpy(), (pha * fgr).mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]


@dataclass
class VideoDefaults:
    input_id: int = 0
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
    alpha_source: AlphaSource = RVMAlpha
    alpha_source_cfg: AlphaSourceDefaults = AlphaSourceDefaults()


class StaticInput(MediaDataLayer):
    def __init__(self, cfg):
        super(StaticInput, self).__init__(cfg)
        path = hasattr(cfg, 'path')
        if not path:
            self.image = np.zeros(self.cfg.size[::-1] + (3,), dtype=np.uint8)

    def read(self):
        return True, self.image


class VideoInput(MediaDataLayer):
    def __init__(self, cfg=VideoDefaults()):
        super(VideoInput, self).__init__(cfg)
        self.window_name = self.cfg.window_name
        cv2.namedWindow(self.window_name)
        self._x, self._y = self.cfg.window_position

    def close(self):
        cv2.destroyAllWindows()

    def show(self, image):
        cv2.moveWindow(self.window_name, self._x, self._y)
        cv2.imshow(self.window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


# FFMPEG BACKEND
class FFMPEGWebCam(VideoInput):
    def __init__(self, cfg=VideoDefaults()):
        super().__init__(cfg)
        ffmpg_cmd = [
            'ffmpeg',
            # FFMPEG_BIN,
            '-f', 'video4linux2',
            '-input_format', 'mjpeg',
            '-framerate', f'{self.cfg.fps}',
            '-video_size', f'{self.cfg.size[0]}x{self.cfg.size[1]}',
            '-i', f'/dev/video{self.cfg.input_id}',
            '-pix_fmt', 'bgr24',  # opencv requires bgr24 pixel format # TODO: override from config RGB
            '-an', '-sn',  # disable audio processing
            '-vcodec', 'rawvideo',
            '-f', 'image2pipe',
            '-',  # output to go to stdout
        ]
        self.read_amount = self.cfg.size[0] * self.cfg.size[1] * 3
        self.process = subprocess.Popen(ffmpg_cmd, stdout=subprocess.PIPE, bufsize=10**8)

    def read(self):
        # transform the bytes read into a numpy array
        frame = np.frombuffer(self.process.stdout.read(self.read_amount), dtype=np.uint8)
        frame = frame.reshape((self.cfg.size[0], self.cfg.size[1], 3))  # height, width, channels
        self.process.stdout.flush()
        if not self.cfg.flip and self.cfg.rgb:
            return True, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif self.cfg.flip and self.cfg.rgb:
            return True, cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        elif self.cfg.flip and not self.cfg.rgb:
            return True, cv2.flip(frame, 1)
        else:
            return True, frame

    def close(self):
        cv2.destroyAllWindows()
        self.process.terminate()


class CV2WebCam(VideoInput):
    def __init__(self, cfg=VideoDefaults()):
        super().__init__(cfg)
        self.cfg = cfg
        self.cap = cv2.VideoCapture(self.cfg.input_id)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.size[1])
        self.window_name = self.cfg.window_name
        cv2.namedWindow(self.window_name)
        self._x, self._y = self.cfg.window_position

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

    def close(self):
        cv2.destroyAllWindows()
        self.cap.release()


class GreenScreenCuda:
    def __init__(self, background=None, mixed_back=None,
                 model_path='visio/segmentation/rvm_mobilenetv3_fp32.torchscript'):
        self.model = torch.jit.load(model_path).cuda().eval()

        if background is None:
            back = torch.tensor([.47, 1, .6]).view(3, 1, 1)
        else:
            back = to_tensor(background).unsqueeze(0)
        self.back = back.cuda()
        # self.model.backbone_scale = 1 / 4
        self._rec = [None] * 4
        self._downsample_ratio = 0.25

    def mix(self, src):
        source = to_tensor(src).unsqueeze_(0) #.permute(0, 3, 1, 2)
        fgr, pha, *self._rec = self.model(source.cuda(), *self._rec, self._downsample_ratio)
        com = fgr * pha + self.back * (1 - pha)
        return com.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()


class AVStreamWriter:
    def __init__(self, cfg=VideoDefaults()):
        command = ['ffmpeg',
                   '-y',
                   # INPUT VIDEO STREAM
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-s', f'{cfg.size[0]}x{cfg.size[1]}',
                   '-pix_fmt', 'rgb24',  # 'bgr24'
                   '-use_wallclock_as_timestamps', '1',
                   '-i', '-',
                   # INPUT AUDIO STREAM
                   '-f', 'alsa',
                   '-ac', '1',
                   # '-channels', '1',
                   # '-sample_rate', '44100',
                   '-i', cfg.audio_device,
                   '-acodec', 'aac',
                   # OUTPUT VIDEO OPTIONS
                   '-vcodec', 'libx264',
                   '-preset', 'ultrafast',
                   # '-tune', 'film',
                   '-qp', '0',
                   # OUTPUT AUDIO OPTIONS
                   '-acodec', 'mp2',
                   cfg.output_filename]

        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, frame):
        self.process.stdin.write(frame.tobytes())

    def close(self):
        self.process.stdin.close()
        self.process.terminate()
        self.process.wait()
        print(self.process.poll())


def rvm_test():
    # gs = GreenScreenCuda()
    mp_selfie_segmentation = mp.solutions.selfie_segmentation

    cap = CV2WebCam()
    det = MPSimpleFaceDetector()
    segm = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    cmap = ColorMap('nipy_spectral')

    WIDTH, HEIGHT = 1280, 720
    #
    # mask = np.ones((HEIGHT, WIDTH, 1), dtype=np.uint8)
    # mask[::4, ...] *= 0
    # mask[:, ::4, :] *= 0
    # mask.flags.writeable = False

    BG_COLOR = (0, 0, 0)
    i = 0

    t_w, t_h = 400, 100
    img, _ = draw_text_image((t_w, t_h), 'COLOR SCHEMA')
    img = np.array(img)
    img_text = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_text.flags.writeable = False

    random_lines = RandomVerticalLines()
    gradient_color = GradientColorizeFilter()
    grid_lines = ColorGridFilter()
    writer = AVStreamWriter()

    hd = MPHandsDetector()
    pd = MPPoseDetector()

    #fake_camera = pyfakewebcam.FakeWebcam('/dev/video2', 1280, 720)
    # with pyvirtualcam.Camera(width=1280, height=720, fps=30) as fake_camera:
    with torch.no_grad():
        while True:
            st = time.time()
            success, frame = cap.read()
            #frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame.flags.writeable = False
            #bboxes = det.get_face_bbox(frame)
            if i % 2 == 0:
                hd.detect(frame)
            i += 1
            #pd.detect(frame)
            res_segm = segm.process(frame)
            condition = np.stack(
                (res_segm.segmentation_mask,) * 3, axis=-1) > 0.1

            cond_gray = res_segm.segmentation_mask > 0.1
            _, xs = np.nonzero(cond_gray)
            if len(xs):
                min_x, max_x = np.min(xs), np.max(xs)
            else:
                min_x, max_x = 0, WIDTH - 1

            gray = random_lines.transform(gray)

            gray[HEIGHT//2:HEIGHT//2 + t_h, WIDTH//2:WIDTH//2 + t_w] = img_text

            cute = gradient_color.transform(gray, endpoints=(min_x, max_x))

            cute = grid_lines.transform(cute)

            frame.flags.writeable = True
            # if len(bboxes):
            #     x_0, y_0, x_1, y_1 = bboxes[0]
            if success:
                # result = frame
                # # bg_image = np.zeros(frame.shape, dtype=np.uint8)
                # bg_image = np.zeros_like(gray, dtype=np.uint8)
                # #bg_image[:] = BG_COLOR
                # result = np.where(cond_gray, gray, bg_image)
                # # result = np.where(condition, result, bg_image)
                # # result = gs.mix(frame)
                # # if len(bboxes) > 0:
                # #     result = result[int(y_0): int(y_1), int(x_0): int(x_1), :]
                # #     result = np.concatenate([result, np.flip(result, axis=1)], axis=1)
                # # else:
                # #     result = result[0]
                # h, w = result.shape[:2]
                # #fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                # #fake_frame[:h, :w, :] = result
                # #fake_camera.send(fake_frame)
                # #fake_camera.sleep_until_next_frame()
                # #fake_camera.schedule_frame(fake_frame)
                writer.write(cute)
                cap.show(cute)
            print(round(1 / (time.time() - st)))
            if cv2.waitKey(1) & 0xFF == 27:
                cap.close()
                writer.close()
                break
