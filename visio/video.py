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
from visio.utils import ColorMap
import pyfakewebcam

# FFMPEG BACKEND
class WebCam:
    def __init__(self, width=1280, height=720, fps=25):
        ffmpg_cmd = [
            'ffmpeg',
            # FFMPEG_BIN,
            '-f', 'video4linux2',
            '-input_format', 'mjpeg',
            '-framerate', f'{fps}',
            '-video_size', f'{width}x{height}',
            '-i', '/dev/video0',
            '-pix_fmt', 'bgr24',  # opencv requires bgr24 pixel format
            '-an', '-sn',  # disable audio processing
            '-vcodec', 'rawvideo',
            '-f', 'image2pipe',
            '-',  # output to go to stdout
        ]
        self.h = height
        self.w = width
        self.read_amount = height * width * 3
        self.process = subprocess.Popen(ffmpg_cmd, stdout=subprocess.PIPE, bufsize=10**8)

    def get_frame(self, flip=True, rgb=True):
        # transform the bytes read into a numpy array
        frame = np.frombuffer(self.process.stdout.read(self.read_amount), dtype=np.uint8)
        frame = frame.reshape((self.h, self.w, 3))  # height, width, channels
        self.process.stdout.flush()
        if flip and rgb:
            return cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        elif flip and not rgb:
            return cv2.flip(frame, 1)
        else:
            return frame

    def close(self):
        self.process.terminate()


@dataclass
class WebCamDefaults:
    input_id: int = 0
    window_name: str = 'video_input'
    width: int = 1280
    height: int = 720
    window_position: Union[int, int] = (600, 0)


class WebCamCV2:
    def __init__(self, cfg=WebCamDefaults()):
        self.cap = cv2.VideoCapture(cfg.input_id)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.window_name = cfg.window_name
        cv2.namedWindow(cfg.window_name)
        self._x, self._y = cfg.window_position

    def read(self):
        return self.cap.read()

    def show(self, image):
        cv2.moveWindow(self.window_name, self._x, self._y)
        cv2.imshow(self.window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def release(self):
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


def rvm_test():
    # gs = GreenScreenCuda()
    mp_selfie_segmentation = mp.solutions.selfie_segmentation

    cap = WebCamCV2()
    det = MPSimpleFaceDetector()
    segm = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    cmap = ColorMap('inferno')

    BG_COLOR = (0, 0, 0)

    #fake_camera = pyfakewebcam.FakeWebcam('/dev/video2', 1280, 720)
    with pyvirtualcam.Camera(width=1280, height=720, fps=30) as fake_camera:
        with torch.no_grad():
            while True:
                st = time.time()
                success, frame = cap.read()
                frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                frame.flags.writeable = False
                bboxes = det.get_face_bbox(frame)
                res_segm = segm.process(frame)
                condition = np.stack(
                    (res_segm.segmentation_mask,) * 3, axis=-1) > 0.1

                cond_gray = res_segm.segmentation_mask > 0.1
                _, xs = np.nonzero(cond_gray)
                min_x = np.min(xs)
                max_x = np.max(xs)

                cute = cmap.process_grad_grayscale(gray, min_x, max_x)

                #print(res_segm.segmentation_mask > 0.1)

                frame.flags.writeable = True
                if len(bboxes):
                    x_0, y_0, x_1, y_1 = bboxes[0]
                if success:
                    result = frame
                    # bg_image = np.zeros(frame.shape, dtype=np.uint8)
                    bg_image = np.zeros_like(gray, dtype=np.uint8)
                    #bg_image[:] = BG_COLOR
                    result = np.where(cond_gray, gray, bg_image)
                    # result = np.where(condition, result, bg_image)
                    # result = gs.mix(frame)
                    # if len(bboxes) > 0:
                    #     result = result[int(y_0): int(y_1), int(x_0): int(x_1), :]
                    #     result = np.concatenate([result, np.flip(result, axis=1)], axis=1)
                    # else:
                    #     result = result[0]
                    h, w = result.shape[:2]
                    #fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    #fake_frame[:h, :w, :] = result
                    #fake_camera.send(fake_frame)
                    #fake_camera.sleep_until_next_frame()
                    #fake_camera.schedule_frame(fake_frame)
                    cap.show(cute)
                print(round(1 / (time.time() - st)))
                if cv2.waitKey(1) & 0xFF == 27:
                    cap.release()
                    break
