import mediapipe as mp
import cv2
import numpy as np


class FaceDetector:
    def __init__(self, model_type=1, det_conf=0.5):
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=model_type, min_detection_confidence=det_conf)
        self._annotations = None
        self._img_h, self._img_w = 0, 0

    def process(self, rgb_input):
        self._annotations = self._detector.process(rgb_input)
        self._img_h, self._img_w = rgb_input.shape[:2]

    def face_bounding_box_absolute(self):
        if self._annotations:
            all_dets = [d.location_data.relative_keypoints for d in self._annotations.detections]
            return [d for d in self._annotations.detections]
        else:
            return []


def get_channel_centre(image, channel=0):
    MIN_BBOX_SIZE = 40
    image = image[..., channel]
    image = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((3, 3), np.uint8)
    # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.erode(image, kernel, iterations=2)
    M = cv2.moments(image)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    h, w = image.shape[:2]
    bbox_w = max(min(cx, w - cx), MIN_BBOX_SIZE)
    bbox_h = max(min(cy, h - cy), MIN_BBOX_SIZE)
    return cx - bbox_w / 2, cy - bbox_h / 2, cx + bbox_w / 2, cy + bbox_h / 2


class MPSimpleFaceDetector:
    def __init__(self, model_type=1, det_conf=0.5, max_detections=1):
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=model_type, min_detection_confidence=det_conf)
        self._max_detections = max_detections

    def face_bbox_on_image(self, image_path):
        return self.get_face_bbox(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))

    def get_face_bbox(self, image):
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = self._detector.process(image)
        bboxes = []
        if not results.detections:
            bbox = get_channel_centre(image)
            bboxes.append(bbox)
        else:
            for detection in results.detections[:self._max_detections]:
                bbox = detection.location_data.relative_bounding_box
                h, w = image.shape[:2]
                bbox_w = bbox.width * w
                bbox_xmin = bbox.xmin * w
                bbox_h = bbox.height * h
                bbox_ymin = bbox.ymin * h
                bbox = (bbox_xmin, bbox_ymin, bbox_xmin + bbox_w, bbox_ymin + bbox_h)
                bboxes.append(bbox)
        return bboxes

