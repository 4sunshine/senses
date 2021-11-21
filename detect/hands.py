import cv2
import mediapipe as mp


class MPHandsDetector:
    def __init__(self, model_complexity=0, det_conf=0.5, min_tracking_conf=0.5):
        self._detector = mp.solutions.hands.Hands(
            model_complexity=model_complexity,
            min_detection_confidence=det_conf,
            min_tracking_confidence=min_tracking_conf
        )

    def detect_on_image(self, image_path):
        return self.get_face_bbox(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))

    def detect(self, image):
        results = self._detector.process(image)
        #print(results.multi_hand_landmarks)

    def close(self):
        self._detector.close()


class MPPoseDetector:
    def __init__(self, model_complexity=0, det_conf=0.5, min_tracking_conf=0.5):
        self._detector = mp.solutions.pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=det_conf,
            min_tracking_confidence=min_tracking_conf,
            enable_segmentation=True,
        )

    def detect_on_image(self, image_path):
        return self.get_face_bbox(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))

    def detect(self, image):
        results = self._detector.process(image)
        #print(results.multi_hand_landmarks)

    def close(self):
        self._detector.close()
