import sys
import cv2

from detect.face import FaceDetector


def detect_face(data_path):
    image = cv2.imread(data_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fd = FaceDetector()
    fd.process(image)
    result = fd.face_bounding_box()
    print(dir(result))
    print(len(result))
    for r in result:
        print(dir(r.location_data))
        print(r.location_data.relative_keypoints)


if __name__ == '__main__':
    data_path = sys.argv[1]
