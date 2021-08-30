import os

from openpifpaf.predictor import Predictor
import cv2


def openpifpaf_test(image_path):
    predictor = Predictor(checkpoint='shufflenetv2k16-wholebody')
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    min_dim = min(h, w)
    for pred, _, meta in predictor.images([image_path]):
        if len(pred):
            print([dir(ann) for ann in pred])
            bbox = [ann.bbox() for ann in pred][0]
            print(bbox)
            #print([ann.bbox_from_keypoints() for ann in pred])
            #print([ann.category for ann in pred])
            print([ann.keypoints for ann in pred])
            print([ann.json_data()['keypoints'] for ann in pred])
            nose = pred[0].json_data()['keypoints'][:2]
            cx, cy = int(nose[0]), int(nose[1])
            min_x, max_x = cx - min_dim // 2, cx + min_dim // 2
            min_y, max_y = cy - min_dim // 2, cy + min_dim // 2
            shift_x = w - max_x if max_x > w else -min_x if min_x < 0 else 0
            shift_y = h - max_y if max_y > h else -min_y if min_y < 0 else 0

            all_keyps = pred[0].json_data()['keypoints']
            for i in range(0, len(all_keyps), 3):
                point = all_keyps[i: i + 3]
                cv2.circle(image, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
            cv2.circle(image, (int(nose[0]), int(nose[1])), 2, (0, 255, 0), -1)
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 2)
            cv2.imshow('W', cv2.resize(image[min_y + shift_y: max_y + shift_y, min_x + shift_x: max_x + shift_x, :], (720, 720)))
            cv2.waitKey()

    cv2.imshow('F', image)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    image_path = sys.argv[1]
    for photo in os.listdir(image_path):
        openpifpaf_test(os.path.join(image_path, photo))
