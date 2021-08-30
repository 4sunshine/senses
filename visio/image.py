import os
import sys
import cv2

import torch
import numpy as np
import clip
from PIL import Image
from glob import glob

from visio.utils import write_pil_text, write_overlay_pil_text, resize_image_to_max


class CLIPPhotoSort:
    # CLIP_BACKBONES = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']

    def __init__(self, clip_backbone="ViT-B/16"):
        self._clip_vis_backbone = clip_backbone
        self._model, self._preprocess = clip.load(self._clip_vis_backbone)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @torch.no_grad()
    def get_probabilities(self, images, texts):
        image = torch.cat([self._preprocess(Image.open(img)).unsqueeze(0) for img in images], dim=0).to(self._device)
        text = clip.tokenize(texts).to(self._device)
        logits_per_image, logits_per_text = self._model(image, text)
        probs_image = logits_per_image.softmax(dim=-1).cpu().numpy()
        probs_text = logits_per_text.softmax(dim=-1).cpu().numpy()
        return probs_image, probs_text

    @torch.no_grad()
    def similarity(self, images, texts):
        # GIVES_POOR_RESULTS
        image = torch.cat([self._preprocess(Image.open(img)).unsqueeze(0) for img in images], dim=0).to(self._device)
        text = clip.tokenize(texts).to(self._device)
        image_features = self._model.encode_image(image)
        text_features = self._model.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity_per_image = (100. * image_features @ text_features.t()).softmax(dim=-1)
        similarity_per_text = (100. * text_features @ image_features.t()).softmax(dim=-1)
        return similarity_per_image.cpu().numpy(), similarity_per_text.cpu().numpy()

    def rate(self, images, texts):
        probs_image, probs_text = self.similarity(images, texts)  #self.get_probabilities(images, texts)
        desc_text = np.argsort(-probs_text, axis=-1)
        sorted_scores = np.take_along_axis(probs_text, desc_text, -1)
        return desc_text, sorted_scores, probs_text, probs_image

    def n_dim_pairs_score(self, images, pairs_of_texts):
        n_dim = len(pairs_of_texts)
        for pair in pairs_of_texts:
            assert len(pair) == 2, f'{pair} is not pair of texts'
            # PAIR: ['Positive Prompt', 'Negative Prompt'], ex. ['good', 'bad']
        all_values = np.empty((len(images), n_dim), dtype=np.float32)  # N_IMAGES x N_DIM
        for i, pair in enumerate(pairs_of_texts):
            probs_image, _ = self.similarity(images, pair)
            dim_values = (probs_image[:, 0] - 0.5) / 0.5
            all_values[:, i] = dim_values
        return all_values


class ImagesListManipulator:
    def __init__(self, target_dir, **kwargs):
        self.params = kwargs
        self.def_color = (255, 255, 255)
        self.target_dir = target_dir
        os.makedirs(self.target_dir, exist_ok=True)

    def draw_aligned_text(self, image, text, align='BC'):
        width, height = image.size
        font_size = min(width, height) // 10
        return write_pil_text(image, text, position=None, size=font_size, color=self.def_color, align=align)

    def draw_overlayed_text(self, image, text, align='BC'):
        width, height = image.size
        font_size = min(width, height) // 10
        return write_overlay_pil_text(image, text, position=None, size=font_size, color=self.def_color, align=align)

    def write_values(self, image_list, values, mask):
        for i, img in enumerate(image_list):
            basename = os.path.basename(img)
            img = Image.open(img)
            img = self.draw_overlayed_text(img, f'{mask}: {values[i]:.1f}')
            img.save(os.path.join(self.target_dir, basename))


def test(images_folder):
    from detect.face import MPSimpleFaceDetector
    det = MPSimpleFaceDetector()
    images = sorted(glob(images_folder + '/*.*g'))
    print(det.face_bbox_on_image(images[0]))


def test_photo_scatter(images_folder):
    images = sorted(glob(images_folder + '/*.*g'))
    what_to_find = [['attractive', 'nasty'], ['smart', 'silly']]
    dual_texts = [['привлекательный', 'отвратительный'], ['умный', 'глупый']]
    pairs_of_texts = [[f'photo of a {mask} person' for mask in masks] for masks in what_to_find]
    print(pairs_of_texts)
    clf = CLIPPhotoSort()
    values = clf.n_dim_pairs_score(images, pairs_of_texts)
    from detect.face import MPSimpleFaceDetector
    from visio.utils import point_crop, create_pil_mask, bbox_crop
    det = MPSimpleFaceDetector()
    all_crops = []
    all_images = []
    all_centers = []
    TARGET_SIZE = (128, 128)
    mask = create_pil_mask(target_size=TARGET_SIZE)
    for image in images:
        image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        bboxes = det.get_face_bbox(image)
        xmin, ymin, xmax, ymax = bboxes[0]
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        # crop = point_crop(image, (int(cx), int(cy)), target_size=TARGET_SIZE, proportion=True)
        crop = bbox_crop(image, bboxes[0], target_size=TARGET_SIZE, proportion=True, max_upscale=2)
        all_crops.append(crop)
        all_images.append(image)
        all_centers.append((cx, cy))

    from visio.utils import image_normalized_scatter_plot, create_gallery
    create_gallery(all_images, centers=all_centers, n_cols=4)
    image_normalized_scatter_plot(values, images=all_crops, mask=mask, xy_labels=what_to_find, translations=dual_texts)


if __name__ == '__main__':
    from detect.face import MPSimpleFaceDetector
    det = MPSimpleFaceDetector()
    images_folder = sys.argv[1]
    images = sorted(glob(images_folder + '/*.*g'))
    print(det.face_bbox_on_image(images[0]))
