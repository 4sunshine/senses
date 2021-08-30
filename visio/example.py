from visio.text import read_text, TextAnimator
from visio.utils import bbox_crop, create_pil_mask
from visio.image import CLIPPhotoSort
from visio.layout import (
    Layout, GalleryDefaults,
    TextDefaults, LogoWithTextDefaults,
    SingleImageDefaults, NormalizedScatterDefaults,
    )


def best_photos():
    from glob import glob
    import sys
    import cv2
    import numpy as np
    from detect.face import MPSimpleFaceDetector

    images_folder = sys.argv[1]
    img_paths = sorted(glob(images_folder + '/*.*g'))

    all_texts = sorted(glob('texts/*.txt'))
    ru_texts = all_texts[::2]
    en_texts = all_texts[1::2]

    lay = Layout()

    def slide_one(slide=0):
        # FIRST SLIDE
        duration = 88
        first_ts = read_text(ru_texts[slide]), read_text(en_texts[slide])
        an_l, an_r = TextAnimator(first_ts[0], duration=duration), TextAnimator(first_ts[1], duration=duration)

        lay.background.save(f'sequence_0{slide}_0000.png')

        is_finished = False
        i = 1
        CHANGE_SMILE_AT = 30

        while not is_finished:
            emoji = [u"\U0001F914"] if i > CHANGE_SMILE_AT else [u"\U0001F917"]
            cur_text_r, is_fin_r = an_r.get_text()
            cur_text_l, is_fin_l = an_l.get_text()
            img = lay.symmetric_texts(cur_text_l, cur_text_r, emojis=emoji)
            img.save(f'sequence_0{slide}_{i:04d}.png')
            i += 1
            is_finished = is_fin_l and is_fin_r

    def slide_two(slide=1):
        duration = 89
        first_ts = read_text(ru_texts[slide]), read_text(en_texts[slide])
        an_l, an_r = TextAnimator(first_ts[0], duration=duration), TextAnimator(first_ts[1], duration=duration)
        max_frames = an_r.max_frames
        is_finished = False
        i = 1

        lay.set_background(lay.schema.white)
        lay.background.save(f'sequence_0{slide}_0000.png')

        det = MPSimpleFaceDetector()
        all_images = []
        all_centers = []

        for image in img_paths:
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
            bboxes = det.get_face_bbox(image)
            xmin, ymin, xmax, ymax = bboxes[0]
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            all_images.append(image)
            all_centers.append((cx, cy))

        weights = [0.2, 0.8, 1]

        while not is_finished:
            gal_cfg = GalleryDefaults(centers=all_centers,
                                      n_max=int(round(len(all_images) * i / max_frames)))
            text_def_l = TextDefaults(font_size=None, text_color=lay.schema.right_2)
            text_def_r = TextDefaults(font_size=None, text_color=lay.schema.left_2)
            text_l, is_fin_l = an_l.get_text()
            text_r, is_fin_r = an_r.get_text()
            elements = [(text_l, text_def_l), (all_images, gal_cfg), (text_r, text_def_r)]
            img = lay.feed_and_render(elements, weights)
            img.save(f'sequence_0{slide}_{i:04d}.png')
            i += 1
            is_finished = is_fin_l and is_fin_r

    def slide_three(slide=2):
        duration = 89
        first_ts = read_text(ru_texts[slide]), read_text(en_texts[slide])
        an_l, an_r = TextAnimator(first_ts[0], duration=duration), TextAnimator(first_ts[1], duration=duration)
        is_finished = False
        i = 1

        lay.set_background(lay.schema.right_2)
        lay.background.save(f'sequence_0{slide}_0000.png')

        weights = [0.3, 0.4, 0.65, 0.7, 1]

        logo_def = LogoWithTextDefaults(text_color=lay.schema.base_color, hide=True)
        logo_path = 'visio/media/github_logo.png'
        text_logo = '/openai/CLIP'

        img_cfg = SingleImageDefaults(captions=None, hide=True)
        text_def_l = TextDefaults(font_size=None, text_color=lay.schema.white)
        text_def_r = TextDefaults(font_size=None, text_color=lay.schema.right_1)
        text_def_p = TextDefaults(font_size=None, text_color=lay.schema.black,
                                  text_back_color=lay.schema.gray)

        while not is_finished:

            text_l, is_fin_l = an_l.get_text()
            text_r, is_fin_r = an_r.get_text()
            text_p = 'PROMPT: "photo of a {CLS} person"\n' \
                     '{CLS}: (attractive, nasty), (smart, silly)' if i > 30 else ''
            elements = [(text_l, text_def_l),
                        (text_p, text_def_p),
                        ('clip_pretrain.png', img_cfg),
                        ((logo_path, text_logo), logo_def),
                        (text_r, text_def_r)]

            if i == 37:
                img_cfg.hide = False
            if i == 45:
                logo_def.hide = False

            img = lay.feed_and_render(elements, weights)
            img.save(f'sequence_0{slide}_{i:04d}.png')
            i += 1
            is_finished = is_fin_l and is_fin_r

    def slide_four(slide=3):
        duration = 88
        first_ts = read_text(ru_texts[slide]), read_text(en_texts[slide])
        an_l, an_r = TextAnimator(first_ts[0], duration=duration), TextAnimator(first_ts[1], duration=duration)
        max_frames = an_r.max_frames
        is_finished = False
        i = 1

        what_to_find = [['attractive', 'nasty'], ['smart', 'silly']]
        dual_texts = [['привлекательный', 'отвратительный'], ['умный', 'глупый']]
        pairs_of_texts = [[f'photo of a {mask} person' for mask in masks] for masks in what_to_find]
        # print(pairs_of_texts)
        clf = CLIPPhotoSort()
        values = clf.n_dim_pairs_score(img_paths, pairs_of_texts)

        values *= 0.5
        values += 0.5
        values *= 100

        captions = []
        for val_pair in values:
            p_0, p_1 = val_pair
            p_0 = int(round(p_0))
            p_1 = int(round(p_1))
            cap = f'{what_to_find[0][0].upper()}: {p_0}\n{what_to_find[1][0].upper()}: {p_1}'
            captions.append(cap)

        lay.set_background(lay.schema.white)
        lay.background.save(f'sequence_0{slide}_0000.png')

        det = MPSimpleFaceDetector()
        all_images = []
        all_centers = []

        for image in img_paths:
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
            bboxes = det.get_face_bbox(image)
            xmin, ymin, xmax, ymax = bboxes[0]
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            all_images.append(image)
            all_centers.append((cx, cy))

        weights = [0.2, 0.8, 1]

        capt_cfg = TextDefaults(font_size=None, alignment='L',
                                text_color=lay.schema.left_2,
                                background_color=lay.schema.black + 'aa')

        while not is_finished:
            gal_cfg = GalleryDefaults(centers=all_centers,
                                      n_max=int(round(len(all_images) * i / max_frames)),
                                      captions=captions,
                                      captions_cfg=capt_cfg)
            text_def_l = TextDefaults(font_size=None, text_color=lay.schema.base_color)
            text_def_r = TextDefaults(font_size=None, text_color=lay.schema.right_1)
            text_l, is_fin_l = an_l.get_text()
            text_r, is_fin_r = an_r.get_text()
            elements = [(text_l, text_def_l), (all_images, gal_cfg), (text_r, text_def_r)]
            img = lay.feed_and_render(elements, weights)
            img.save(f'sequence_0{slide}_{i:04d}.png')
            i += 1
            is_finished = is_fin_l and is_fin_r

    def slide_five(slide=4):
        duration = 88
        first_ts = read_text(ru_texts[slide]), read_text(en_texts[slide])
        an_l, an_r = TextAnimator(first_ts[0], duration=duration), TextAnimator(first_ts[1], duration=duration)
        max_frames = an_r.max_frames
        is_finished = False
        i = 1

        what_to_find = [['attractive', 'nasty'], ['smart', 'silly']]
        dual_texts = [['привлекательный', 'отвратительный'], ['умный', 'глупый']]
        pairs_of_texts = [[f'photo of a {mask} person' for mask in masks] for masks in what_to_find]
        # print(pairs_of_texts)
        # clf = CLIPPhotoSort()
        # values = clf.n_dim_pairs_score(img_paths, pairs_of_texts)
        # np.save('values.npy', values)

        values = np.load('values.npy')
        captions = []
        for val_pair in values:
            p_0, p_1 = val_pair
            p_0 = int(round(p_0))
            p_1 = int(round(p_1))
            cap = f'{what_to_find[0][0].upper()}: {p_0}\n{what_to_find[1][0].upper()}: {p_1}'
            captions.append(cap)

        lay.set_background(lay.schema.right_1)
        lay.background.save(f'sequence_0{slide}_0000.png')

        det = MPSimpleFaceDetector()
        all_images = []
        all_centers = []
        all_crops = []

        RADIUS = 68
        CROP_SIZE = (2 * (RADIUS - 4), 2 * (RADIUS - 4))

        mask = create_pil_mask(target_size=CROP_SIZE)

        for image in img_paths:
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
            bboxes = det.get_face_bbox(image)
            xmin, ymin, xmax, ymax = bboxes[0]
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            all_images.append(image)
            all_centers.append((cx, cy))
            crop = bbox_crop(image, bboxes[0], target_size=CROP_SIZE, proportion=True, max_upscale=2)
            all_crops.append(crop)

        weights = [0.15, 0.85, 1]

        while not is_finished:
            n_max = min(len(values), int(round(len(values) * 3 * i / 2 / duration)))
            sca_cfg = NormalizedScatterDefaults(radius=RADIUS, mask=mask,
                                                xy_labels=what_to_find, translations=dual_texts,
                                                ellipse_base=lay.schema.left_2,
                                                ellipse_outline=lay.schema.left_2,
                                                label_color_a=lay.schema.white,
                                                label_color_b=lay.schema.gray)

            text_def_l = TextDefaults(font_size=None, text_color=lay.schema.left_1)
            text_def_r = TextDefaults(font_size=None, text_color=lay.schema.right_2)
            text_l, is_fin_l = an_l.get_text()
            text_r, is_fin_r = an_r.get_text()
            elements = [(text_l, text_def_l),
                        ((values.copy()[: n_max], all_crops[: n_max]), sca_cfg),
                        (text_r, text_def_r)]
            img = lay.feed_and_render(elements, weights)
            img.save(f'sequence_0{slide}_{i:04d}.png')
            i += 1
            is_finished = is_fin_l and is_fin_r

    def slide_six(slide=5):
        duration = 88
        first_ts = read_text(ru_texts[slide]), read_text(en_texts[slide])
        an_l, an_r = TextAnimator(first_ts[0], duration=duration), TextAnimator(first_ts[1], duration=duration)
        max_frames = an_r.max_frames
        is_finished = False
        i = 1

        what_to_find = [['attractive', 'nasty'], ['smart', 'silly']]
        dual_texts = [['привлекательный', 'отвратительный'], ['умный', 'глупый']]
        pairs_of_texts = [[f'photo of a {mask} person' for mask in masks] for masks in what_to_find]
        # print(pairs_of_texts)
        # clf = CLIPPhotoSort()
        # values = clf.n_dim_pairs_score(img_paths, pairs_of_texts)
        #
        # values *= 0.5
        # values += 0.5
        # values *= 100
        # np.save('values.npy', values)

        values = np.load('values.npy')
        captions = []
        for val_pair in values:
            p_0, p_1 = val_pair
            p_0 = int(round(p_0))
            p_1 = int(round(p_1))
            cap = f'{what_to_find[0][0].upper()}: {p_0}\n{what_to_find[1][0].upper()}: {p_1}'
            captions.append(cap)

        lay.set_background(lay.schema.right_1)
        lay.background.save(f'sequence_0{slide}_0000.png')

        det = MPSimpleFaceDetector()
        all_images = []
        all_centers = []
        all_crops = []

        RADIUS = 68
        CROP_SIZE = (2 * (RADIUS - 4), 2 * (RADIUS - 4))

        mask = create_pil_mask(target_size=CROP_SIZE)

        for image in img_paths:
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
            bboxes = det.get_face_bbox(image)
            xmin, ymin, xmax, ymax = bboxes[0]
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            all_images.append(image)
            all_centers.append((cx, cy))
            crop = bbox_crop(image, bboxes[0], target_size=CROP_SIZE, proportion=True, max_upscale=2)
            all_crops.append(crop)

        weights = [0.15, 0.85, 1]

        eps = 1.e-6

        vec = np.array([[1, 1]], dtype=np.float32)
        vec_norm = np.linalg.norm(vec, axis=-1, keepdims=True) + eps
        # values_norm = np.linalg.norm(values, axis=-1, keepdims=True)
        importance = values @ vec.T / (vec_norm ** 2 + eps)
        importance = importance[:, 0]

        main_line = np.array([0., 0., 1., 1.], dtype=np.float32)
        LINE_FINISH = 45
        FADE_IN_OPACITY = 15

        while not is_finished:
            # n_max = min(len(values), int(round(len(values) * 3 * i / 2 / duration)))
            line_multi = min(1, i / LINE_FINISH)
            text_opacity = int(255 * max(0, min(1, ((i - LINE_FINISH) / FADE_IN_OPACITY)))**2)

            sca_cfg = NormalizedScatterDefaults(radius=RADIUS, mask=mask,
                                                xy_labels=what_to_find, translations=dual_texts,
                                                ellipse_base=lay.schema.left_2,
                                                ellipse_outline=lay.schema.left_2,
                                                label_color_a=lay.schema.white,
                                                label_color_b=lay.schema.gray,
                                                additional_line_color=lay.schema.base_color,
                                                text_opacity=text_opacity)

            text_def_l = TextDefaults(font_size=None, text_color=lay.schema.right_2)
            text_def_r = TextDefaults(font_size=None, text_color=lay.schema.left_1)
            text_l, is_fin_l = an_l.get_text()
            text_r, is_fin_r = an_r.get_text()
            lines = [line_multi * main_line]
            elements = [(text_l, text_def_l),
                        ((values.copy(), all_crops,
                          lines, importance if i > LINE_FINISH else None), sca_cfg),
                        (text_r, text_def_r)]
            img = lay.feed_and_render(elements, weights)
            img.save(f'sequence_0{slide}_{i:04d}.png')
            i += 1
            is_finished = is_fin_l and is_fin_r

    def slide_seven(slide=6):
        duration = 88
        first_ts = read_text(ru_texts[slide]), read_text(en_texts[slide])
        an_l, an_r = TextAnimator(first_ts[0], duration=duration), TextAnimator(first_ts[1], duration=duration)
        max_frames = an_r.max_frames
        is_finished = False
        i = 1

        what_to_find = [['attractive', 'nasty'], ['smart', 'silly']]
        dual_texts = [['привлекательный', 'отвратительный'], ['умный', 'глупый']]
        pairs_of_texts = [[f'photo of a {mask} person' for mask in masks] for masks in what_to_find]
        # print(pairs_of_texts)
        # clf = CLIPPhotoSort()
        # values = clf.n_dim_pairs_score(img_paths, pairs_of_texts)

        #np.save('values.npy', values)
        values = np.load('values.npy')

        captions = []
        for val_pair in values:
            p_0, p_1 = val_pair
            p_0 = int(round(p_0))
            p_1 = int(round(p_1))
            cap = f'{what_to_find[0][0].upper()}: {p_0}\n{what_to_find[1][0].upper()}: {p_1}'
            captions.append(cap)

        lay.set_background(lay.schema.right_1)
        lay.background.save(f'sequence_0{slide}_0000.png')

        det = MPSimpleFaceDetector()
        all_images = []
        all_centers = []
        all_crops = []

        RADIUS = 68
        CROP_SIZE = (2 * (RADIUS - 4), 2 * (RADIUS - 4))

        mask = create_pil_mask(target_size=CROP_SIZE)

        for image in img_paths:
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
            bboxes = det.get_face_bbox(image)
            xmin, ymin, xmax, ymax = bboxes[0]
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            all_images.append(image)
            all_centers.append((cx, cy))
            crop = bbox_crop(image, bboxes[0], target_size=CROP_SIZE, proportion=True, max_upscale=2)
            all_crops.append(crop)

        weights = [0.15, 0.85, 1]

        eps = 1.e-6

        vec = np.array([[1, 1]], dtype=np.float32)
        vec_norm = np.linalg.norm(vec, axis=-1, keepdims=True) + eps
        # values_norm = np.linalg.norm(values, axis=-1, keepdims=True)
        importance = values @ vec.T / (vec_norm ** 2 + eps)
        importance = importance[:, 0]

        main_line = np.array([0., 0., 1., 1.], dtype=np.float32)
        DISCARD_FINISH = 45

        while not is_finished:
            # n_max = min(len(values), int(round(len(values) * 3 * i / 2 / duration)))
            text_opacity = 255 #int(255 * max(0, min(1, ((i - LINE_FINISH) / FADE_IN_OPACITY)))**2)

            keep_n_discarded = max(0., min(1 - i / DISCARD_FINISH, 1.))
            rank = i > DISCARD_FINISH
            sca_cfg = NormalizedScatterDefaults(radius=RADIUS, mask=mask,
                                                xy_labels=what_to_find, translations=dual_texts,
                                                ellipse_base=lay.schema.left_2,
                                                ellipse_outline=lay.schema.left_2,
                                                label_color_a=lay.schema.white,
                                                label_color_b=lay.schema.gray,
                                                additional_line_color=lay.schema.base_color,
                                                text_opacity=text_opacity,
                                                filter='GE',
                                                keep_n_discarded=keep_n_discarded,
                                                text_color=lay.schema.white,
                                                rank=rank)

            text_def_l = TextDefaults(font_size=None, text_color=lay.schema.right_2)
            text_def_r = TextDefaults(font_size=None, text_color=lay.schema.left_1)
            text_l, is_fin_l = an_l.get_text()
            text_r, is_fin_r = an_r.get_text()
            lines = [main_line]
            elements = [(text_l, text_def_l),
                        ((values.copy(), all_crops,
                          lines, importance), sca_cfg),
                        (text_r, text_def_r)]
            img = lay.feed_and_render(elements, weights)
            img.save(f'sequence_0{slide}_{i:04d}.png')
            i += 1
            is_finished = is_fin_l and is_fin_r

    def slide_eight(slide=7):
        duration = 88
        first_ts = read_text(ru_texts[slide]), read_text(en_texts[slide])
        an_l, an_r = TextAnimator(first_ts[0], duration=duration), TextAnimator(first_ts[1], duration=duration)
        max_frames = an_r.max_frames
        is_finished = False

        what_to_find = [['attractive', 'nasty'], ['smart', 'silly']]
        dual_texts = [['привлекательный', 'отвратительный'], ['умный', 'глупый']]
        pairs_of_texts = [[f'photo of a {mask} person' for mask in masks] for masks in what_to_find]
        # print(pairs_of_texts)
        #clf = CLIPPhotoSort()
        #values = clf.n_dim_pairs_score(img_paths, pairs_of_texts)

        values = np.load('values.npy')

        eps = 1.e-6

        vec = np.array([[1, 1]], dtype=np.float32)
        vec_norm = np.linalg.norm(vec, axis=-1, keepdims=True) + eps
        # values_norm = np.linalg.norm(values, axis=-1, keepdims=True)
        importance = values @ vec.T / (vec_norm ** 2 + eps)
        importance = importance[:, 0]

        mask = np.all(values >= 0, axis=-1)
        mask = np.logical_not(mask)
        masked_imp = np.ma.array(-importance, mask=mask)
        sorted_index = masked_imp.argsort()
        ranks = [np.where(sorted_index == i)[0][0] + 1 for i in range(len(values))]
        captions = []
        for ind in sorted_index[::-1]:
            if mask[ind]:
                continue
            cap = f'SCORE: {100 * importance[ind]:.0f}\nRANK: {ranks[ind]:02d}'
            captions.append(cap)

        lay.set_background(lay.schema.white)
        lay.background.save(f'sequence_0{slide}_0000.png')

        det = MPSimpleFaceDetector()
        all_images = []
        all_centers = []

        for ind in sorted_index[::-1]:
            if mask[ind]:
                continue
            image = cv2.cvtColor(cv2.imread(img_paths[ind]), cv2.COLOR_BGR2RGB)
            bboxes = det.get_face_bbox(image)
            xmin, ymin, xmax, ymax = bboxes[0]
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            all_images.append(image)
            all_centers.append((cx, cy))

        weights = [0.2, 0.8, 1]

        capt_cfg = TextDefaults(font_size=None, alignment='L',
                                text_color=lay.schema.left_2,
                                background_color=lay.schema.black + 'aa')
        i = 1
        PHOTO_DURATION = 30
        while not is_finished:
            current_ind = min(i // PHOTO_DURATION, len(all_images))
            gal_cfg = GalleryDefaults(centers=all_centers[current_ind: current_ind + 1],
                                      n_max=None,#int(round(len(all_images) * i / max_frames)),
                                      captions=captions[current_ind: current_ind + 1],
                                      n_cols=1,
                                      captions_cfg=capt_cfg)
            text_def_l = TextDefaults(font_size=None, text_color=lay.schema.right_1)
            text_def_r = TextDefaults(font_size=None, text_color=lay.schema.base_color)
            text_l, is_fin_l = an_l.get_text()
            text_r, is_fin_r = an_r.get_text()
            elements = [(text_l, text_def_l),
                        (all_images[current_ind: current_ind + 1], gal_cfg), (text_r, text_def_r)]
            img = lay.feed_and_render(elements, weights)
            img.save(f'sequence_0{slide}_{i:04d}.png')
            i += 1
            is_finished = is_fin_l and is_fin_r

    def slide_nine(slide=8):
        # FIRST SLIDE
        duration = 88
        first_ts = read_text(ru_texts[slide]), read_text(en_texts[slide])
        an_l, an_r = TextAnimator(first_ts[0], duration=duration), TextAnimator(first_ts[1], duration=duration)
        lay.set_background(lay.schema.white)
        lay.background.save(f'sequence_0{slide}_0000.png')

        is_finished = False
        i = 1

        ADD_SMILE_AT = 50

        while not is_finished:
            emoji = [u"\U0001F9D0", u"\U0001F60E"] if i > ADD_SMILE_AT else [u"\U0001F9D0"]
            cur_text_r, is_fin_r = an_r.get_text()
            cur_text_l, is_fin_l = an_l.get_text()
            img = lay.symmetric_texts(cur_text_l, cur_text_r, emojis=emoji)
            img.save(f'sequence_0{slide}_{i:04d}.png')
            i += 1
            is_finished = is_fin_l and is_fin_r

    # slide_one()
    # slide_two()
    # slide_three()
    # slide_four()
    # slide_five()
    # slide_six()
    # slide_seven()
    # slide_eight()
    # slide_nine()

