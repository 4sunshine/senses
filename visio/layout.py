from dataclasses import dataclass
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont, ImageOps
from typing import List, Union
import numpy as np

from visio.text import (draw_text_image,
                        find_equal_font_size,
                        get_font_size_for_target_size,
                        get_rectangle_xy,
                        x_anchor_by_alignment)

from visio.utils import (point_crop,
                         convert_hex_to_rgb,
                         change_color,
                         get_symmetric_texts,
                         get_ellipse_bbox,
                         get_image_tl,
                         )

from visio.filters import ColorMap


# TODO: REFACTOR IMPORTS AND IMPLEMENT ELEMENTS IN CORRESPONDING FILES
# TODO: UNIFY CROP TYPES IN A SINGLE OPERATING CLASS


@dataclass
class Schema:
    base_color: str = '#E85231'
    left_1: str = '#FFD829'
    left_2: str = '#86F591'
    right_1: str = '#9629FF'
    right_2: str = '#27D3F5'
    white: str = '#FFFFFF'
    black: str = '#000000'
    gray: str = '#A6ACAF'

    def get_opaque(self, color, value):
        if isinstance(value, int):
            assert 0 <= value <= 255
            value = format(value, 'x')
        else:
            value = '00'
        color = self.__getattribute__(color)
        return color + value


@dataclass
class TextDefaults:
    font_size: int = None
    alignment: str = 'C'
    font: str = 'visio/fonts/NotoSansJP-Medium.otf'
    text_color: str = '#00ff00'
    background_color: str = '#00000000'
    text_back_color: str = '#00000000'
    spacing: int = 2
    type: str = 'text_image'


@dataclass
class GalleryDefaults:
    centers: None
    n_max: int = None
    n_cols: int = 4
    spacing: int = 4
    spacing_color: str = '#ffffff'
    type: str = 'gallery'
    captions: List[str] = None
    captions_cfg: TextDefaults = TextDefaults(font_size=None)
    cap_part: int = 4


@dataclass
class SingleImageDefaults:
    type: str = 'single_image'
    captions: List[str] = None
    captions_cfg: TextDefaults = TextDefaults(font_size=None)
    scale_to_height: bool = False
    hide: bool = False


@dataclass
class LogoWithTextDefaults:
    font_size: int = None
    type: str = 'logo_with_text'
    captions: List[str] = None
    captions_cfg: TextDefaults = TextDefaults(font_size=None)
    alignment: str = 'L'
    scale_to_height: bool = True
    font: str = 'visio/fonts/NotoSansJP-Medium.otf'
    text_color: str = '#00ff00'
    background_color: str = '#00000000'
    text_back_color: str = '#00000000'
    spacing: int = 2
    hide: bool = False


@dataclass
class EmptyBoxDefaults:
    type: str = 'empty_box'
    hide: bool = False


@dataclass
class NormalizedScatterDefaults:
    type: str = 'normalized_scatter'
    radius: int = 68
    mask: Image = None
    xy_labels: List[List[str]] = None
    translations: List[List[str]] = None
    background_color: str = '#00000000'
    line_color: str = '#ffffffff'
    border: int = 2
    n_max: int = None
    hide: bool = False
    ellipse_base: str = '#55ff00'
    ellipse_outline: str = '#55ff00aa'
    label_color_a: str = '#ffffff'
    label_color_b: str = '#dddddd'
    additional_line_color: str = '#ffffffff'
    additional_line_width: int = 4
    color_map: ColorMap = ColorMap('bwr')
    font: str = 'visio/fonts/NotoSansJP-Medium.otf'
    text_opacity: int = 255
    filter: str = None  # 'GE', 'LT'
    filter_val: Union[int, float] = 0
    keep_n_discarded: float = 0.
    rank: bool = False
    text_color: str = '#ffffff'


def get_divided_bboxes(original_size, origin=(0, 0), weights=(0.5, 0.5), axis='x'):
    w, h = original_size
    x_0, y_0 = origin

    if axis == 'x':
        points = [0] + [int(round(weight * w)) for weight in weights[:-1]] + [w]
        x_tls = [x_0 + p for p in points[:-1]]
        x_brs = [x_0 + p for p in points[1:]]
        y_tls = [y_0] * len(weights)
        y_brs = [y_0 + h] * len(weights)
    else:
        points = [0] + [int(round(weight * h)) for weight in weights[:-1]] + [h]
        y_tls = [y_0 + p for p in points[:-1]]
        y_brs = [y_0 + p for p in points[1:]]
        x_tls = [x_0] * len(weights)
        x_brs = [x_0 + w] * len(weights)

    bboxes = [(x_tl, y_tl, x_br, y_br)
              for x_tl, y_tl, x_br, y_br in zip(x_tls, y_tls, x_brs, y_brs)]

    return bboxes


def update_with_margin(size, origin, margin):
    w, h = size
    w -= margin[0] + margin[2]
    h -= margin[1] + margin[3]
    x_0, y_0 = origin
    x_0 += margin[0]
    y_0 += margin[1]
    return (x_0, y_0), (w, h)


ELEMENT_TYPES = ('text_image', 'gallery', 'single_image', 'logo_with_text')


class Element:
    def __init__(self, content, *args, **kwargs):
        self.content, self.cfg = self.init_content(content, *args, **kwargs)

    @staticmethod
    def init_content(content, *args, **kwargs):
        return content, {}

    def render(self, position, size, base):
        pass

    def destroy(self):
        self.content = None
        self.cfg = {}


class TextImage(Element):
    @staticmethod
    def init_content(text, *args, **kwargs):
        cfg = args[0]
        texts = text.splitlines()
        if not texts:
            img = Image.new('RGBA', cfg.target_size, color=cfg.background_color)
            info = {
                'bboxes': [cfg.target_size],
                'text': text,
                'size': cfg.target_size,
                'origin': cfg.origin,
                'type': 'text_image',
            }
            return img, OmegaConf.create(info)

        if cfg.font_size is None:
            font_size = get_font_size_for_target_size(texts, cfg.target_size, cfg.font,
                                                      font_size=32, spacing=cfg.spacing)
        else:
            font_size = cfg.font_size
        font = ImageFont.truetype(cfg.font, font_size)
        ascent, descent = font.getmetrics()
        w, h = cfg.target_size
        img = Image.new('RGBA', cfg.target_size, color=cfg.background_color)
        t_draw = ImageDraw.Draw(img)
        # t_draw.line((0, h / 2, w, h / 2), fill='#abb2b9')
        heights = []
        widths = []

        for t in texts:
            w_t, h_t = font.getsize(t)
            # (width, baseline), (offset_x, offset_y) = font.font.getsize(t)
            heights.append(h_t)
            widths.append(w_t)

        max_w = max(widths)

        total_height = cfg.spacing * (len(heights) - 1) + sum(heights)
        x, anchor = x_anchor_by_alignment(w, cfg.alignment)
        y = (h - total_height) / 2 + (heights[0] - descent)
        heights.append(0)  # FAKE 0 TO AN END
        bboxes = []
        for i, t in enumerate(texts):
            xy = get_rectangle_xy(anchor, x, y, widths[i], heights[i], descent)
            bboxes.append((xy[0][0], xy[0][1], xy[1][0], xy[1][1]))
            if cfg.text_back_color not in ('#00000000', (0, 0, 0, 0)):
                t_draw.rectangle(xy, fill=cfg.text_back_color)
            t_draw.text((x, y), t, anchor=anchor, fill=cfg.text_color, font=font)
            y += cfg.spacing + heights[i + 1]
        info = {
            'bboxes': bboxes,
            'text': text,
            'size': cfg.target_size,
            'origin': cfg.origin,
            'type': 'text_image',
            'max_width': max_w,
        }
        return img, OmegaConf.create(info)

    def render(self, position, size, base):
        base.paste(self.content, position, self.content)


class EmptyBox(Element):
    @staticmethod
    def init_content(_, *args, **kwargs):
        cfg = args[0]
        info = {
            'bboxes': [],
            'size': cfg.target_size,
            'origin': cfg.origin,
            'type': 'empty_box',
        }
        return None, OmegaConf.create(info)


class SingleImage(TextImage):
    @staticmethod
    def init_content(image_path, *args, **kwargs):
        cfg = args[0]
        img = Image.open(image_path).convert('RGBA')
        target_w, target_h = cfg.target_size
        w, h = img.size
        scale_x, scale_y = target_w / w, target_h / h
        if cfg.scale_to_height:
            scale = scale_y
        else:
            scale = min(scale_x, scale_y)
        new_size = (int(round(scale * w)), int(round(scale * h)))
        x_offset = (target_w - new_size[0]) // 2
        y_offset = (target_h - new_size[1]) // 2

        img = img.resize(new_size, Image.LANCZOS if scale < 1.2 else None)

        info = {
            'size': cfg.target_size,
            'origin': cfg.origin,
            'type': 'single_image',
            'offset': (x_offset, y_offset),
            'hide': cfg.hide,
        }
        return img, OmegaConf.create(info)

    def render(self, position, size, base):
        if not self.cfg.hide:
            new_x = position[0] + self.cfg.offset[0]
            new_y = position[1] + self.cfg.offset[1]
            base.paste(self.content, (new_x, new_y), self.content)


class NormalizedScatter(SingleImage):
    @staticmethod
    def init_content(values_images, *args, **kwargs):
        cfg = args[0]
        values, images, lines, importance = values_images

        radius = cfg.radius

        n_max = cfg.n_max if cfg.n_max else len(values)

        assert values.shape[1] == 2 and len(values.shape) == 2
        if images:
            assert len(values) == len(images)
        if isinstance(radius, list):
            assert len(radius) == len(values)
        elif isinstance(radius, int):
            radius = [radius] * n_max

        min_side = min(cfg.target_size) - cfg.border
        target_size = (min_side, min_side)

        w, h = target_size

        background = Image.new('RGBA', target_size, cfg.background_color)
        background = ImageOps.expand(background, cfg.border, fill=cfg.line_color)
        draw = ImageDraw.Draw(background)

        if lines:
            for line in lines:
                x_0, y_0, x_1, y_1 = line
                draw.line(((x_0 + 1.) * w / 2, (-y_0 + 1.) * h / 2,
                           (x_1 + 1.) * w / 2, (-y_1 + 1.) * h / 2),
                          fill=cfg.additional_line_color,
                          width=cfg.additional_line_width)

        filter = cfg.filter
        # MASK CREATED FOR FORMAT NP.MA.MASKED_ARRAY
        if filter is not None:
            filter_val = cfg.filter_val
            if filter == 'GE':
                mask = np.all(values >= filter_val, axis=-1)
            elif filter == 'LT':
                mask = np.all(values < filter_val, axis=-1)
            else:
                print('Incorrect filter. Keeping all values')
                mask = [True] * len(values)
            mask = np.logical_not(mask)
            n_keep = sum(mask)
            n_keep *= cfg.keep_n_discarded
            n_keep = max(0, min(sum(mask), int(round(n_keep))))
            n_kept, i = 0, 0
            while (n_kept < n_keep) and (i < len(mask)):
                if mask[i]:
                    mask[i] = False
                    n_kept += 1
                i += 1
        else:
            mask = [False] * len(values)

        values[:, 0] *= w / 2
        values[:, 0] += w / 2
        values[:, 1] *= -h / 2
        values[:, 1] += h / 2

        draw.line((0, h / 2, w, h / 2), fill=cfg.line_color)
        draw.line((w / 2, 0, w / 2, h), fill=cfg.line_color)

        xy_labels = cfg.xy_labels

        if xy_labels:
            translations = cfg.translations  # kwargs.get('translations', None)
            if translations:
                xy_labels = [label for pair in xy_labels for label in pair]
                translations = [label for pair in translations for label in pair]
                flips = (True, False, False, False)
                xs = (w, 0, w / 2, w / 2)
                ys = (h / 2, h / 2, 0, h)
                line_types = ('H', 'H', 'V', 'V')
                for i in range(4):
                    t, ox, oy = get_symmetric_texts(texts=[xy_labels[i].upper(), translations[i].upper()],
                                                    flip=flips[i], line_type=line_types[i],
                                                    colors=(cfg.label_color_a, cfg.label_color_b))
                    _, t_h = t.size
                    if i == 3:
                        oy += t_h
                    background.paste(t, (int(xs[i] - ox), int(ys[i] - oy)), t)

        font = ImageFont.truetype(cfg.font, size=cfg.radius // 2)
        opacity = cfg.text_opacity

        ranks = []
        if importance is not None:
            weights = [int(round(255 * (imp + 1.) / 2)) for imp in importance]
            if cfg.rank:
                masked_imp = np.ma.array(-importance, mask=mask)
                ranks = masked_imp.argsort()
                ranks = [np.where(ranks == i)[0][0] + 1 for i in range(len(weights))]
        else:
            weights = []

        for i, (x, y) in enumerate(values[: n_max]):
            if mask[i]:
                continue

            if weights:
                # weight = int(round(255 * (importance[i] + 1.) / 2))
                fill = cfg.color_map[weights[i]]
                outline_fill = fill
            else:
                fill = cfg.ellipse_base
                outline_fill = cfg.ellipse_outline
            draw.ellipse(get_ellipse_bbox(x, y, radius[i]),
                         fill=fill, outline=outline_fill)
        # LATER OPTIMIZE IT
        pos_offset = cfg.radius // 10
        weights_pos = (cfg.radius + pos_offset, cfg.radius + pos_offset)
        ranks_pos = (cfg.radius - pos_offset, cfg.radius - pos_offset)
        if images:
            for i, (x, y) in enumerate(values[: n_max]):
                if mask[i]:
                    continue

                im = Image.fromarray(images[i], 'RGB').convert('RGBA')
                if weights:
                    draw_im = ImageDraw.Draw(im)
                    draw_im.text(weights_pos, f'{100 * importance[i]:.0f}',
                                 fill=cfg.color_map[weights[i]] + (opacity,),
                                 font=font, anchor='lt')
                    if ranks:
                        draw_im.line((cfg.radius, 3 * cfg.radius // 2,
                                      cfg.radius, cfg.radius // 2),
                                     fill=cfg.text_color, width=cfg.additional_line_width // 2)
                        draw_im.line((cfg.radius // 2, cfg.radius,
                                      3 * cfg.radius // 2, cfg.radius),
                                     fill=cfg.text_color, width=cfg.additional_line_width // 2)
                        draw_im.text(ranks_pos, f'{ranks[i]:02d}',
                                     fill=cfg.text_color,
                                     font=font, anchor='rb')

                im = im.convert('RGB')

                background.paste(im, get_image_tl(x, y, im.size), mask=cfg.mask)

        target_w, target_h = cfg.target_size

        x_offset = (target_w - w - cfg.border) // 2
        y_offset = (target_h - h - cfg.border) // 2

        info = {
            'size': cfg.target_size,
            'origin': cfg.origin,
            'type': 'normalized_scatter',
            'offset': (x_offset, y_offset),
            'hide': cfg.hide,
            'translations': cfg.translations,
        }
        return background, OmegaConf.create(info)


class LogoWithText(SingleImage):
    @staticmethod
    def init_content(logo_text, *args, **kwargs):
        logo, text = logo_text
        cfg = args[0]
        logo_img = SingleImage(logo, cfg).content
        rgb_text_color = convert_hex_to_rgb(cfg.text_color)

        logo_img = change_color(logo_img, (0, 0, 0), rgb_text_color)

        text_img_obj = TextImage(text, cfg)
        text_img = text_img_obj.content
        l_w, l_h = logo_img.size
        text_w = text_img_obj.cfg.max_width
        img = Image.new('RGBA', (l_w + text_w, l_h))
        img.paste(logo_img, (0, 0), logo_img)
        img.paste(text_img, (l_w, 0), text_img)
        target_width, target_height = cfg.target_size
        x_offset = (target_width - l_w - text_w) // 2
        y_offset = (target_height - l_h) // 2

        info = {
            'size': cfg.target_size,
            'origin': cfg.origin,
            'type': 'logo_with_text',
            'offset': (x_offset, y_offset),
            'hide': cfg.hide,
        }

        return img, OmegaConf.create(info)


class Gallery(TextImage):
    @staticmethod
    def init_content(images, *args, **kwargs):
        cfg = args[0]
        w, h = cfg.target_size

        if cfg.centers:
            assert len(cfg.centers) == len(images)
        if cfg.n_max is None:
            n_max = len(images)
        else:
            n_max = min(cfg.n_max, len(images))
        gallery = Image.new('RGBA', cfg.target_size, color=cfg.spacing_color)

        s_x = (w - (cfg.n_cols + 1) * cfg.spacing) / cfg.n_cols
        s_x = int(round(s_x))
        n_rows = len(images) // cfg.n_cols + (len(images) % cfg.n_cols > 0)
        s_y = (h - (n_rows + 1) * cfg.spacing) / n_rows
        s_y = int(round(s_y))

        if cfg.captions:
            assert len(cfg.captions) == len(images)
            cfg.captions_cfg.target_size = (s_x, s_y // cfg.cap_part)
            cfg.captions_cfg.origin = (0, s_y - s_y // cfg.cap_part)

        for i, img in enumerate(images[:n_max]):
            if cfg.centers is None:
                cy, cx = img.shape[:2]
                cx /= 2
                cy /= 2
            else:
                cx, cy = cfg.centers[i]
            img = point_crop(img, (int(cx), int(cy)), target_size=(s_x, s_y), proportion=True)
            x_tl = cfg.spacing + (s_x + cfg.spacing) * (i % cfg.n_cols)
            y_tl = cfg.spacing + (s_y + cfg.spacing) * (i // cfg.n_cols)
            img = Image.fromarray(img, 'RGB').convert('RGBA')
            if cfg.captions:
                caption = TextImage(cfg.captions[i], cfg.captions_cfg)
                caption.render(cfg.captions_cfg.origin, 0, img)
            gallery.paste(img, (x_tl, y_tl))

        info = {
            'type': 'gallery',
            'origin': cfg.origin,
            'size': cfg.target_size,
        }

        return gallery, OmegaConf.create(info)


ELEMENTS = {
    'text_image': TextImage,
    'gallery': Gallery,
    'single_image': SingleImage,
    'logo_with_text': LogoWithText,
    'normalized_scatter': NormalizedScatter,
}

DEFAULTS = {
    'text_image': TextDefaults,
    'gallery': GalleryDefaults,
    'single_image': SingleImageDefaults,
    'logo_with_text': LogoWithTextDefaults,
    'normalized_scatter': NormalizedScatterDefaults,
}


class Layout:
    """
    LATER MAKE IT OPTIMIZED AND GENERAL WITH FIELDS
    .size
    .margins
    .main_size
    .origin
    .render()
    """
    def __init__(self, size=(1080, 1920), margins=(40, 250),
                 schema=Schema()):
        self.size = size
        self.margins = margins
        self.schema = schema
        self.main_origin = margins
        self.main_size = (size[0] - 2 * margins[0], size[1] - 2 * margins[1])
        self.background = Image.new('RGBA', self.size, self.schema.base_color)

    def set_background(self, color):
        self.background = Image.new('RGBA', self.size, color)

    def symmetric_texts(self, text_a, text_b, axis='y', margin=(0, 0, 0, 0), emojis=[]):
        back = self.background.copy()
        origin, size = update_with_margin(self.main_size, self.main_origin, margin)
        if axis == 'y':
            bboxes = get_divided_bboxes(size, origin)
            sizes = [(bb[2] - bb[0], bb[3] - bb[1]) for bb in bboxes]

            font_size = find_equal_font_size([text_a, text_b], sizes[0])

            img_l, info_l = draw_text_image(sizes[0], text_a, 'R',
                                            text_color=self.schema.right_1, font_size=font_size)
            img_r, info_r = draw_text_image(sizes[1], text_b, 'L',
                                            text_color=self.schema.right_2, font_size=font_size)

            bboxes_l = info_l['bboxes']
            y_min_l, y_max_l = bboxes_l[0][1], bboxes_l[-1][3]

            bboxes_r = info_r['bboxes']
            y_min_r, y_max_r = bboxes_r[0][1], bboxes_r[-1][3]

            y_min = min(y_min_r, y_min_l)
            y_max = max(y_max_r, y_max_l)
            x = bboxes[0][2]

            y_min += self.main_origin[1] + margin[1]
            y_max += self.main_origin[1] + margin[1]

            draw = ImageDraw.Draw(back)
            draw.line((x, y_min, x, y_max), fill=self.schema.black, width=1)
            if emojis:
                fnt_e = ImageFont.truetype('visio/fonts/NotoColorEmoji_WindowsCompatible.ttf',
                                           size=109, layout_engine=ImageFont.LAYOUT_RAQM)
                draw.text((int(x), int(y_min)), emojis[0], embedded_color=True, font=fnt_e, anchor='mb')
                if len(emojis) > 1:
                    draw.text((int(x), int(y_max)), emojis[1], embedded_color=True, font=fnt_e, anchor='mt')

            back.paste(img_l, bboxes[0], img_l)
            back.paste(img_r, bboxes[1], img_r)

        return back

    def init_linear(self, elements, weights, margin=(0, 0, 0, 0), axis='y'):
        assert len(elements) == len(weights)
        #back = self.background.copy()
        origin, size = update_with_margin(self.main_size, self.main_origin, margin)
        bboxes = get_divided_bboxes(size, origin, weights, axis)
        sizes = [(bb[2] - bb[0], bb[3] - bb[1]) for bb in bboxes]
        new_elements = []
        for i, (content, cfg) in enumerate(elements):
            cfg.target_size = sizes[i]
            cfg.origin = (bboxes[i][0], bboxes[i][1])
            new_elements.append(ELEMENTS[cfg.type](content, cfg))
        return new_elements

    def render(self, elements):
        back = self.background.copy()
        for el in elements:
            el.render(el.cfg.origin, 0, back)
        return back

    def feed_and_render(self, elements, weights, margin=(0, 0, 0, 0), axis='y'):
        return self.render(self.init_linear(elements, weights, margin, axis))


if __name__ == '__main__':
    pass
