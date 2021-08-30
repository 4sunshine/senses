import cv2.cv2
from PIL import ImageDraw, ImageFont, Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt


class ColorMap:
    def __init__(self, cmap='bwr'):
        """TABLE SHAPE IS 256 x 3"""
        self.table, self.cmap, self.palette = self._get_table(cmap)

    @staticmethod
    def _get_table(cmap):
        image = np.arange(256, dtype=np.uint8)[:, np.newaxis]
        cm = plt.get_cmap(cmap)
        colored_image = cm(image)
        colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)
        table = colored_image[:, 0]
        table = [tuple(col) for col in table]
        return table, cmap, cm

    def set_color_map(self, cmap):
        self.table, self.cmap, self.palette = self._get_table(cmap)

    def __getitem__(self, item):
        if isinstance(item, np.ndarray):
            colored_image = self.palette(item)
            return (colored_image[:, :, :3] * 255).astype(np.uint8)
        elif isinstance(item, Image.Image):
            img = item.convert('L')
            img = np.array(img.getchannel(0))
            colored_image = self.palette(img)
            return Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8))
        else:
            item = max(0, min(255, item))
            return self.table[item]


ALIGNMENTS = ('BC', 'TC', 'C', 'LC', 'RC')


def align_text_position(t_w, t_h, im_w, im_h, align='BC', offset_y=0):
    if align == 'BC':
        x_t, y_t = (im_w - t_w) / 2, im_h - 3 * t_h / 2
    elif align == 'TC':
        x_t, y_t = (im_w - t_w) / 2, t_h / 2
    elif align == 'LC':
        x_t, y_t = 0, (im_h - t_h) / 2
    elif align == 'RC':
        x_t, y_t = im_w - t_w, (im_h - t_h) / 2
    else:
        x_t, y_t = (im_w - t_w) / 2, (im_h - t_h) / 2
    y_t += offset_y
    return x_t, y_t


def write_pil_text(image, text, position=None, size=32, color=(255, 255, 255), align=None,
                   font="visio/fonts/NotoSansJP-Medium.otf"):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font, size)
    if align is not None:
        ascent, descent = font.getmetrics()
        t_w, t_h = draw.textsize(text, font)
        (width, baseline), (offset_x, offset_y) = font.font.getsize(text)
        centered_height = 2 * ascent - offset_y
        #offset_y = 0.5 * (t_h - centered_height)
        t_h = ascent + descent
        im_w, im_h = image.size
        position = align_text_position(t_w, t_h, im_w, im_h, align)
    elif position is None:
        position = (size, 0)
    draw.text(position, text, color, font)
    return image


def write_overlay_pil_text(image, text, position=None, size=32, color=(255, 255, 255), align='BC', overlay=True,
                           overlay_color=(0, 255, 0), transparency=.5, font="visio/fonts/NotoSansJP-Medium.otf"):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font, size)
    t_w, t_h = draw.textsize(text, font)
    if align is not None:
        im_w, im_h = image.size
        position = align_text_position(t_w, t_h, im_w, im_h, align)
    elif position is None:
        position = (size, 0)
    if overlay:
        image = image.convert('RGBA')
        overlay = Image.new('RGBA', image.size, overlay_color + (0,))
        draw_over = ImageDraw.Draw(overlay)
        (width, baseline), (offset_x, offset_y) = font.font.getsize(text)
        x0, y0 = position
        y0 += offset_y
        opacity = int(255 * transparency)
        draw_over.rectangle((x0, y0, x0 + t_w, y0 + t_h - offset_y),
                            fill=overlay_color + (opacity,))
        image = Image.alpha_composite(image, overlay)
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
    draw.text(position, text, color, font)
    return image


def resize_image_to_max(image, max_size=1920):
    h, w = image.shape[:2]
    max_dim = max(h, w)
    if max_dim > max_size:
        scale = max_size / max_dim
        new_size = (int(scale * w), int(scale * h))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
    return image


def get_proportional_size(sx, sy, w, h):
    y_ratio = h / sy
    x_ratio = w / sx
    scale = min(y_ratio, x_ratio)
    return min(w, int(round(scale * sx))), min(h, int(round(scale * sy)))


def point_crop(image, point, crop_size=None, target_size=None, proportion=False):
    h, w = image.shape[:2]
    x, y = point
    if crop_size is not None:
        c_w, c_h = crop_size  # CROP WIDTH & HEIGHT
    elif proportion and target_size is not None:
        c_w, c_h = get_proportional_size(target_size[0], target_size[1], w, h)
    else:
        c_w = c_h = min(h, w)
    min_x, max_x = x - c_w // 2, x + c_w // 2
    min_y, max_y = y - c_h // 2, y + c_h // 2

    shift_x = w - max_x if max_x > w else -min_x if min_x < 0 else 0
    shift_y = h - max_y if max_y > h else -min_y if min_y < 0 else 0

    image = image[min_y + shift_y: max_y + shift_y, min_x + shift_x: max_x + shift_x, :]
    if target_size is not None:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    return image


def bbox_crop(image, bbox, target_size=None, proportion=True, max_upscale=3):
    h, w = image.shape[:2]
    x_tl, y_tl, x_br, y_br = bbox
    x, y = int((x_tl + x_br) / 2), int((y_tl + y_br) / 2)
    bbox_w, bbox_h = x_br - x_tl, y_br - y_tl

    if proportion and target_size is not None:
        max_w, max_h = int(max_upscale * bbox_w), int(max_upscale * bbox_h)
        if min(max_w, w) == max_w and min(max_h, h) == max_h:
            w_, h_ = max_w, max_h
        else:
            w_, h_ = get_proportional_size(max_w, max_h, w, h)
        c_w, c_h = get_proportional_size(target_size[0], target_size[1], w_, h_)
    else:
        c_w, c_h = min(bbox_w, w), min(bbox_h, h)
    min_x, max_x = x - c_w // 2, x + c_w // 2
    min_y, max_y = y - c_h // 2, y + c_h // 2

    shift_x = w - max_x if max_x > w else -min_x if min_x < 0 else 0
    shift_y = h - max_y if max_y > h else -min_y if min_y < 0 else 0

    image = image[min_y + shift_y: max_y + shift_y, min_x + shift_x: max_x + shift_x, :]
    if target_size is not None:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    return image


def get_ellipse_bbox(x, y, radius):
    return x - radius, y - radius, x + radius, y + radius


def get_image_tl(x, y, size):
    w, h = size
    return int(x - w / 2), int(y - h / 2)


def create_pil_mask(type='circle', target_size=(128, 128)):
    image = Image.new('L', target_size)
    draw = ImageDraw.Draw(image)
    if type == 'circle':
        draw.ellipse((0, 0, target_size[0] - 1, target_size[1] - 1), fill='white')
    return image


def horizontal_labeled_line(image, draw, width, height, text='LOVELY TEXT',
                            line_color='#abb2b9', text_color='#ffffff',
                            thickness=0, margin=0,
                            font="visio/fonts/NotoSansJP-Medium.otf", size=32):
    draw.line((0 + margin, height / 2, width - margin, height / 2), fill=line_color, width=thickness)
    # TODO: CREATE SYMMETRIC TEXTS PAIR
    text = 'ХОРОШИЙ ТЕКСТ'
    font = ImageFont.truetype(font, size)
    w, h = font.getsize(text)
    img_txt = Image.new('RGBA', font.getsize(text))
    t_draw = ImageDraw.Draw(img_txt)
    t_draw.text((0, h), text, anchor='ls', fill=text_color, font=font)
    img_txt = img_txt.rotate(90, expand=True)
    (wd, baseline), (offset_x, offset_y_ru) = font.font.getsize(text)
    image.paste(img_txt, (-offset_y_ru + margin, int(height / 2) - w), img_txt)

    text = 'LOVELY TEXT'
    w, h = font.getsize(text)
    img_txt = Image.new('RGBA', font.getsize(text))
    t_draw = ImageDraw.Draw(img_txt)
    t_draw.text((w, h), text, anchor='rs', fill=text_color, font=font)
    img_txt = img_txt.rotate(90, expand=True)
    (width, baseline), (offset_x, offset_y) = font.font.getsize(text)
    image.paste(img_txt, (-offset_y + offset_y_ru + margin, int(height / 2)), img_txt)

    img = symmetric_text_pair()
    image.paste(img, (400, 200), img)

    return image, draw


def symmetric_text_pair(
        texts=('LATIN TEXT', 'РУССКИЙ ТЕКСТ'),
        colors=('#ffffff', '#dddddd'),
        space_between=4, size=32,
        font="visio/fonts/NotoSansJP-Medium.otf"
):
    font = ImageFont.truetype(font, size)
    ascent, descent = font.getmetrics()

    w_0, h_0 = font.getsize(texts[0])
    w_1, h_1 = font.getsize(texts[1])

    target_h = max(h_0, h_1)
    target_w = w_0 + w_1 + space_between

    img = Image.new('RGBA', (target_w, target_h))
    t_draw = ImageDraw.Draw(img)
    t_draw.text((w_0, target_h - descent), texts[0], anchor='rs', fill=colors[0], font=font)
    t_draw.text((w_0 + space_between, target_h - descent), texts[1], anchor='ls', fill=colors[1], font=font)

    return img, w_0 + space_between / 2, descent


def get_symmetric_texts(
        line_type='H', flip=False,
        texts=('LATIN TEXT', 'РУССКИЙ ТЕКСТ'),
        colors=('#ffffff', '#dddddd'),
        space_between=4, size=32,
        font="visio/fonts/NotoSansJP-Medium.otf"
):
    assert line_type in ('H', 'V')
    img, x_center, descent = symmetric_text_pair(texts, colors, space_between, size, font)
    width, height = img.size
    if line_type == 'H':
        if flip:
            x_offset = height
            y_offset = x_center
            img = img.rotate(270, expand=True)
        else:
            y_offset = width - x_center
            x_offset = 0
            img = img.rotate(90, expand=True)
    else:
        x_offset = x_center
        y_offset = 0
        if flip:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            y_offset = height
    return img, x_offset, y_offset


def image_normalized_scatter_plot(values,
                                  target_size=(1000, 1000),
                                  radius=68,
                                  images=None,
                                  mask=None,
                                  xy_labels=None,
                                  background_color='#000000ff',
                                  line_color='#ffffffff',
                                  **kwargs):
    assert values.shape[1] == 2 and len(values.shape) == 2
    if images:
        assert len(values) == len(images)
    if isinstance(radius, list):
        assert len(radius) == len(values)
    if isinstance(radius, int):
        radius = [radius] * len(values)
    w, h = target_size
    values[:, 0] *= w / 2
    values[:, 0] += w / 2
    values[:, 1] *= -h / 2
    values[:, 1] += h / 2
    background = Image.new('RGBA', target_size, background_color)
    draw = ImageDraw.Draw(background)

    draw.line((0, h / 2, w, h / 2), fill=line_color)
    draw.line((w / 2, 0, w / 2, h), fill=line_color)

    if xy_labels:
        translations = kwargs.get('translations', None)
        if translations:
            xy_labels = [label for pair in xy_labels for label in pair]
            translations = [label for pair in translations for label in pair]
            flips = (True, False, False, False)
            xs = (w, 0, w / 2, w / 2)
            ys = (h / 2, h / 2, 0, h)
            line_types = ('H', 'H', 'V', 'V')
            for i in range(4):
                t, ox, oy = get_symmetric_texts(texts=[xy_labels[i].upper(), translations[i].upper()],
                                                flip=flips[i], line_type=line_types[i])
                _, h = t.size
                if i == 3:
                    oy += h
                background.paste(t, (int(xs[i] - ox), int(ys[i] - oy)), t)

    for i, (x, y) in enumerate(values):
        draw.ellipse(get_ellipse_bbox(x, y, radius[i]), fill='#55ff00', outline='#55ff00aa')

    if images:
        for i, (x, y) in enumerate(values):
            im = Image.fromarray(images[i], 'RGB')
            background.paste(im, get_image_tl(x, y, im.size), mask=mask)

    background.save('test.png')


def create_table(imgs, nrows=3, ncols=2):
    """IMAGES SHOULD APPEAR IN A ROW-WISE ORDER"""

    def stack_horizontally(images):
        return np.concatenate(images, axis=1)

    def stack_vertically(images):
        return np.concatenate(images, axis=0)

    img = stack_horizontally(imgs[: ncols])
    for i in range(1, nrows):
        img_h = stack_horizontally(imgs[i * ncols: (i+1) * ncols])
        img = stack_vertically([img, img_h])
    return img


def create_gallery(images, centers=None, target_size=(1040, 1040), n_cols=3,
                   spacing=4, spacing_color='#ffffff', n_max=None):
    if centers:
        assert len(centers) == len(images)
    if n_max is None:
        n_max = len(images)
    else:
        n_max = min(n_max, len(images))
    gallery = Image.new('RGBA', target_size, color=spacing_color)
    w, h = target_size
    s_x = (w - (n_cols + 1) * spacing) / n_cols
    s_x = int(round(s_x))
    n_rows = len(images) // n_cols + (len(images) % n_cols > 0)
    s_y = (h - (n_rows + 1) * spacing) / n_rows
    s_y = int(round(s_y))
    for i, img in enumerate(images[:n_max]):
        if centers is None:
            cy, cx = img.shape[:2]
            cx /= 2
            cy /= 2
        else:
            cx, cy = centers[i]
        img = point_crop(img, (int(cx), int(cy)), target_size=(s_x, s_y), proportion=True)
        x_tl = spacing + (s_x + spacing) * (i % n_cols)
        y_tl = spacing + (s_y + spacing) * (i // n_cols)
        img = Image.fromarray(img, 'RGB')
        gallery.paste(img, (x_tl, y_tl))

    gallery.save('test_gallery.png')


def convert_hex_to_rgb(hex_val):
    hex_val = hex_val.lstrip('#')
    return tuple(int(hex_val[i: i + 2], 16) for i in (0, 2, 4))


def change_color(img, color_to_change, new_color, binarized=True):
    """ https://stackoverflow.com/a/3753428 """
    r_c, g_c, b_c = color_to_change

    data = np.array(img)   # "data" is a height x width x 4 numpy array
    red, green, blue, alpha = data.T  # Temporarily unpack the bands for readability

    if binarized:
        red *= 0
        green *= 0
        blue *= 0

    # Replace white with red... (leaves alpha values alone...)
    replace_areas = (red == r_c) & (blue == g_c) & (green == b_c)
    data[..., :-1][replace_areas.T] = new_color  # Transpose back needed

    return Image.fromarray(data)
