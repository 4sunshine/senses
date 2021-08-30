from PIL import Image, ImageDraw, ImageFont


def get_font_size_for_target_size(texts, target_size,
                                  font='visio/fonts/NotoSansJP-Medium.otf', font_size=32, spacing=2):
    if not texts:
        return 32
    fnt = ImageFont.truetype(font, size=font_size, layout_engine=ImageFont.LAYOUT_RAQM)
    heights, widths = [], []
    for text in texts:
        width, height = fnt.getsize(text)
        widths.append(width)
        heights.append(height)
    max_width = max(widths)
    all_heights = sum(heights) + (len(heights) - 1) * spacing
    h_scale = target_size[1] / all_heights
    w_scale = target_size[0] / max_width
    target_scale = min(h_scale, w_scale)
    return int(font_size * target_scale)


def x_anchor_by_alignment(w, alignment):
    """w: TARGET IMAGE WIDTH"""
    if alignment == 'L':
        x = 0
        anchor = 'ls'
    elif alignment == 'R':
        x = w
        anchor = 'rs'
    else:
        x = w // 2
        anchor = 'ms'
    return x, anchor


def get_rectangle_xy(anchor, x, y, width, height, descent):
    if anchor == 'ls':
        x_tl = x
        y_tl = y + descent - height
        x_br = x + width
        y_br = y + descent
    elif anchor == 'rs':
        x_tl = x - width
        y_tl = y + descent - height
        x_br = x
        y_br = y + descent
    else:
        x_tl = x - width // 2
        y_tl = y + descent - height
        x_br = x + width // 2
        y_br = y + descent
    return [(x_tl, y_tl), (x_br, y_br)]


def find_equal_font_size(all_texts, target_size, font='visio/fonts/NotoSansJP-Medium.otf',
                         spacing=2):
    font_sizes = []
    for text in all_texts:
        texts = text.splitlines()
        font_size = get_font_size_for_target_size(texts, target_size, font, font_size=32, spacing=spacing)
        font_sizes.append(font_size)
    return min(font_sizes)


def draw_text_image(target_size, text, alignment='C',
                    font='visio/fonts/NotoSansJP-Medium.otf',
                    text_color='#00ff00', background_color='#00000000',
                    text_back_color='#00000000', spacing=2,
                    font_size=None):
    texts = text.splitlines()
    if font_size is None:
        font_size = get_font_size_for_target_size(texts, target_size, font, font_size=32, spacing=spacing)
    font = ImageFont.truetype(font, font_size)
    ascent, descent = font.getmetrics()
    w, h = target_size
    img = Image.new('RGBA', target_size, color=background_color)
    t_draw = ImageDraw.Draw(img)
    # t_draw.line((0, h / 2, w, h / 2), fill='#abb2b9')
    heights = []
    widths = []

    for t in texts:
        w_t, h_t = font.getsize(t)
        # (width, baseline), (offset_x, offset_y) = font.font.getsize(t)
        heights.append(h_t)
        widths.append(w_t)

    total_height = spacing * (len(heights) - 1) + sum(heights)
    x, anchor = x_anchor_by_alignment(w, alignment)
    y = (h - total_height) / 2 + (heights[0] - descent)
    heights.append(0)  # FAKE 0 TO AN END
    bboxes = []
    for i, t in enumerate(texts):
        xy = get_rectangle_xy(anchor, x, y, widths[i], heights[i], descent)
        bboxes.append((xy[0][0], xy[0][1], xy[1][0], xy[1][1]))
        if text_back_color not in ('#00000000', (0, 0, 0, 0)):
            t_draw.rectangle(xy, fill=text_back_color)
        t_draw.text((x, y), t, anchor=anchor, fill=text_color, font=font)
        y += spacing + heights[i + 1]
    info = {
        'bboxes': bboxes,
    }
    return img, info


def read_text(file_name):
    with open(file_name, encoding='utf-8') as f:
        text_r = f.read()
    return text_r


class TextAnimator:
    def __init__(self, text, max_frames=60, duration=90):
        self.text, self.size = self.set_text(text)
        self.max_frames = max_frames
        self.frame = 0
        self.duration = duration

    def set_text(self, text):
        self.frame = 0
        return text, len(text)

    def get_text(self):
        is_finished = self.frame >= self.duration
        if not is_finished:
            self.frame += 1
            n_letters = min(int(round(self.size * self.frame / self.max_frames)), self.size + 1)
            if n_letters == 0:
                text = self.text[0]
            else:
                text = self.text[:n_letters]
        else:
            text = self.text
        return text, is_finished


if __name__ == '__main__':
    text = 'ЗДАРОВА\nПАРЕНЬ!\nOK'
    with open('input_1_en.txt', encoding='utf-8') as f:
        text = f.read()
    for i in range(1, len(text) + 1):
        cur_text = text[:i]
        print(cur_text)
        img, _ = draw_text_image((512, 512), text[:i], alignment='C',
                                 text_color='#FFD829')
        img.save(f'text_test_{i:02d}.png')
