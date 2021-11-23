from pptx import Presentation
from omegaconf import OmegaConf


class PPTToElements(object):
    def __init__(self, filepath, external_cfg):
        prs = Presentation(filepath)
        self.width, self.height = prs.slide_width, prs.slide_height
        self.target_w, self.target_h = external_cfg.size
        self.origin_x, self.origin_y = external_cfg.origin
        all_texts = self.texts(prs)
        # print(all_texts)

    @staticmethod
    def parse_alignment(alignment):
        alignment = str(alignment).lower()
        if 'center' in alignment:
            alignment = 'center'
        else:
            alignment = ''
        return alignment

    def texts(self, prs):
        texts = []
        for slide in prs.slides:
            slide_texts = []
            for shape in slide.shapes:
                # print(shape.shape_type) TODO: CHECK LATER FOR A VIDEO/PIC SUPPORT
                x_0, y_0 = shape.left, shape.top
                s_w, s_h = shape.width, shape.height
                # n_paragraphs = len(shape.text_frame.paragraphs)
                if not shape.has_text_frame:
                    continue
                for paragraph in shape.text_frame.paragraphs:
                    spacing = paragraph.line_spacing
                    if spacing is None:
                        spacing = 0  # TODO: CHECK DEFAULTS
                    align = paragraph.alignment
                    for run in paragraph.runs:
                        text = run.text
                        font_size = run.font.size
                        text_w, text_h = s_w, font_size
                        text_ox, text_oy = x_0, y_0
                        y_0 += font_size + int(round(font_size * spacing))
                        # TRY: color = run.font.color EXCEPT
                        text_data = {
                            'data': text,
                            'size': (int(round(self.target_w * text_w / self.width)),
                                     int(round(self.target_h * text_h / self.height))),
                            'origin': (self.origin_x + int(round(self.target_w * text_ox / self.width)),
                                       self.origin_y + int(round(self.target_h * text_oy / self.height))),
                            'alignment': self.parse_alignment(align),  # TODO: LATER ALIGNMENT SHOULD BE INT CONST
                        }
                        slide_texts.append(OmegaConf.create(text_data))
            texts.append(slide_texts)
        return texts


def test_ppt_class(path):
    from visio.video import VideoDefaults
    cfg = VideoDefaults()
    s = PPTToElements(path, cfg)
