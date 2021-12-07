import numpy as np
from pptx import Presentation
from pptx.util import Pt
from omegaconf import OmegaConf

from visio.video import VideoInput, AVStreamWriter, CV2WebCam, VideoDefaults, StaticInput, MPSimpleFaceDetector

import torch
import time
import cv2


class PPTToElements(object):
    DEFAULT_PT_SIZE = 2400

    def __init__(self, filepath, external_cfg):
        prs = Presentation(filepath)
        self.width, self.height = prs.slide_width, prs.slide_height
        self.target_w, self.target_h = external_cfg.size
        self.origin_x, self.origin_y = external_cfg.origin
        all_texts = self.texts(prs)
        print(all_texts)

    @staticmethod
    def parse_alignment(alignment):
        alignment = str(alignment).lower()
        if 'center' in alignment:
            alignment = 'center'
        else:
            alignment = ''
        return alignment

    def texts(self, prs):
        def get_level_parameters():
            pass

        def get_defaults(shapes, placeholders):
            text_shapes_ids = [shape.shape_id for shape in shapes if shape.has_text_frame]
            slide_defaults = dict()
            for p in placeholders:
                if p.shape_id not in text_shapes_ids:
                    continue
                sizes = []
                aligns = []
                anchors = []
                txBody = p.element.txBody
                anchor = txBody.bodyPr.get('anchor')
                for child in txBody.iterchildren():
                    if 'lstStyle' in child.tag:
                        for style in child.iterchildren():
                            align = style.get('algn')
                            for props in style.iterchildren():
                                if 'defRPr' in props.tag:
                                    def_font_size = Pt(int(props.get('sz')) / 100)
                                    cur_align = props.get('algn')
                                    if cur_align is None:
                                        aligns.append(align)
                                    else:
                                        aligns.append(cur_align)
                                    if def_font_size is None:
                                        sizes.append(self.height // 10)
                                    else:
                                        sizes.append(def_font_size)
                                    anchors.append(anchor)  # TODO: CHECK THIS

                slide_defaults[p.shape_id] = {
                    'anchors': anchors,
                    'aligns': aligns,
                    'sizes': sizes,
                }
            return slide_defaults

        def get_master_defaults(master_slide):
            # TODO: LATER IMPLEMENT MAPPING
            # key_mapping = {'title': 2, 'body': 3}
            # print(dir(master_slide))
            # print(master_slide.shapes._spTree.xml)
            # print(dir(master_slide.shapes._spTree))
            # for s in master_slide.shapes._spTree.iterchildren():
            #     if '}sp' in s.tag:
            #         for s_child in s.iterchildren():
            #             if 'nvSpPr' in s_child.tag:
            #                 #print(dir(s_child))
            #                 print(dir(s_child))
            # raise
            #         # print(dir(s))
            #         # print(dir(s.nvSpPr))
            #         # print(s.nvSpPr.shape_id)
            #         # print(s.nvSpPr.shape_name)
            # # raise

            master_defaults = dict()
            for p in master_slide.element.iterchildren():
                if 'txStyles' in p.tag:
                    for child in p.iterchildren():  ## ITERATE OVER (TITLE STYLE, BODY, OTHER)
                        typename = child.tag.split('}')[-1].replace('Style', '')
                        aligns = []
                        sizes = []
                        anchors = []
                        for style in child.iterchildren():  ## ITERATE OVER LEVELS
                            align = style.get('algn')
                            for props in style.iterchildren():
                                print('I AM HERE')
                                print(props.tag)
                                if ('defRPr' in props.tag):# and ('lvl' in props.tag):
                                    print(props.tag)
                                    size = props.get('sz')
                                    if size is None:
                                        print('CHU')
                                        size = self.DEFAULT_PT_SIZE
                                    def_font_size = Pt(int(size) / 100)
                                    cur_align = props.get('algn')
                                    if cur_align is None:
                                        aligns.append(align)
                                    else:
                                        aligns.append(cur_align)
                                    if def_font_size is None:
                                        sizes.append(self.height // 10)
                                    else:
                                        sizes.append(def_font_size)
                                    anchors.append(None)  # TODO: CHECK THIS

                        master_defaults[typename] = {
                            'anchors': anchors,
                            'aligns': aligns,
                            'sizes': sizes,
                        }

            def mapping(id):
                if id == 2:
                    return 'title'
                elif id == 3:
                    return 'body'
                else:
                    return 'other'

            return master_defaults, mapping

        texts = []
        master_defaults, master_mapping = get_master_defaults(prs.slide_master)

        for slide in prs.slides:
            defaults = get_defaults(slide.shapes, slide.slide_layout.placeholders)
            slide_texts = []
            for shape in slide.shapes:
                if not shape.has_text_frame:
                    continue
                #print(shape.shape_id)
                # print(shape.shape_type) # TODO: CHECK LATER FOR A VIDEO/PIC SUPPORT
                # print(dir(shape))
                # print(dir(shape._sp))
                # print(shape._sp._get_xfrm_attr('type'))

                x_0, y_0 = shape.left, shape.top
                s_w, s_h = shape.width, shape.height

                # print(dir(shape))

                for paragraph in shape.text_frame.paragraphs:
                    spacing = paragraph.line_spacing
                    if spacing is None:
                        spacing = 0  # TODO: CHECK DEFAULTS
                    align = paragraph.alignment

                    # print(dir(paragraph))
                    # print(paragraph._defRPr.xml)
                    # raise

                    # print(dir(paragraph))
                    # print(paragraph.level)
                    # print(align)

                    for run in paragraph.runs:
                        text = run.text
                        font_size = run.font.size
                        # print(text)
                        # print(dir(run))
                        if font_size is None:
                            print(defaults)
                            print(master_defaults)
                            if len(defaults[shape.shape_id]['sizes']) > 0:
                                font_size = defaults[shape.shape_id]['sizes'][paragraph.level]
                            else:
                                font_size = master_defaults[master_mapping(shape.shape_id)]['sizes'][paragraph.level]
                            print(font_size)
                            print('Changed')

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

