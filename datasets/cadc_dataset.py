import os
import json
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset


class CADCDataset(MonoDataset):
    RAW_HEIGHT = 1024
    RAW_WIDTH = 1280

    def __init__(self, *args, **kwargs):
        super(CADCDataset, self).__init__(*args, **kwargs)
        self.forename = {"rain": "2018-08-17-09-45-58_2018-08-17-10-", "fog": "2018-10-25-07-37-26_2018-10-25-"}

    def get_color(self, path, do_flip):
        color = self.loader(path)
        w, h = color.size
        color = color.crop((0, 234, w, 774))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
