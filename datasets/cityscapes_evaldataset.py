import os
import json
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset


class CityscapesEvalDataset(MonoDataset):
    """Cityscapes evaluation dataset - here we are loading the raw, original images rather than
    preprocessed triplets, and so cropping needs to be done inside get_color.
    """
    RAW_HEIGHT = 1024
    RAW_WIDTH = 2048

    def __init__(self, *args, **kwargs):
        super(CityscapesEvalDataset, self).__init__(*args, **kwargs)

    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits

        txt file is of format:
            aachen aachen_000000 4
        """
        city, frame_name = self.filenames[index].split()
        side = None

        return city, frame_name, side

    def get_color(self, city, frame_name, side, do_flip, is_sequence=False):
        if side is not None:
            raise ValueError("Cityscapes dataset doesn't know how to deal with sides yet")
        path, name = self.get_image_path(city, frame_name)
        color = self.loader(path)

        # crop down to cityscapes size
        w, h = color.size
        crop_h = h * 3 // 4
        color = color.crop((192, 256, 1856, crop_h))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color, name

    def get_image_path(self, city, frame_name):
        folder = "leftImg8bit"
        split = "test"
        image_path = os.path.join(
            self.data_path, folder, split, city, frame_name + '_leftImg8bit.png')
        image_name = os.path.join(city, frame_name + '_leftImg8bit.png')
        return image_path, image_name

