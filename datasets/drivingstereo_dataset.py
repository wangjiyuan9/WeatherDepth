import os
import json
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset


class DrivingStereoDataset(MonoDataset):
    RAW_HEIGHT = 800
    RAW_WIDTH = 1762

    def __init__(self, *args, **kwargs):
        super(DrivingStereoDataset, self).__init__(*args, **kwargs)
        self.forename = {"rain": "2018-08-17-09-45-58_2018-08-17-10-", "fog": "2018-10-25-07-37-26_2018-10-25-"}

    def get_color(self, weather, name, do_flip):
        path, name = self.get_image_path(weather, name)
        color = self.loader(path)

        # w, h = color.size
        # color = color.crop((0, 250, w, h))
        # if do_flip:
        #     color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color, name

    def get_image_path(self, weather, frame_name):
        folder = "left-image-full-size"
        if "fog" in weather:
            weather = "foggy"
        if "rain" in weather:
            weather = "rainy"
        image_path = os.path.join(self.data_path, weather, folder, frame_name)
        image_name = os.path.join(weather, folder, frame_name)
        if self.debug>=3:
            print(image_name)
        return image_path, image_name

    def index_to_name(self, weather, index):
        if "fog" in weather:
            return self.forename["fog"] + self.filenames[index] + ".png"
        if "rain" in weather:
            return self.forename["rain"] + self.filenames[index] + ".png"
