from __future__ import absolute_import, division, print_function

from PIL import Image
import random


import torch.utils.data as data
from torchvision import transforms
import options as g
from torchvision import transforms as T
from utils import *
from . import pair_transforms
from torchvision.utils import save_image


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloader    """

    def __init__(self, opts, filenames, is_train=False, img_ext='.png'):
        super(MonoDataset, self).__init__()
        self.vis_mode = opts.vis_mode
        self.split = opts.eval_split
        self.data_path = opts.data_path
        self.filenames = filenames

        #here is the code to handle the weather folder, ignore it if you just use the original weatherKITTI dataset
        folder_name = {'rain': [], 'fog': [], 'snow': [], '3': [], '2': [], 'train': []}
        folder_name['rain'] = ['mix_rain/50mm'] if opts.train_strategy == 'mix' and opts.start_level == 1 else ['rain/50mm'] + ['mix_rain/50mm']
        if opts.gan:
            folder_name['rain'] = ['mix_rain/50mm'] if opts.train_strategy == 'mix' and opts.start_level == 1 else ['raingan/data'] + ['mix_rain/50mm']
        folder_name['fog'] = ['fog/75m'] if opts.train_strategy == 'mix' and opts.start_level == 1 else ['fog/150m'] + ['fog/75m']
        folder_name['snow'] = ['mix_snow/data'] if opts.train_strategy == 'mix' and opts.start_level == 1 else ['snowgan/data'] + ['mix_snow/data']
        folder_name['train'] = [] if opts.train_strategy == 'mix' and opts.start_level == 0 else ['rgb/data']
        if opts.weather == 'all':
            folder_name['train'] += folder_name['rain'] + folder_name['fog'] + folder_name['snow']
        elif opts.weather == 'rain':
            folder_name['train'] += folder_name['rain']
        elif opts.weather == 'fog':
            folder_name['train'] += folder_name['fog']
        elif opts.weather == 'snow':
            folder_name['train'] += folder_name['snow']

        self.folder_name = folder_name
        self.gan = opts.gan
        self.candidate_folder = []
        self.height = opts.height
        self.width = opts.width
        self.interp = Image.ANTIALIAS
        self.debug = opts.debug
        self.frame_ids = opts.novel_frame_ids + [0]
        self.is_train = is_train
        self.use_crop = not opts.no_crop if is_train else False
        self.img_ext = img_ext
        self.net_type = opts.net_type
        # new add
        self.mix_rate = opts.mix_rate
        self.train_strategy = opts.train_strategy
        self.contrast_data = [['rgb/data']] if self.train_strategy != 'cur' else []
        self.org_pjct = opts.org_pjct
        self.curr_version = opts.curr_version
        self.current_mode = None
        self.rcd = False
        self.loader = pil_loader
        self.to_tensor = pair_transforms.ToTensor() if opts.net_type == 'plane' else T.ToTensor()
        self.weather = opts.weather
        self.next_map = {0: opts.contrast_with[0], 1: opts.contrast_with[1], 2: opts.contrast_with[2]}
        self.target_scales = opts.scales
        if self.use_crop:
            self.data_aug = transforms.Compose([self.to_tensor,
                                                pair_transforms.RandomResizeCrop((self.height, self.width), factor=(0.75, 1.5)),
                                                pair_transforms.RandomGamma(min=0.8, max=1.2),
                                                pair_transforms.RandomBrightness(min=0.5, max=2.0),
                                                pair_transforms.RandomColorBrightness(min=0.8, max=1.2)])
        else:
            self.data_aug = transforms.Compose([self.to_tensor,
                                                pair_transforms.Resize((self.height, self.width)),
                                                pair_transforms.RandomGamma(min=0.8, max=1.2),
                                                pair_transforms.RandomBrightness(min=0.5, max=2.0),
                                                pair_transforms.RandomColorBrightness(min=0.8, max=1.2)])
        if opts.net_type == "plane":
            self.val_data_aug = transforms.Compose([self.to_tensor,
                                                    pair_transforms.Resize((
                                                        self.height, self.width))])
            # g.defalut_height, g.defalut_width))])
        else:
            try:
                self.brightness = (0.8, 1.2)
                self.contrast = (0.8, 1.2)
                self.saturation = (0.8, 1.2)
                self.hue = (-0.1, 0.1)
            except TypeError:
                self.brightness = 0.2
                self.contrast = 0.2
                self.saturation = 0.2
                self.hue = 0.1
            self.resize = {}
            for i in self.target_scales:
                s = 2 ** i
                self.resize[i] = T.Resize((self.height // s, self.width // s),
                    interpolation=self.interp)

    def preprocess(self, inputs, color_aug):
        """ We create color_aug objects ahead of time and apply the same enhancements to all the images in the project. This ensures that all images fed into the pose network receive the same enhancement.
        """
        #Adjusts the color image to the desired scale and expands as needed
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k or "color_weather" in k:
                n, im, i = k
                last_scale = (n, im, -1)
                target_scale = 0
                # Minimize additional inputs to reduce memory
                inputs[(n, im, target_scale)] = self.resize[target_scale](inputs[last_scale])
                last_scale = (n, im, target_scale)
                if n == "color" and im == 0:
                    for s in self.target_scales:
                        if s == -1 or s == 0:
                            continue
                        else:
                            inputs[(n, im, s)] = self.resize[s](inputs[last_scale])  # todoz 这里是对一个resize的图像反复的resize缩小
                            last_scale = (n, im, s)
        if inputs["save_mode"] and self.do_contrast:
            inputs[("contrast", "l", 0)] = self.to_tensor(inputs[("color", 0, 0)])
            inputs[("color_cst", 'l', 0)] = self.to_tensor(color_aug(inputs[("color", 0, 0)]))
        for k in list(inputs):
            f = inputs[k]
            if "color" in k or "color_weather" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)  # tensor 化
                if im == 0 and i == 0 and "color_weather" in k:
                    inputs[("color_aug", im, i)] = self.to_tensor(color_aug(f))

        if not inputs["save_mode"] and self.do_contrast:
            inputs[("contrast", "l", 0)] = self.resize[0](inputs[("contrast", "l", -1)])
            f = inputs[("contrast", "l", 0)]
            inputs[("contrast", "l", 0)] = self.to_tensor(f)
            inputs[("color_cst", "l", 0)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.
        """
        if not self.is_train:
            inputs = {'save_mode': True}
            ld_mode = self.current_mode
            if "eigen" in self.split:
                get_image = self.get_color(ld_mode, self.filenames[index].split(), self.filenames[index].split()[2], False)
            elif "cadc" in self.split:
                get_image = self.get_color(os.path.join(self.data_path, self.filenames[index]), False)
            elif "stereo" in self.split:
                filename = self.index_to_name(ld_mode, index)
                get_image, inputs["name"] = self.get_color(ld_mode, filename, False)

            if self.net_type == 'plane':
                inputs[("color", "l", -1)] = get_image
                inputs = self.val_data_aug(inputs)
                del inputs[("color", "l", -1)]
            else:
                inputs[("color", 0, 0)] = self.to_tensor(self.resize[0](get_image))
            if self.debug >= 2:
                print(ld_mode, "test", index)
            return inputs

        # region image_process()
        inputs = {'save_mode': False}
        do_flip = self.is_train and random.random() > 0.5
        do_color_aug = self.is_train and random.random() > 0.5

        if self.train_strategy == 'mix':
            length = self.folder_name['train'].__len__()
            ranges = self.mix_rate
            assert len(ranges) == length, "The length of mix_rate must be equal to the length of folder_name['train']"
            r = random.random()
            for j in range(len(ranges)):
                if r < ranges[j]:
                    aug_folder = self.folder_name['train'][j]
                    break
        elif self.train_strategy == 'cur':
            aug_folder = self.option_folder[random.randint(0, len(self.option_folder) - 1)]
        else:
            aug_folder = self.folder_name['train'][0] if self.weather == 'clear' else self.folder_name['train'][2]

        if self.do_contrast:
            contrast_folder = self.contrast_data[-1][random.randint(0, len(self.contrast_data[-1]) - 1)] if self.contrast_data[-1] != [] else []
            if self.net_type == 'plane':
                inputs['save_mode'] = (contrast_folder == 'rgb/data')
            else:
                if "l" not in self.frame_ids:
                    self.frame_ids.append("l")
                if contrast_folder == [] and "l" in self.frame_ids:
                    self.frame_ids.remove("l")
                if contrast_folder == 'rgb/data':
                    inputs['save_mode'] = True
                    if "l" in self.frame_ids:
                        self.frame_ids.remove("l")

        if self.debug >= 2:
            print(aug_folder, self.is_train, index)
            if self.do_contrast:
                print("VS", contrast_folder)
        frame = self.filenames[index].split()
        base_folder = 'rgb/data' if self.org_pjct else self.aug_folder

        if self.net_type == 'plane':
            if do_flip:
                inputs[("color", "r", -1)] = self.get_color(base_folder, frame, "l", do_flip)
                inputs[("color", "l", -1)] = self.get_color(base_folder, frame, "r", do_flip)
                inputs[("color_weather", "r", -1)] = self.get_color(aug_folder, frame, "l", do_flip)
                inputs[("color_weather", "l", -1)] = self.get_color(aug_folder, frame, "r", do_flip)
                if self.do_contrast and not inputs['save_mode']:
                    inputs[("color_contrast", "r", -1)] = self.get_color(contrast_folder, frame, "l", do_flip)
                    inputs[("color_contrast", "l", -1)] = self.get_color(contrast_folder, frame, "r", do_flip)
            else:
                inputs[("color", "l", -1)] = self.get_color(base_folder, frame, "l", do_flip)
                inputs[("color", "r", -1)] = self.get_color(base_folder, frame, "r", do_flip)
                inputs[("color_weather", "l", -1)] = self.get_color(aug_folder, frame, "l", do_flip)
                inputs[("color_weather", "r", -1)] = self.get_color(aug_folder, frame, "r", do_flip)
                if self.do_contrast and not inputs['save_mode']:
                    inputs[("color_contrast", "l", -1)] = self.get_color(contrast_folder, frame, "l", do_flip)
                    inputs[("color_contrast", "r", -1)] = self.get_color(contrast_folder, frame, "r", do_flip)

            inputs = self.data_aug(inputs)

            del inputs[("color_weather", 'l', -1)]
            del inputs[("color_weather", "r", -1)]
            del inputs[("color", "l", -1)]
            del inputs[("color", "r", -1)]
            try:
                del inputs[("color_contrast", "l", -1)]
                del inputs[("color_contrast", "r", -1)]
            except:
                pass

            K = self.K.copy()
            K[0, :] *= self.width
            K[1, :] *= self.height
            inv_K = np.linalg.pinv(K)
            inputs["K"] = torch.from_numpy(K)
            inputs["inv_K"] = torch.from_numpy(inv_K)

        else:
            side = frame[2]
            for i in self.frame_ids:
                if i == "s":
                    other_side = {"r": "l", "l": "r"}[side]  # All images are flipped independently, so there are no inconsistencies
                    inputs[("color", i, -1)] = self.get_color(base_folder, frame, other_side, do_flip)
                elif i == "l":
                    inputs[("contrast", i, -1)] = self.get_color(contrast_folder, frame, side, do_flip)
                else:
                    # Here it is -1 and 0 and 1
                    frame_copy = frame.copy()
                    frame_copy[1] = str(int(frame_copy[1]) + int(i))
                    inputs[("color", i, -1)] = self.get_color(base_folder, frame_copy, side, do_flip)
                    if i == 0:
                        inputs[("color_weather", i, -1)] = self.get_color(aug_folder, frame, side, do_flip)

            for scale in self.target_scales:
                K = self.K.copy()
                K[0, :] *= self.width // (2 ** scale)
                K[1, :] *= self.height // (2 ** scale)

                inv_K = np.linalg.pinv(K)

                inputs[("K", scale)] = torch.from_numpy(K)
                inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

            if do_color_aug:
                color_aug = T.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
            else:
                color_aug = (lambda x: x)

            self.preprocess(inputs, color_aug)
            for i in self.frame_ids:
                if ("color_aug", i, -1) in inputs:
                    del inputs[("color_aug", i, -1)]
                if ("color", i, -1) in inputs:
                    del inputs[("color", i, -1)]
                for j in (self.target_scales + [-1]):
                    if ("color_weather", i, j) in inputs:
                        del inputs[("color_weather", i, j)]
                    if ("contrast", i, j) in inputs:
                        del inputs[("contrast", i, j)]

            if "s" in self.frame_ids:
                stereo_T = np.eye(4, dtype=np.float32)
                baseline_sign = -1 if do_flip else 1
                side_sign = -1 if side == "l" else 1
                stereo_T[0, 3] = side_sign * baseline_sign * 0.1

                inputs["stereo_T"] = torch.from_numpy(stereo_T)
        # endregion

        # # Below is the code to help save the image, and observe the effect of data enhancement and the consistency of each input
        # for key, ipt in inputs.items():
        #     if 'color' in key or 'color_aug' in key or 'color_cst' in key:
        #         image = Image.fromarray(np.uint8(ipt.numpy().transpose(1, 2, 0) * 255))
        #         image.save(os.path.join('./figures/input', str(index)+key[0]+key[1] + '.png'))

        stereo_T_l = np.eye(4, dtype=np.float32)
        stereo_T_l[0, 3] = 0.1
        stereo_T_r = np.eye(4, dtype=np.float32)
        stereo_T_r[0, 3] = -0.1

        # All (RT,t) represents the change in view from t to left, except for "l" "l" indicates the change in view from left to right
        inputs[("Rt", "l")] = torch.from_numpy(stereo_T_l)
        inputs[("Rt", "r")] = torch.from_numpy(stereo_T_r)

        return inputs

    def change_data(self, class_level):
        if self.gan:
            g.class_map[1] = ['raingan/data', 'fog/150m', 'snowgan/data']
        if self.curr_version != 1:
            self.candidate_folder = g.class_map[class_level]
            print('change level from:', class_level - 1, end=' ')
            print('to:', class_level)
        else:
            self.candidate_folder += list(set(g.class_map[class_level]) - set(self.candidate_folder))
            print('Add level:', class_level)
        # The loading mode satisfies the triple constraints of curr version, cur level, and weather
        self.option_folder = list(set(self.folder_name['train']) & set(self.candidate_folder))
        if self.debug >= 1:
            print(f"now_folder:{' '.join(self.option_folder)}")

        folder_ = list(set(self.folder_name['train']) & set(g.class_map[self.next_map[class_level]]))
        if self.contrast_data == []:
            self.contrast_data = [folder_]

        if folder_ != self.contrast_data[-1]:
            self.contrast_data.append(folder_)
        if self.debug >= 1:
            print('contrast_data:', folder_)

    def specify_data(self, ld_mode):
        if (self.debug >= 1 and not self.is_train) or self.is_train:
            print('specify to:', ld_mode)
        self.current_mode = ld_mode

    def change_strategy(self, strategy):
        if (self.debug >= 1 and not self.is_train) or self.is_train:
            print('change strategy to:', strategy)
        self.train_strategy = strategy
