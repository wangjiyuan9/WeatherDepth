# CopyRight WJY2023
# This file is used to store the basic global variables and options for the project.
from __future__ import absolute_import, division, print_function

import os
import argparse

from layers import HomographyWarp

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in
MIN_DEPTH = 1e-3
MAX_DEPTH = 80
defalut_height = 384
defalut_width = 1280

index_map = {0: 'abs_rel', 1: 'sq_rel', 2: 'rmse', 3: 'rmse_log', 4: 'a1', 5: 'a2', 6: 'a3', }
class_map = {0: ['rgb/data'], 1: ['rain/50mm', 'fog/150m', 'snowgan/data'], 2: ['mix_rain/50mm', 'fog/75m', 'mix_snow/data'], None: []}

defalut_mixloss = True
defalut_denseaspp = True
defalut_planeres = True
defalut_flip = True


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Planedepth options")
        # my options
        self.parser.add_argument("--data_path",
            type=str,
            help="path to the training data")
        self.parser.add_argument("--log_dir",
            type=str,
            help="log directory")

        self.parser.add_argument("--use_multi_gpu", "--umg",
            action="store_true",
            help="if set, use multi gpu")
        self.parser.add_argument("--train_strategy",
            type=str,
            choices=["mix", "cur", "clear"],
            help="training strategy, mix: mixup, cur: curriculum",
            default="mix")
        self.parser.add_argument("--curr_version", "--cur_vis",
            type=int,
            help="curriculum visualization, 0:org_cur,1:mix_cur,2:self_step,3:ss+contrast,4:used in vit,5:used in planedepth,0-3 used in ablation study",
            default=None)
        self.parser.add_argument("--org_pjct",
            help="if set 1,use the clear image for photometric re-projection estimation",
            action="store_true", default=True)
        self.parser.add_argument("--mix_rate",
            nargs="+",
            help="if --use the pre-define curriculum learning, use this value to control the curriculum learning: e.g. set to [0,0.5], and total epoch=20 we train the 1st level 10 epoch, 2nd level last 10 epoch.",
            type=float,
            default=[0, 0.5])
        self.parser.add_argument("--weather",
            type=str,
            help="the weather you want to train",
            choices=["all", "clear", "rain", "fog", "snow"],
            default="rain")
        self.parser.add_argument("--debug",
            help="0:no debug,1:output per epoch,2:output every image,3 highest debug,out put all",
            type=int,
            default=0)
        self.parser.add_argument("--cta_wadd",
            type=float,
            help="contrast_weight_add;in paper, named w_cst, the original contrastive loss weight",
            default=1e-2)
        self.parser.add_argument("--only_contrast", "--only",
            type=int,
            default=-1,
            help="training will only do the contrastive learning in 'only_contrast' level")
        self.parser.add_argument("--cta_wpat",
            type=int,
            default=0,
            help="contrast_patient;only use in ablative study. If set to 1, at each level ,model will begin the contrastive learning after 1 epoch ")

        self.parser.add_argument("--start_level",
            type=int,
            help="set start level or mix part: when use curriculum, means the start level,as the mix,means the mix part(1==all,2==max)",
            default=-1
        )
        self.parser.add_argument("--max_patience", "--maxp",
            type=int,
            help="set max_patience",
            default=3
        )
        self.parser.add_argument("--contrast_with",
            nargs="+",
            type=int,
            help="contrast target level, for example,[0,0,1] means the 0th level contrast with 0th level, 1st level contrast with 0th level, 2nd level contrast with 1th level",
            default=[1, 2, None]
        )
        self.parser.add_argument("--scales",
            nargs="+",
            type=int,
            help="scales used in the loss",
            default=[0, 1, 2, 3])
        self.parser.add_argument("--loss",
            type=str,
            help="loss function",
            default="log"
        )
        self.parser.add_argument("--do_save",
            help="if set,save the model",
            action="store_true",
        )
        self.parser.add_argument("--gan",
            help="if set,use gan as the 2nd class",
            action="store_true",
        )
        self.parser.add_argument("--test_with_torch", '-twt',
            action="store_true",
            help="if set,use gpu to test"
        )
        self.parser.add_argument("--test_with_weather", '-tww',
            action="store_true",
            help="if set,use weather kitti to test"
        )
        self.parser.add_argument("--self_supervised", '-ss',
            action="store_true",
            help="if set,use self_supervised"
        )
        self.parser.add_argument("--local_rank", type=int, default=0)
        self.parser.add_argument("--cuda_devices", "--cuda", type=int, default=0)
        self.parser.add_argument("--change_freq", "--cf", type=int, default=2, help="change the w_curr every cf epoch")
        self.parser.add_argument("--max_wurr", "--mw", type=float, default=10, help="max time of w_curr")
        # TRAINING options
        self.parser.add_argument("--model_name",
            type=str,
            help="the name of the folder to save the model in",
            default="mdp")
        self.parser.add_argument("--split",
            type=str,
            help="which training split to use",
            default="eigen_full_left")
        self.parser.add_argument("--num_layers",
            type=int,
            help="number of resnet layers",
            default=50,
            choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
            type=str,
            help="dataset to train on",
            default="kitti",
            choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"])
        self.parser.add_argument("--height",
            type=int,
            help="input image height",
            default=192)
        self.parser.add_argument("--width",
            type=int,
            help="input image width",
            default=640)
        # the following options are used in the original planedepth, please refer to the original paper for more details
        self.parser.add_argument("--alpha_smooth",
            type=float,
            help="disparity smoothness weight",
            default=0.04)
        self.parser.add_argument("--self_distillation",
            type=float,
            help="self_distillation weight",
            default=0.)
        self.parser.add_argument("--gamma_smooth",
            type=float,
            help="gamma of smooth loss",
            default=2)
        self.parser.add_argument("--disparity_smoothness",
            type=float,
            help="disparity smoothness weight",
            default=0.001)
        self.parser.add_argument("--alpha_pc",
            type=float,
            help="perceptual loss weight",
            default=0.1)
        self.parser.add_argument("--disp_min",
            type=float,
            help="minimum depth",
            default=2.)
        self.parser.add_argument("--disp_max",
            type=float,
            help="maximum depth",
            default=300.)
        self.parser.add_argument("--disp_levels",
            type=int,
            help="num levels of disp",
            default=49)
        self.parser.add_argument("--disp_layers",
            type=int,
            help="num layers of disp",
            default=2)
        self.parser.add_argument("--novel_frame_ids",
            nargs="+",
            type=int,
            help="frames to load",
            default=[])
        self.parser.add_argument("--net_type",
            type=str,
            help="train which network",
            default="plane",
        )
        self.parser.add_argument("--num_ep",
            type=int,
            help="train which stage",
            default=8)
        self.parser.add_argument("--warp_type",
            type=str,
            help="the type of warp",
            default="disp_warp",
            choices=["depth_warp", "disp_warp", "homography_warp"])
        self.parser.add_argument("--match_aug",
            action="store_true",
            help="if set, use color augmented data to compute loss")
        self.parser.add_argument("--use_denseaspp",
            action="store_true",
            help="use DenseAspp block in ResNet",
            default=defalut_denseaspp)
        self.parser.add_argument("--use_mom",
            action="store_true",
            help="use mirror occlusion mask")
        self.parser.add_argument("--flip_right",
            action="store_true",
            help="use fliped right image to train",
            default=defalut_flip)
        self.parser.add_argument("--pc_net",
            type=str,
            help="the type of net to compute pc loss",
            default="vgg19",
            choices=["vgg19", "resnet18"])
        self.parser.add_argument("--xz_levels",
            type=int,
            help="num levels of xz plane",
            default=14)
        self.parser.add_argument("--yz_levels",
            type=int,
            help="num levels of yz plane",
            default=0)
        self.parser.add_argument("--use_mixture_loss",
            action="store_true",
            help="use mixture loss",
            default=defalut_mixloss)
        self.parser.add_argument("--alpha_self",
            type=float,
            help="perceptual loss weight",
            default=0.)
        self.parser.add_argument("--depth_regression_space",
            type=str,
            help="how to compute regression depth",
            default="inv",
            choices=["inv", "exp"])
        self.parser.add_argument("--render_probability",
            action="store_true",
            help="If set, render probability as NeRF")
        self.parser.add_argument("--plane_residual",
            action="store_true",
            help="If set, use residual plane based on init plane",
            default=defalut_planeres)
        self.parser.add_argument("--no_crop",
            action="store_true",
            help="if set, do not use resize crop data aug")
        self.parser.add_argument("--pe_type",
            type=str,
            help="the type of positional embedding",
            default="neural",
            choices=["neural", "frequency"])
        self.parser.add_argument("--use_colmap",
            action="store_true",
            help="if set, use colmap instead of predicting pose by posenet")
        self.parser.add_argument("--colmap_path",
            type=str,
            help="path to the colmap data",
            default="./kitti_colmap")
        self.parser.add_argument("--no_stereo",
            action="store_true",
            help="if set, disable stereo supervised")

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
            type=int,
            help="batch size",
            default=8)
        self.parser.add_argument("--learning_rate",
            type=float,
            help="learning rate",
            default=1e-4)
        self.parser.add_argument("--beta_1",
            type=float,
            help="beta1 of Adam",
            default=0.5)
        self.parser.add_argument("--beta_2",
            type=float,
            help="beta2 of Adam",
            default=0.999)
        self.parser.add_argument("--start_epoch",
            type=int,
            help="epoch at start",
            default=0)
        self.parser.add_argument("--num_epochs",
            type=int,
            help="number of epochs",
            default=50)
        self.parser.add_argument('--milestones',
            default=[30, 40], nargs='*',
            help='epochs at which learning rate is divided by 2')
        self.parser.add_argument("--scheduler_step_size",
            nargs="+",
            type=int,
            help="step size of the scheduler",
            default=[20, 25, 29])

        # ABLATION options
        self.parser.add_argument("--avg_reprojection",
            help="if set, uses average reprojection loss",
            action="store_true")
        self.parser.add_argument("--automask",
            help="if set, do auto-masking",
            action="store_true")

        # SYSTEM options
        self.parser.add_argument("--num_workers",
            type=int,
            help="number of dataloader workers",
            default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
            type=str,
            help="name of model to load")
        self.parser.add_argument("--models_to_load",
            nargs="+",
            type=str,
            help="models to load",
            default=["encoder", "depth"])
        self.parser.add_argument("--stage1_weights_folder",
            type=str,
            help="path of teacher model to load")

        # LOGGING options
        self.parser.add_argument("--log_frequency",
            type=int,
            help="number of batches between each tensorboard log",
            default=500)

        self.parser.add_argument("--log_img_frequency",
            type=int,
            help="number of batches between each tensorboard log",
            default=250)

        self.parser.add_argument("--use_ssim",
            help="if set, use ssim in the loss",
            action="store_true")

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
            help="if set evaluates in stereo mode",
            action="store_true")
        self.parser.add_argument("--eval_mono",
            help="if set evaluates in mono mode",
            action="store_true")
        self.parser.add_argument("--disable_median_scaling",
            help="if set disables median scaling in evaluation",
            action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
            help="if set multiplies predictions by this number",
            type=float,
            default=1)
        self.parser.add_argument("--ext_disp_to_eval",
            type=str,
            help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
            type=str,
            default="eigen_raw",
            help="which split to run eval on")
        self.parser.add_argument("--save_depth",
            help="if set saves depth predictions to npy",
            action="store_true")
        self.parser.add_argument("--no_eval",
            help="if set disables evaluation",
            action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
            help="if set assume we are loading eigen results from npy but "
                 "we want to evaluate using the new benchmark.",
            action="store_true")
        self.parser.add_argument("--eval_out_dir",
            help="if set will output the disparities to this folder",
            type=str)
        self.parser.add_argument("--post_process",
            help="if set will perform the flipping post processing "
                 "from the original monodepth paper",
            action="store_true")
        self.parser.add_argument("--save_strategy",
            choices=["overwrite", "append"],
            default="overwrite",
            help="set to append if you want to continue save models"
        )

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
