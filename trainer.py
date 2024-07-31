# coding=utf-8
# The full version of the trainer.py for WeatherDepth based on MonoViT/Planedepth. We additionally provide the ablation study part code: mixture training(in paper table v, w/o CC), curriculum training without contrast learning(paper table v, w/o C), pre-defined curriculum learning, and a semi-supervised curriculum-schedule learning strategy. The code is based on the original Monodepth2 codebase.
# Author: Jiyuan Wang
# Created: 2024-7-31
# Origin used for paper: https://arxiv.org/abs/2310.05556v2
# Hope you can cite our paper if you use the code for your research.
from __future__ import absolute_import, division, print_function
import copy
import random

import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from tensorboardX import SummaryWriter
import copy
import json

from layers import *

from my_utils import *
import options as g
from evaluate_depth_HR import *
import warnings

warnings.filterwarnings("ignore")


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed) # torch doc says that torch.manual_seed also work for CUDA
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(self, options):
        # region pre-set
        pid = os.getpid()
        print('pid: ', pid)
        self.opt = modify_opt(options)
        print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
        self.device = torch.device("cuda")
        if self.opt.net_type == "vit":
            # MonoViT model settings: use 640*192 resolution and monocular training(instead of stereo training)
            self.opt.flip_right = False
            self.num_scales = len(self.opt.scales)
            self.opt.novel_frame_ids = [-1, 1]
            self.num_pose_frames = 2
            self.opt.width = 640
            self.opt.height = 192
            self.opt.split = "eigen_zhou"
            print("Using SSIM loss", end=',')
            self.ssim = SSIM()
            self.ssim.to(self.device)
        # multi_gpu setup
        if self.opt.use_multi_gpu:
            dist.init_process_group(backend='nccl')
            self.local_rank = self.opt.local_rank
            self.opt.batch_size = self.opt.batch_size // torch.cuda.device_count()
            torch.cuda.set_device(self.local_rank)
            init_seeds(1 + self.local_rank, not self.opt.debug)
        else:
            init_seeds(0, not self.opt.debug)

        self.opt.model_name = "{}{}_{}{}_{}".format("tmp" if self.opt.debug else "", self.opt.model_name,
            self.opt.train_strategy, str(self.opt.curr_version), self.opt.weather)
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        self.save_folder = os.path.join(self.log_path, "models", "weights_{}")

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        if self.opt.flip_right:
            self.opt.batch_size = self.opt.batch_size // 2
        # region mix_rate
        # NOTE: mix rate only use in ablation study! If you just want to train the model, please ignore this region.
        # please refer to the help of opt.mixrate.
        nedone, rcd_rate = False, 0
        for rate in self.opt.mix_rate:
            if rate > 1:
                nedone, rcd_rate = True, int(rate)
                break
        if not nedone:
            self.opt.mix_rate.append(1)
        else:
            tmp = []
            for i in range(rcd_rate):
                tmp.append((i + 1) / rcd_rate)
            self.opt.mix_rate = tmp
        print("Mixing rate is: ", self.opt.mix_rate)
        # endregion
        self.parameters_to_train = []
        self.target_sides = ["r"]
        # endregion
        # region building network
        self.create_models()
        if self.opt.net_type == "vit":
            self.params = [{
                "params": self.parameters_to_train,
                "lr": 1e-4
            },
                {
                    "params": list(self.models["encoder"].parameters()),
                    "lr": 5e-5
                }]
            self.model_optimizer = optim.AdamW(self.params)
            self.model_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.model_optimizer, 0.9)
        elif self.opt.net_type == "monodepth2":
            self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
            self.model_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.model_optimizer, milestones=self.opt.scheduler_step_size, gamma=0.1)
        elif self.opt.net_type == "plane":
            self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate, betas=(self.opt.beta_1, self.opt.beta_2))
            self.model_lr_scheduler = optim.lr_scheduler.MultiStepLR(
                self.model_optimizer, milestones=self.opt.milestones, gamma=0.5)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:  ", self.opt.log_dir)
        print("Training is using:  ", self.device)
        # endregion
        # region setup dataset
        datasets_dict = {"kitti": datasets.KITTIRAWDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "./splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))

        if self.opt.debug:
            train_filenames = train_filenames[:100]
            val_filenames = val_filenames[:40]
            self.opt.num_workers = 0 if self.opt.debug >= 2 else self.opt.num_workers  # 待完善
            self.opt.num_epochs = min(10, self.opt.num_epochs) if (self.opt.debug >= 1 and self.opt.start_epoch == 0) else self.opt.num_epochs

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // (self.opt.batch_size * torch.cuda.device_count()) * (self.opt.num_epochs - self.opt.start_epoch)

        def worker_init(worker_id):
            worker_seed = torch.utils.data.get_worker_info().seed % (2 ** 32)
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        self.train_dataset = self.dataset(self.opt, train_filenames, is_train=True)
        if self.opt.use_multi_gpu:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
            self.train_loader = DataLoader(
                self.train_dataset, self.opt.batch_size, False,
                num_workers=self.opt.num_workers, sampler=self.train_sampler, pin_memory=True, drop_last=True, worker_init_fn=worker_init, collate_fn=rmnone_collate)
        else:
            self.train_loader = DataLoader(
                self.train_dataset, self.opt.batch_size, False,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, worker_init_fn=worker_init, collate_fn=rmnone_collate)

        self.val_dataset = self.dataset(self.opt, val_filenames, is_train=False)
        # in online_validation, we don't use the distributed sampler
        self.val_loader = DataLoader(
            self.val_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)
        # endregion
        # region Other functions
        if self.opt.net_type == "plane":
            if self.opt.pc_net == "vgg19":
                self.pc_net = Vgg19_pc().cuda()
            elif self.opt.pc_net == "resnet18":
                self.pc_net = Resnet18_pc().cuda()
            self.softmax = nn.Softmax(1)
        else:
            self.backproject_depth = {}
            self.project_3d = {}
            for scale in self.opt.scales:
                h = int(self.opt.height // (2 ** scale))
                w = int(self.opt.width // (2 ** scale))
                self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
                self.backproject_depth[scale].to(self.device)
                self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
                self.project_3d[scale].to(self.device)

        self.init_key, self.best_abs, self.record_epoch = 100, 10, 0
        self.record_key = {g.load_map['average']: [self.init_key]}
        self.class_level = self.opt.start_level
        self.w_curr = self.opt.cta_wadd if self.opt.cta_wadd > 0 else 0.1
        if self.opt.curr_version <= 5 and self.opt.train_strategy == 'cur':
            self.patience_max = self.opt.max_patience if not self.opt.debug else 1
            self.schedul_class()

        if self.opt.use_multi_gpu:
            if dist.get_rank() == 0:
                self.create_summary_writer()
        else:
            self.create_summary_writer()

        print("Loading ground truth depths...", end=' ')
        gt_path = modify_opt(path=os.path.join(self.opt.data_path, "gt_depths.npz"))
        self.gt_depths = np.load(gt_path, allow_pickle=True)
        print("√")
        # endregion

    # 预创建函数
    def create_summary_writer(self):
        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items , {:d} validation items\n".format(
            len(self.train_dataset), len(self.val_dataset)))
        remove_logfolder(self.log_path, self.opt.save_strategy == "overwrite")
        if self.opt.net_type == "plane":
            save_code("./trainer.py", self.log_path)
            save_code("./networks/depth_decoder.py", self.log_path)
            save_code("./train_ResNet.sh", self.log_path)
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.save_opts()

        self.log_file = open(os.path.join(self.log_path, "logs.log"), 'w')

    def create_models(self):
        print("==>Building network:")
        self.models = {}
        if self.opt.net_type == "vit":
            print("train vit net")
            self.models["encoder"] = networks.mpvit_small()
            self.models["encoder"].num_ch_enc = [64, 128, 216, 288, 288]
            self.models["encoder"].to(self.device)

            self.models["depth"] = networks.HR_DepthDecoder()
            self.models["depth"].to(self.device)
            self.parameters_to_train += list(self.models["depth"].parameters())
        elif self.opt.net_type == "plane":
            print("train plane net")
            self.models["encoder"] = networks.ResnetEncoder(self.opt.num_layers, True)
            self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc,
                self.opt.disp_levels,
                self.opt.disp_min,
                self.opt.disp_max,
                self.opt.num_ep,
                pe_type=self.opt.pe_type,
                use_denseaspp=self.opt.use_denseaspp,
                xz_levels=self.opt.xz_levels,
                yz_levels=self.opt.yz_levels,
                use_mixture_loss=self.opt.use_mixture_loss,
                render_probability=self.opt.render_probability,
                plane_residual=self.opt.plane_residual)
            if self.opt.use_multi_gpu:
                for model_name, model in self.models.items():
                    model = model.to(self.device)
                    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                    self.models[model_name] = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
                    self.parameters_to_train += list(self.models[model_name].parameters())
            else:
                for model_name, model in self.models.items():
                    model = model.to(self.device)
                    self.models[model_name] = model
                    self.parameters_to_train += list(model.parameters())
        elif self.opt.net_type == "monodepth2":
            print("train monodepth2 net")
            self.models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, True)
            self.models["encoder"].to(self.device)

            self.models["depth"] = networks.Monov2Decoder(
                self.models["encoder"].num_ch_enc, self.opt.scales)
            self.models["depth"].to(self.device)

            self.parameters_to_train += list(self.models["encoder"].parameters())
            self.parameters_to_train += list(self.models["depth"].parameters())
        if not self.opt.net_type == "plane":
            self.models["pose_encoder"] = networks.ResnetEncoder(18, True, num_input_images=self.num_pose_frames)
            self.models["pose_encoder"].to(self.device)

            self.models["pose"] = networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc, num_input_features=1,
                num_frames_to_predict_for=2)
            self.models["pose"].to(self.device)

            self.parameters_to_train += list(self.models["pose_encoder"].parameters())
            self.parameters_to_train += list(self.models["pose"].parameters())

    def set_train(self):
        for m in self.models.values():
            m.train()

    def set_eval(self):
        for m in self.models.values():
            m.eval()

    # 训练序列
    def train(self):
        self.epoch = 0
        self.step = 0
        for self.epoch in range(self.opt.start_epoch):
            self.model_lr_scheduler.step()
        if self.opt.net_type == "vit":
            depth_lr = self.model_optimizer.param_groups[1]['lr']
            pose_lr = self.model_optimizer.param_groups[0]['lr']
            print(f'\nStarting from epoch {self.epoch} and current learning rate for depth is {depth_lr} and pose lr is {pose_lr}')
        elif self.opt.net_type == "monodepth2":
            starting_lr = self.model_optimizer.param_groups[0]['lr']
            print(f'\nStarting from epoch {self.epoch} and current learning rate is {starting_lr}')

        print("==>Training started...")
        for self.epoch in range(self.opt.start_epoch, self.opt.num_epochs):
            # Switch datasets according to the scale of the mix rate, only used in ablation study
            if self.opt.curr_version < 2:
                if not (self.epoch + 1) <= self.opt.mix_rate[self.class_level] * self.opt.num_epochs:
                    self.schedul_class()
            schedule_loss = self.run_epoch()
            if not self.opt.use_multi_gpu or dist.get_rank() == 0:
                if self.opt.do_save:
                    self.save_model(str(self.epoch))  # "last_model
                else:
                    self.save_model("last")
            # region update w_curr
            if self.opt.curr_version <= 5:
                self.record_epoch += 1
                if self.record_epoch % self.opt.change_freq == 0 and self.record_epoch > 0 and self.opt.train_strategy == 'cur':
                    if 0 <= self.opt.cta_wadd <= 0.02:
                        self.w_curr = min(self.w_curr * 2, self.opt.cta_wadd * self.opt.max_wurr)
                    elif self.opt.cta_wadd > 0 and self.opt.net_type == 'vit':
                        self.w_curr = min(self.w_curr * 2, self.opt.cta_wadd * self.opt.max_wurr)
                    elif self.opt.cta_wadd == -1:  # self-adaptive
                        self.w_curr = schedule_loss['pure_loss'] / (5 * schedule_loss['contrast_loss'] / self.w_curr) if schedule_loss['contrast_loss'] > 0 else 0.1
                    elif self.opt.cta_wadd == -2:  # pre-defined
                        self.w_curr = {0: 0.1, 1: 0.1, 2: 0.08}[self.class_level]

                if self.opt.train_strategy == 'cur' and self.opt.curr_version >= 2:
                    if self.opt.self_supervised:
                        if self.average_patience >= self.patience_max and self.class_level < 2:
                            finish = self.schedul_class()
                            if finish:
                                break
                    else:
                        if self.independent_patience >= self.patience_max and self.average_patience >= self.patience_max:
                            finish = self.schedul_class()
                            if finish:
                                break
            # endregion
        print("Training finished after {} epochs,".format(self.epoch))

    def run_epoch(self):
        if self.opt.use_multi_gpu:
            self.train_sampler.set_epoch(self.epoch)
        record_loss, record_contast_loss, all_batch = 0, 0, 0
        self.set_train()
        self.train_dataset.do_contrast = self.do_contrast()
        for batch_idx, inputs in enumerate(self.train_loader):
            if inputs is None:
                self.model_optimizer.zero_grad()
                self.model_optimizer.step()
                self.step += 1
                continue
            before_op_time = time.time()

            if self.opt.flip_right:
                inputs = self.add_flip_right_inputs(inputs)

            outputs, outputs_cst, losses = self.process_batch(inputs)
            record_loss += losses["loss/total_loss"].item()
            record_contast_loss += losses["loss/contrast_loss"].item() if self.do_contrast() else -1

            self.model_optimizer.zero_grad()
            losses["loss/total_loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % 100 == 0 and self.step < self.opt.log_frequency
            late_phase = self.step % self.opt.log_frequency == 0

            if early_phase or late_phase:
                average_loss = {"pure_loss": (record_loss - record_contast_loss) / (batch_idx + 1) if record_contast_loss > 0 else record_loss / (batch_idx + 1),
                                "contrast_loss": record_contast_loss / (batch_idx + 1), "loss": record_loss / (batch_idx + 1)}
                if self.opt.use_multi_gpu:
                    if dist.get_rank() == 0:
                        self.log_loss(batch_idx, duration, average_loss)
                else:
                    self.log_loss(batch_idx, duration, average_loss)

            self.step += 1
            if batch_idx == 20:
                if self.opt.use_multi_gpu:
                    if dist.get_rank() == 0:
                        self.log_img("train", inputs, outputs, outputs_cst)
                else:
                    self.log_img("train", inputs, outputs, outputs_cst)
            all_batch = batch_idx
            if not self.opt.use_multi_gpu:
                del inputs, outputs, outputs_cst, losses

        average_loss = {"pure_loss": (record_loss - record_contast_loss) / (all_batch + 1) if record_contast_loss > 0 else record_loss / (all_batch + 1)
            , "contrast_loss": record_contast_loss / (all_batch + 1), "loss": record_loss / (all_batch + 1)}
        if not self.opt.use_multi_gpu or dist.get_rank() == 0:
            with torch.no_grad():
                self.val(average_loss)
        if self.opt.use_multi_gpu:
            dist.barrier()
        self.model_lr_scheduler.step()
        return average_loss

    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.net_type == "plane":
            features = self.models["encoder"](inputs[("color_aug", "l")])
            outputs = self.models["depth"](features, inputs["grid"])
        else:
            feats = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models['depth'](feats)
            outputs.update(self.predict_poses(inputs))

        self.pred_novel_images(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)
        losses["loss/contrast_loss"] = torch.tensor(-1, device=self.device)
        outputs_cst = None
        if self.do_contrast():
            if self.opt.net_type == "plane":
                if self.opt.train_strategy == 'mix':
                    with torch.no_grad():
                        outputs_cst = self.models["depth"](self.models["encoder"](inputs["color_cst", "l"]), inputs["grid"])
                else:
                    outputs_cst = self.models["depth"](self.models["encoder"](inputs["color_cst", "l"]), inputs["grid"])
                # compute contrast loss
                load_name = 'disp'
                if self.opt.curr_version == 5:
                    org_label = outputs[load_name].clone().detach()
                    cst_label = outputs_cst[load_name].clone().detach()
                    if self.opt.train_strategy == 'cur':
                        if self.opt.contrast_with[self.class_level] > self.class_level:
                            contrast_loss = self.compute_surpervised_loss(outputs_cst[load_name], org_label)
                        else:
                            contrast_loss = self.compute_surpervised_loss(outputs[load_name], cst_label)
                    elif self.opt.train_strategy == 'mix':
                        contrast_loss = self.compute_surpervised_loss(outputs[load_name], cst_label)
                elif self.opt.curr_version == 4:
                    contrast_loss = self.compute_surpervised_loss(outputs[load_name], outputs_cst[load_name])
                losses["loss/contrast_loss"] = self.w_curr * contrast_loss
                losses["loss/total_loss"] += losses["loss/contrast_loss"]
            else:
                outputs_cst = self.models["depth"](self.models["encoder"](inputs["color_cst", 'l', 0]))
                self.pred_novel_images(inputs, outputs_cst, "cst")
                losses = self.compute_contrast_loss(outputs, outputs_cst, losses)

        if self.opt.debug >= 2:
            print('\033[91m' + "loss:" + str(losses["loss/total_loss"].item())[:5] + 'contrast_loss:' +
                  str(losses["loss/contrast_loss"].item())[:5] + '\033[0m')

        return outputs, outputs_cst, losses

    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences."""
        outputs = {}
        frameIDs = self.opt.novel_frame_ids + [0]
        if self.num_pose_frames == 2:
            pose_feats = {f_i: inputs["color", f_i, 0] for f_i in frameIDs}
            for f_i in frameIDs:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

                    outputs[("pose_feats", 0, f_i)] = pose_inputs

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        return outputs

    def val(self, loss):
        """online validation"""
        if self.opt.debug == 1:
            print("Train √  Begin validation...")
        writer = self.writers["val"]
        STEREO_SCALE_FACTOR = 5.4
        grid = meshgrid(torch.linspace(-1, 1, g.defalut_width), torch.linspace(-1, 1, g.defalut_height), indexing="xy")
        grid = torch.stack(grid, dim=0).cuda()
        cv2.setNumThreads(0)
        load_val_mode = self.val_dataset.folder_name['train']
        self.set_eval()
        MIN_DEPTH, MAX_DEPTH = g.MIN_DEPTH, g.MAX_DEPTH
        error_all, finish = [], False
        for ld_mode in load_val_mode:
            self.val_dataset.specify_data(ld_mode)
            pred_disps = []
            with torch.no_grad():
                for batch_idx, inputs in enumerate(self.val_loader):
                    input_color = inputs[("color", "l")].cuda() if self.opt.net_type == "plane" else inputs[("color", 0, 0)].cuda()
                    # print(input_color.shape)
                    if self.opt.net_type == "plane":
                        grids = grid[None, ...].expand(input_color.shape[0], -1, -1, -1)
                        if self.opt.use_multi_gpu:
                            features = self.models["encoder"].module(input_color)
                            outputs = self.models["depth"].module(features, grids)
                        else:
                            features = self.models["encoder"](input_color)
                            outputs = self.models["depth"](features, grids)
                        pred_disp = outputs["disp"][:, 0]
                    else:
                        outputs = self.models["depth"](self.models["encoder"](input_color))
                        pred_disp, _ = disp_to_depth(outputs[("disp", 0)], 0.1, 100)
                        pred_disp = pred_disp[:, 0]
                    pred_disp = pred_disp.cpu().numpy()
                    pred_disps.append(pred_disp)
            self.opt.disable_median_scaling = True
            self.opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
            errors, ratios = [], []

            # region compute errors

            pred_disps = np.concatenate(pred_disps)
            for i in range(pred_disps.shape[0]):
                gt_depth = self.gt_depths[i]
                gt_height, gt_width = gt_depth.shape[:2]

                pred_disp = pred_disps[i]
                pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
                if self.opt.net_type == "plane":
                    pred_depth = 0.1 * 0.58 * g.defalut_width / (pred_disp)
                else:
                    pred_depth = 1 / pred_disp
                if i == 352:
                    if self.epoch == 0 and ld_mode == 'rgb/data':
                        VisulizeDepth(None, gt_depth, 'val_gt_depth', process='None', writer=writer)
                    VisulizeDepth(None, pred_depth, 'val_pred_depth/{}'.format(ld_mode), process='None', writer=writer, rcd=self.epoch)
                # create mask
                gt_depth[gt_depth < MIN_DEPTH] = MIN_DEPTH
                gt_depth[gt_depth > MAX_DEPTH] = MAX_DEPTH
                mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

                crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                 0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
                crop_mask = np.zeros(gt_depth.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

                pred_depth = pred_depth[mask]
                gt_depth = gt_depth[mask]
                pred_depth *= self.opt.pred_depth_scale_factor
                if self.opt.net_type == 'vit':
                    ratio = np.median(gt_depth) / np.median(pred_depth)
                    ratios.append(ratio)
                    pred_depth *= ratio

                pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
                pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

                errors.append(compute_errors(gt_depth, pred_depth))
            mean_errors = np.array(errors).mean(0)
            if not self.opt.self_supervised:
                if g.load_map[ld_mode] not in self.record_key:
                    self.record_key[g.load_map[ld_mode]] = [self.init_key]
                self.record_key[g.load_map[ld_mode]].append(mean_errors[2])
            error_all.append(mean_errors)
            # endregion

            for ind, error in enumerate(mean_errors):
                writer.add_scalar('{}/{}'.format(g.load_map[ld_mode], g.index_map[ind]), error, self.epoch)
            if self.opt.debug >= 1:
                print("-> {}: {}".format(g.load_map[ld_mode], mean_errors))

        # region recoder error
        mean_errors = np.array(error_all).mean(0)
        var_errors = np.array(error_all).var(0)
        if not self.opt.self_supervised:
            self.record_key[g.load_map['average']].append(mean_errors[2])
        current_abs = mean_errors[0]
        if current_abs < self.best_abs:
            self.best_abs = current_abs
            self.best_epoch = self.epoch
            self.save_model('best' + str(self.class_level))
        for ind, error in enumerate(mean_errors):
            writer.add_scalar('{}/{}'.format(g.load_map["average"], g.index_map[ind]), error, self.epoch)
            if self.opt.debug >= 1:
                print("-> {}_{}: {}".format(g.load_map["average"], g.index_map[ind], error), end=' ')
            writer.add_scalar('{}/{}'.format(g.load_map["variance"], g.index_map[ind]), var_errors[ind], self.epoch)
            if self.opt.debug >= 1:
                print(", {}_{}: {}".format(g.load_map["variance"], g.index_map[ind], var_errors[ind]))
        # endregion
        self.set_train()
        # record loss
        writer.add_scalar('loss/total_loss', loss['loss'], self.epoch), writer.add_scalar('loss/pure_loss', loss['pure_loss'], self.epoch), writer.add_scalar('loss/contrast_loss',
            loss['contrast_loss'], self.epoch), writer.add_scalar('loss/real_contrast_loss', loss['contrast_loss'] / self.w_curr, self.epoch)
        if not self.opt.self_supervised:
            # we alse provide the way to use the online validation to schedule the curriculum learning, but we think it is not satisfy the self-supervised learning baseline.
            into_which, difference, average_dif = list(set(self.val_dataset.folder_name['train']) & set(g.class_map[self.class_level])), 0, self.record_key[g.load_map['average']][-1] - self.record_key[g.load_map['average']][-2]
            if self.opt.train_strategy == 'cur':
                for into in into_which:
                    difference += self.record_key[g.load_map[into]][-1] - self.record_key[g.load_map[into]][-2]
                difference /= len(into_which)
                if self.opt.debug:
                    print('\033[91m' + "-> " + str(self.class_level) + " level rmse difference= " + str(difference) + " ,average_dif= " + str(average_dif) + '\033[0m')
                if (difference > 0):
                    self.independent_patience += 1
                if (average_dif > 0):
                    self.average_patience += 1
        else:
            difference = 0
            self.independent_patience = self.opt.max_patience + 1
            self.record_key[g.load_map['average']].append(loss['pure_loss'])
            average_dif = self.record_key[g.load_map['average']][-1] - self.record_key[g.load_map['average']][-2]
            thereshold = -0.0005 if self.opt.net_type == 'vit' else 0
            if self.opt.debug:
                print('\033[91m' + "-> " + "average_dif= " + str(average_dif) + '\033[0m')
            if self.opt.train_strategy == 'cur':
                if (average_dif > thereshold):
                    self.average_patience += 1
        return difference, average_dif

    def pred_novel_images(self, inputs, outputs, mode='train'):
        """ Same with plandepth and monovit, we just mix them together. """
        if self.opt.net_type == 'plane':
            B, N, H, W = outputs["probability"].shape
            source_side = "l"
            disps = outputs["disp_layered"]
            pix_coords = meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
            pix_coords = torch.stack(pix_coords, dim=0).cuda().float()
            pix_coords = pix_coords[None, None, ...].expand(B, N, -1, -1, -1).clone()
            pix_coords[:, :, 0, :, :] += disps
            pix_coords[:, :, 0, :, :] /= (W - 1)
            pix_coords[:, :, 1, :, :] /= (H - 1)
            pix_coords = (pix_coords - 0.5) * 2
            pix_coords = pix_coords.reshape(B * N, 2, H, W)
            pix_coords = pix_coords.permute(0, 2, 3, 1)
            padding_mask = outputs["padding_mask"][:, :, None, :, :]

            if self.opt.match_aug:
                color_name = "color_aug"
            else:
                color_name = "color"

            features = torch.cat([inputs[(color_name, source_side)][:, None].expand(-1, N, -1, -1, -1).reshape(B * N, 3, H, W), outputs["logits"].reshape(B * N, 1, H, W)], dim=1)

            if self.opt.use_mixture_loss:
                features = torch.cat([features, outputs["sigma"].reshape(B * N, 1, H, W)], dim=1)

            rec_features = F.grid_sample(
                features,
                pix_coords,
                padding_mode="zeros",
                align_corners=True).reshape(B, N, -1, H, W)
            # only stereo could compute as this.
            rec_features = rec_features * padding_mask
            outputs[("rgb_rec_layered", "r")] = rec_features[:, :, :3, ...]
            outputs[("logit_rec", "r")] = rec_features[:, :, 3, ...]
            if self.opt.render_probability:
                alpha = 1. - torch.exp(-F.relu(outputs[("logit_rec", "r")][:, :-1, ...]) * outputs["dists"])
                ones = torch.ones_like(alpha[:, :1, ...])
                alpha = torch.cat([alpha, ones], dim=1)
                probability_rec = alpha * torch.cumprod(torch.cat([ones, 1. - alpha + 1e-10], dim=1), dim=1)[:, :-1, ...]
                outputs[("probability_rec", "r")] = probability_rec
            else:
                outputs[("probability_rec", "r")] = self.softmax(outputs[("logit_rec", "r")])
            if self.opt.use_mixture_loss:
                sigma_rec = rec_features[:, :, 4, ...].clone()
                # sigma_rec[sigma_rec==0] = 1.
                sigma_rec = torch.clamp(sigma_rec, 0.01, 1.)
                outputs[("sigma_rec", "r")] = sigma_rec
                outputs[("pi_rec", "r")] = pi_rec = outputs[("probability_rec", "r")]
                weights_rec = pi_rec / sigma_rec
                weights_rec = weights_rec / weights_rec.sum(1, True)
                outputs[("probability_rec", "r")] = weights_rec
            outputs[("rgb_rec", "r")] = (outputs[("rgb_rec_layered", "r")] * outputs[("probability_rec", "r")][:, :, None]).sum(1)
        else:
            for scale in self.opt.scales:
                disp = outputs[("disp", scale)]
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0
                _, depth = disp_to_depth(disp, 0.1, 100)
                outputs[("depth", scale)] = depth
                if self.opt.curr_version != 3 and mode != 'train':
                    continue
                for i, frame_id in enumerate(self.opt.novel_frame_ids):
                    if frame_id == "s":
                        T = inputs["stereo_T"]
                    else:
                        T = outputs[("cam_T_cam", 0, frame_id)]
                    cam_points = self.backproject_depth[source_scale](depth, inputs[("inv_K", source_scale)])
                    pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)
                    outputs[("sample", frame_id, scale)] = pix_coords
                    outputs[("color", frame_id, scale)] = F.grid_sample(inputs[("color", frame_id, source_scale)], outputs[("sample", frame_id, scale)], padding_mode="border",
                        align_corners=True)  # clear image

    def compute_reprojection_loss(self, pred, target):
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return reprojection_loss

    # region compute loss
    def perceptual_loss(self, pred, target, source=None):
        pred_vgg = self.pc_net(pred)
        target_vgg = self.pc_net(target)
        if source is not None:
            source_vgg = self.pc_net(source)

        loss_pc = 0
        for i in range(3):
            l_p = ((pred_vgg[i] - target_vgg[i]) ** 2).mean(1, True)
            if source is not None:  # automask
                l_p_auto = ((source_vgg[i] - target_vgg[i]) ** 2).mean(1, True)
                l_p, _ = torch.cat([l_p, l_p_auto], dim=1).min(1, True)
            loss_pc += l_p.mean()
        return loss_pc

    def compute_surpervised_loss(self, pred, target, valid_pixels=None):
        """ Calculate the supervision loss (depth cue loss) for forecasting. - valid_pixels Mask of a valid depth-cueping pixel (i.e., non-zero depth value)"""
        if valid_pixels is None:
            valid_pixels = torch.ones(target.shape, device=self.device)
        if self.opt.loss == 'log':
            loss = torch.log(torch.abs(target - pred) + 1) * valid_pixels
        elif self.opt.loss == 'l1':
            loss = F.smooth_l1_loss(pred, target, reduction='none') * valid_pixels
        loss = loss.sum() / (valid_pixels.sum() + 1e-7)
        return loss

    def compute_losses(self, inputs, outputs, mode='train'):
        losses = {}
        total_loss = 0
        if self.opt.net_type == 'plane':
            B, N, H, W = outputs["probability"].shape
            if mode == 'train':
                losses["loss/ph_loss"] = 0
                losses["loss/pc_loss"] = 0
                losses["loss/total_loss"] = 0
            color_name = "color"
            pred = outputs[("rgb_rec", "r")]  # The Reprojected Image, Is Equivalent To outputs[("color")]
            target = inputs[(color_name, "r")]
            # MHHloss
            error = torch.abs(outputs[("rgb_rec_layered", "r")] - target[:, None]).mean(2)  # 相当于outputs[("color_layer")]
            ph_loss = multimodal_loss(error, outputs[("sigma_rec", "r")], outputs[("pi_rec", "r")], dist='lap')  # .mean()
            ph_loss = ph_loss.mean()
            losses["loss/ph_loss"] += ph_loss
            total_loss += ph_loss
            # PCloss
            pc_loss = self.perceptual_loss(pred, target).mean()
            losses["loss/pc_loss"] += pc_loss
            total_loss += self.opt.alpha_pc * pc_loss
            # SMoothloss
            losses["loss/total_loss"] += total_loss
            smooth_loss = get_smooth_loss_disp(outputs["disp"][..., int(0.2 * W):], inputs[("color", "l")][..., int(0.2 * W):], gamma=self.opt.gamma_smooth)
            losses["loss/smooth_loss"] = smooth_loss
            losses["loss/total_loss"] += self.opt.alpha_smooth * smooth_loss
            return losses
        else:
            for scale in self.opt.scales:
                loss, reprojection_losses, identity_reprojection_losses, source_scale = 0, [], [], 0
                color = inputs[("color", 0, scale)]
                target = inputs[("color", 0, source_scale)]
                disp = outputs[("disp", scale)]

                for frame_id in self.opt.novel_frame_ids:
                    pred = outputs[("color", frame_id, scale)]  # This is the image after warp, the size is the same, this is the latitude of the same size as the image
                    reprojection_losses.append(self.compute_reprojection_loss(pred, target))
                reprojection_losses = torch.cat(reprojection_losses, 1)
                reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)

                for frame_id in self.opt.novel_frame_ids:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))
                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
                identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1, keepdim=True)
                identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape, device=self.device) * 0.00001

                to_optimise, idxs = torch.min(torch.cat((identity_reprojection_loss, reprojection_loss), dim=1), dim=1)
                loss += to_optimise.mean()
                # smoothloss
                mean_disp = disp.mean(2, True).mean(3, True)
                norm_disp = disp / (mean_disp + 1e-7)
                smooth_loss = get_smooth_loss_disp(norm_disp, color)

                loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
                total_loss += loss
                losses["loss/{}".format(scale)] = loss
            total_loss /= self.num_scales
            losses["loss/total_loss"] = total_loss

            return losses

    def compute_contrast_loss(self, outputs, outputs_cst, losses):
        '''calculate_the_contrast_loss'''
        if self.opt.train_strategy == 'cur':
            contrast_weight = self.w_curr if self.record_epoch >= self.opt.cta_wpat else 0
        else:
            contrast_weight = self.w_curr
        loss, load_name = 0, 'depth'
        for scale in self.opt.scales:
            org_label = outputs[(load_name, scale)].clone().detach()
            cst_label = outputs_cst[(load_name, scale)].clone().detach()
            if self.opt.curr_version == 3:
                combined = torch.cat([outputs["reprojection_loss_org/{}".format(scale)], outputs_cst["cst_reprojection_loss_org/{}".format(scale)]], dim=1)
                idx = torch.argmin(combined, dim=1).unsqueeze(1).detach()
                org_mask, cst_mask = (idx == 0).float(), (idx == 1).float()
                org_contrast_loss = self.compute_surpervised_loss(outputs_cst[(load_name, scale)], org_label, org_mask)
                cst_contrast_loss = self.compute_surpervised_loss(outputs[(load_name, scale)], cst_label, cst_mask)
                contrast_loss = contrast_weight * (org_contrast_loss + cst_contrast_loss) / (2 ** scale)
            elif self.opt.curr_version == 4 or self.opt.curr_version == 0:
                # Make a direct comparison
                contrast_loss = contrast_weight * self.compute_surpervised_loss(outputs[(load_name, scale)], outputs_cst[(load_name, scale)])
            elif self.opt.curr_version == 5:
                # Learn frome simple level results
                if self.opt.train_strategy == 'cur':
                    if self.opt.contrast_with[self.class_level] > self.class_level:
                        contrast_loss = contrast_weight * self.compute_surpervised_loss(outputs_cst[(load_name, scale)], org_label)
                    else:
                        contrast_loss = contrast_weight * self.compute_surpervised_loss(outputs[(load_name, scale)], cst_label)
                elif self.opt.train_strategy == 'mix':
                    contrast_loss = contrast_weight * self.compute_surpervised_loss(outputs[(load_name, scale)], cst_label)
            loss += contrast_loss
            losses['loss/contrast_loss_{}'.format(scale)] = contrast_loss
        losses["loss/contrast_loss"] = loss / self.num_scales
        if self.opt.debug >= 2:
            print('loss_contrast:{}'.format(losses["loss/contrast_loss"]))
        losses['loss/total_loss'] += losses["loss/contrast_loss"]

        return losses

    # endregion
    # Utility functions
    def add_flip_right_inputs(self, inputs):
        new_inputs = {}
        new_inputs[("color", "l")] = torch.cat([inputs[("color", "l")], inputs[("color", "r")].flip(-1)], dim=0)
        new_inputs[("color", "r")] = torch.cat([inputs[("color", "r")], inputs[("color", "l")].flip(-1)], dim=0)
        new_inputs[("color_aug", "l")] = torch.cat([inputs[("color_aug", "l")], inputs[("color_aug", "r")].flip(-1)], dim=0)
        try:
            new_inputs[("color_cst", "l")] = torch.cat([inputs[("color_cst", "l")], inputs[("color_cst", "r")].flip(-1)], dim=0)
        except:
            pass
        grid_fliped = inputs["grid"].clone()
        grid_fliped[:, 0, :, :] *= -1.
        grid_fliped = grid_fliped.flip(-1)
        new_inputs["grid"] = torch.cat([inputs["grid"], grid_fliped], dim=0)

        new_inputs["K"] = inputs["K"].repeat(2, 1, 1)
        new_inputs["inv_K"] = inputs["inv_K"].repeat(2, 1, 1)

        new_inputs[("Rt", "l")] = inputs[("Rt", "l")].repeat(2, 1, 1)
        new_inputs[("Rt", "r")] = inputs[("Rt", "r")].repeat(2, 1, 1)

        for novel_frame_id in self.opt.novel_frame_ids:
            new_inputs[("color", novel_frame_id)] = torch.cat([inputs[("color", novel_frame_id)], inputs[("color", novel_frame_id)].flip(-1)], dim=0)
            new_inputs[("color_aug", novel_frame_id)] = torch.cat([inputs[("color_aug", novel_frame_id)], inputs[("color_aug", novel_frame_id)].flip(-1)], dim=0)

        return new_inputs

    def do_contrast(self):
        mode = self.opt.only_contrast
        level = self.class_level
        condition = (mode == 0 and level > 0) or \
                    (mode == 1 and level > 1) or \
                    mode == -1 or \
                    (mode == -2 and level < 2)

        if self.opt.cta_wpat != 0:
            condition = condition and self.record_epoch >= self.opt.cta_wpat

        return condition

    def schedul_class(self):
        #Important! The function is used to schedule the curriculum learning. Please refer to the Algorithm 1 in the paper.(Maybe a little difference)
        self.record_epoch, self.independent_patience, self.average_patience, finish = 0, 0, 0, False
        self.class_level += 1
        if self.opt.self_supervised:
            self.patience_max = self.patience_max + 1 if self.patience_max <= 3 else self.patience_max
        self.w_curr = self.opt.cta_wadd if self.opt.cta_wadd > 0 else self.w_curr
        max_level = len(self.opt.mix_rate) if self.opt.curr_version < 2 else 2

        if self.class_level == max_level and self.opt.max_patience > 3 and self.class_level == 2:
            self.opt.load_weights_folder = self.log_path + "/best1"
            print("-> Load best model,which is ", self.best_epoch, " epoch")
            self.load_model()

        if self.class_level <= max_level:
            self.train_dataset.change_data(self.class_level)
        else:
            finish = True
        if self.opt.debug >= 1:
            print("-> Training change!")
        return finish

    def log_loss(self, batch_idx, duration, losses):
        """Print a logging statement to the terminal        """
        samples_per_sec = self.opt.batch_size * torch.cuda.device_count() / duration
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} |contrast: {:.5f} with w={:.5f} and real: {:.5f}|"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, losses["pure_loss"], losses["contrast_loss"], self.w_curr, losses["contrast_loss"] / self.w_curr, ))

    def log_img(self, mode, inputs, outputs, outputs_cst):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        j = 0
        if self.opt.net_type == 'plane':
            for frame_id in ["l", "r"] + self.opt.novel_frame_ids:
                writer.add_image(
                    "color_{}".format(frame_id),
                    inputs[("color", frame_id)][j].data, self.epoch)

            writer.add_image("color_aug", inputs[("color_aug", "l")][j].data, self.epoch)
            if "color_cst" in inputs:
                writer.add_image("color_cst", inputs[("color_cst", "l")][j].data, self.epoch)
            for frame_id in self.target_sides:
                writer.add_image(
                    "color_pred_{}".format(frame_id),
                    outputs[("rgb_rec", frame_id)][j].data, self.epoch)

            writer.add_image("disp/org", normalize_image(outputs["disp"][j]), self.epoch)
            if outputs_cst is not None:
                writer.add_image("disp/cst", normalize_image(outputs_cst["disp"][j]), self.epoch)
        else:
            scale = 0
            for frame_id in (self.opt.novel_frame_ids + [0]):  # 写color，color_pred，color_aug
                writer.add_image("color_{}_{}/{}".format(frame_id, 0, j), inputs[("color", frame_id, 0)][j].data, self.epoch)
                try:
                    writer.add_image("color_pred_{}_{}/{}".format(frame_id, 0, j), outputs[("color", frame_id, 0)][j].data, self.epoch)
                except KeyError:
                    pass
                if frame_id == 0:
                    writer.add_image("color_weather_{}_{}/{}".format(frame_id, 1, j), inputs[("color_aug", frame_id, 0)][j].data, self.epoch)
                if frame_id == "s" and self.opt.use_depth_hints:
                    try:
                        writer.add_image("depth_hints_mask{}/{}".format(scale, j), outputs["depth_hint_pixels/{}".format(scale)][j][None, ...], self.epoch)
                        writer.add_image("color_depth_hint{}/{}".format(scale, j), outputs[("color_depth_hint", frame_id, scale)][j][None, ...], self.epoch)
                    except KeyError:
                        pass
            scale = 0
            m = (outputs[('disp', scale)][j]).min()
            M = (outputs[('disp', scale)][j]).max()
            try:
                writer.add_image("disp_{}/{}".format(scale, j), (outputs[('disp', scale)][j] - m) / ((M - m) if m != M else 1e5), self.epoch)
            except KeyError:
                pass

    # IO函数
    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = self.log_path
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opts.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, folder_name):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, folder_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            if self.opt.use_multi_gpu:
                to_save = model.module.state_dict()
            else:
                to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("==>loading model from folder {}".format(self.opt.load_weights_folder))
        if self.opt.net_type == 'vit':
            self.opt.models_to_load = ['encoder', 'depth', 'pose_encoder', 'pose']
        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n), end=" ")
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            if self.opt.use_multi_gpu:
                model_dict = self.models[n].module.state_dict()
            else:
                model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location=self.device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            if self.opt.use_multi_gpu:
                self.models[n].module.load_state_dict(model_dict)
            else:
                self.models[n].load_state_dict(model_dict)
            print("√")

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights...", end=" ")
            optimizer_dict = torch.load(optimizer_load_path, map_location=self.device)
            self.model_optimizer.load_state_dict(optimizer_dict)
            print("√")
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
