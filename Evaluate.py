# Code for evaluating the monocular depth estimation model, enhanced from the original Monodepth2 codebase.
# Enhanced details:
# The code can be used to evaluate the model on the KITTI, DrivingStereo, and CADC datasets. You can easily add your model at the func: create_model
# The code supports evaluation of the model with different baseline models, including Monodepth2(ICCV2019),WaveletDepth(CVPR2021), MonoViT(3DV2022), and PlaneDepth(CVPR2023). You can eaily add your model at the func: create_dataset
# The code use cuda to further speed up the evaluation process.
# Author: Jiyuan Wang
# Created: 2023-6-1
# Origin used for paper: https://arxiv.org/abs/2310.05556v2
# Hope you can cite our paper if you use the code for your research.
from __future__ import absolute_import, division, print_function
import os, sys
from layers import depth_to_disp, disp_to_depth
from torchvision.utils import save_image
import time
from torch.utils.data import DataLoader
import warnings
from tqdm import tqdm, trange
from options import *
import datasets
import networks
import cv2
from utils import *
from PIL import Image
# from my_utils import *

sys.path.append('../')
warnings.filterwarnings("ignore")
cv2.setNumThreads(0)
splits_dir = os.path.join(os.path.dirname(__file__), "./splits")
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity_torch(l_disp, r_disp):
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = torch.meshgrid(torch.linspace(0, 1, w, device=l_disp.device), torch.linspace(0, 1, h, device=l_disp.device))
    l = torch.transpose(l, 0, 1)
    l_mask = (1.0 - torch.clamp(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = torch.flip(l_mask, dims=[2])
    return m_disp  # r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return m_disp  # r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def create_dataset(opt):
    if opt.eval_split != 'stereo':
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        filenames = filenames[:40] if opt.debug else filenames
    else:
        opt.data_path = opt.data_path.replace("kitti", "drivingstereo")  # or you can set your drivingstereo dataset path here
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "rain.txt"))  # or you can change rain.txt to fog.txt to test the fog subsplit.
        filenames = filenames[:10] if opt.debug else filenames
        dataset = datasets.DrivingStereoDataset(opt, filenames, is_train=False)
        dataset.specify_data(class_map[1][0])

    if opt.eval_split == 'cadc':
        opt.data_path = opt.data_path.replace("kitti", "cadcd")
        dataset = datasets.CADCDataset(opt, filenames, is_train=False)
    elif 'eigen' in opt.eval_split:
        dataset = datasets.KITTIRAWDataset(opt, filenames, is_train=False, )
        dataset.specify_data(class_map[0][0])
    opt.num_workers = 0 if opt.debug else opt.num_workers
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers,
        pin_memory=True, drop_last=False)
    return dataloader, dataset, filenames


def create_model(opt):
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    encoder = None
    if opt.net_type == "plane":
        encoder_dict = torch.load(encoder_path)
        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc,
            opt.disp_levels,
            opt.disp_min,
            opt.disp_max,
            opt.num_ep,
            pe_type=opt.pe_type,
            use_denseaspp=opt.use_denseaspp,
            xz_levels=opt.xz_levels,
            yz_levels=opt.yz_levels,
            use_mixture_loss=opt.use_mixture_loss,
            render_probability=opt.render_probability,
            plane_residual=opt.plane_residual)
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))
    elif opt.net_type == "vit":
        encoder_dict = torch.load(encoder_path, map_location='cuda:0')
        encoder = networks.mpvit_small()
        encoder.num_ch_enc = [64, 128, 216, 288, 288]
        depth_decoder = networks.HR_DepthDecoder()
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path, map_location='cuda:0'))
    elif opt.net_type == "vithr":
        depth_dict = torch.load(decoder_path)
        new_dict = {}
        for k, v in depth_dict.items():
            name = k[7:]
            new_dict[name] = v
        depth_decoder = networks.DeepNet('mpvitnet')
        depth_decoder.load_state_dict({k: v for k, v in new_dict.items() if k in depth_decoder.state_dict()})
    elif opt.net_type == "wav":
        encoder = networks.ResnetEncoder(opt.num_layers, False)
        encoder_dict = torch.load(encoder_path, map_location="cuda:0")
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

        depth_decoder = networks.DepthWaveProgressiveDecoder(encoder.num_ch_enc, scales=range(4), log_wavlet=False)
        depth_decoder.load_state_dict(torch.load(decoder_path, map_location="cuda:0"), strict=False)
    elif opt.net_type == "mdp2":
        encoder = networks.ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location="cuda:0")
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)

        depth_decoder = networks.Monov2Decoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
        loaded_dict = torch.load(decoder_path, map_location="cuda:0")
        depth_decoder.load_state_dict(loaded_dict)
    try:
        encoder.cuda()
        encoder.eval()
    except:
        print("Vit for HR with no encoder")
    depth_decoder.cuda()
    depth_decoder.eval()
    return encoder, depth_decoder


def inference(opt, dataset, dataloader, encoder, depth_decoder):
    before = time.time()
    if opt.test_with_torch:
        if opt.post_process:
            probabilities_max = torch.zeros(2 * len(dataset), device='cuda')
        else:
            probabilities_max = torch.zeros(len(dataset), device='cuda')
        pred_disps = torch.zeros((len(dataset), opt.height, opt.width), device='cuda')
    else:
        pred_disps, probabilities_max = [], []
    grid = meshgrid(torch.linspace(-1, 1, opt.width), torch.linspace(-1, 1, opt.height), indexing="xy")
    grid = torch.stack(grid, dim=0)
    i, start_idx, names = 0, 0, []  # name is use for drivingstereo dataset for it use the 'image with same name' to save the gt depth, ignore it if you don't use this dataset.
    with torch.no_grad():
        for data in tqdm(dataloader):

            input_color = data[("color", "l")].cuda() if opt.net_type == "plane" else data[("color", 0, 0)].cuda()
            end_idx = start_idx + input_color.shape[0]
            if opt.eval_split == 'stereo':
                names.append(data["name"])
            if opt.post_process:
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
            grids = grid[None, ...].expand(input_color.shape[0], -1, -1, -1).cuda()

            if opt.net_type == "plane":
                output = depth_decoder(encoder(input_color), grids)
                pred_disp = output["disp"][:, 0] if opt.test_with_torch else output["disp"][:, 0].cpu().numpy()
            else:
                output = depth_decoder(input_color) if "hr" in opt.net_type else depth_decoder(encoder(input_color))
                pred_disp, _ = disp_to_depth(output[("disp", 0)], 0.1, 80)
                pred_disp = pred_disp[:, 0]

            if opt.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity_torch(pred_disp[:N], torch.flip(pred_disp[N:], dims=[2])) if opt.test_with_torch else batch_post_process_disparity(pred_disp[:N],
                    pred_disp[N:, :, ::-1])
            if opt.test_with_torch:
                if opt.net_type == "plane":
                    max_prob, _ = torch.max(output["probability"], axis=1)
                    if opt.post_process:
                        probabilities_max[2 * start_idx:2 * end_idx] = torch.mean(max_prob, dim=[1, 2])
                    else:
                        probabilities_max[start_idx:end_idx] = torch.mean(max_prob, dim=[1, 2])
                pred_disps[start_idx:end_idx] = pred_disp
            else:
                pred_disps.append(pred_disp)
                if opt.net_type == "plane":
                    probabilities_max.append(np.max(output["probability"].cpu().numpy(), axis=1).mean(-1).mean(-1))

            start_idx = end_idx
            i += opt.batch_size

    if opt.test_with_torch:
        pred_disps = pred_disps.cpu().numpy()
        probabilities_max = probabilities_max.cpu().numpy()
    pred_disps = np.concatenate(pred_disps) if not opt.test_with_torch else pred_disps
    probabilities_max = np.concatenate(probabilities_max) if not opt.test_with_torch else probabilities_max
    return pred_disps, probabilities_max, names


def evaluate(opt, pred_disps, probabilities_max, names, gt_depths):
    errors = []
    ratios = []
    for i in range(pred_disps.shape[0]):
        bs = opt.batch_size
        # load gt
        if opt.eval_split == 'stereo':
            depth_path = os.path.join(opt.data_path, names[i // bs][i % bs]).replace("left-image", "depth-map")
            if opt.debug >= 3:
                print(depth_path)
            depth_png = np.array(Image.open(depth_path), dtype=int)
            # make sure we have a proper 16bit depth map here.. not 8bit!
            assert (np.max(depth_png) > 255)
            gt_depth = depth_png.astype(np.float32) / 256
            # gt_depth = gt_depth[250:800, :]
        elif opt.eval_split == 'cadc':
            gt_depth = gt_depths[i][234:774, 0:1280]
        else:
            gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 0.1 * 0.58 * opt.width / (pred_disp) if opt.net_type == "plane" else 1 / pred_disp

        if "eigen" in opt.eval_split:
            gt_depth[gt_depth < MIN_DEPTH] = MIN_DEPTH
            gt_depth[gt_depth > MAX_DEPTH] = MAX_DEPTH
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(gt_depth.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        err = compute_errors(gt_depth, pred_depth)

        # print_errors(np.array(err), i, type='markdown')  # Annotate this line for the final result
        errors.append(err)
    return errors, ratios


def print_errors(errors, name, type='latex'):
    if type == 'latex':
        print(("{:>20}").format(name), end='')
        print(("&{:10.3f}" * 7).format(*errors.tolist()) + "\\\\")
    elif type == 'markdown':
        print(("|{:>20}").format(name), end='')
        print(("|{:10.3f}" * 7).format(*errors.tolist()) + "|")


def print_title(name):
    print(("{:>20}").format(name), end='')
    print(("&{:>10}" * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3") + "\\\\")


def evaluate_all(opt):
    """Evaluates a pretrained model using a specified test set     """
    # region pretreatment
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.cuda_devices)
    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)
    print("-> Loading weights from {}".format(opt.load_weights_folder))

    dataloader, dataset, filenames = create_dataset(opt)
    encoder, depth_decoder = create_model(opt)

    gt_depths = None
    if opt.eval_split != 'stereo':
        gt_depths = np.load(os.path.join(opt.data_path, "gt_depths.npy"), allow_pickle=True)
    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    print("-> Computing predictions with size {}x{}".format(opt.width, opt.height))
    # endregion

    print_title(opt.eval_split)
    if opt.test_with_weather:
        error_all = []
        load_val_mode = ['rgb/data', 'raingan/data', 'fog/150m', 'snowgan/data', 'mix_rain/50mm', 'mix_snow/data', 'fog/75m']
        for mode in load_val_mode:
            dataset.specify_data(mode)
            pred_disps, probabilities_max, names = inference(opt, dataset, dataloader, encoder, depth_decoder)
            errors, ratios = evaluate(opt, pred_disps, probabilities_max, names, gt_depths)
            mean_errors = np.array(errors).mean(0)
            print_errors(mean_errors, mode)
            error_all.append(mean_errors)
        mean_errors = np.array(error_all).mean(0)
        print_errors(mean_errors, 'average')
    else:
        pred_disps, probabilities_max, names = inference(opt, dataset, dataloader, encoder, depth_decoder)
        errors, ratios = evaluate(opt, pred_disps, probabilities_max, names, gt_depths)

        if not opt.disable_median_scaling:
            ratios = np.array(ratios)
            med = np.median(ratios)
            print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

        mean_errors = np.array(errors).mean(0)
        print_errors(mean_errors, opt.eval_split)


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate_all(options.parse())
