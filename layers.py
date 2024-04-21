from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def depth_to_disp(depth, min_depth, max_depth):
    """将深度图转换回视差,
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth

    disp = 1 / (depth + 1e-5)
    p_disp = (disp - min_disp) / (max_disp - min_disp)
    p_disp[depth <= 0] = 0
    p_disp[p_disp <= 0] = 0

    return p_disp


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class WavConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, norm_layer=None, use_refl=False):
        super(WavConvBlock, self).__init__()

        if kernel_size == 3:
            self.conv = Conv3x3(in_channels, out_channels, use_refl=use_refl)#3*3的卷积，padding=1
        elif kernel_size == 1:
            self.conv = Conv1x1(in_channels, out_channels)
        else:
            raise NotImplementedError

        self.nonlin = nn.ELU(inplace=True)
        if norm_layer is not None:
            self.norm_layer = norm_layer(out_channels)
        else:
            self.norm_layer = nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm_layer(out)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class Conv1x1(nn.Module):
    """Conv1x1
    """
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv(x)
        return out
##############################################################################################################
class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords
##############################################################################################################



# class BackprojectDepth(nn.Module):
#     """Layer to transform a depth image into a point cloud
#     """
#
#     def __init__(self, height, width):
#         super(BackprojectDepth, self).__init__()
#
#         self.height = height
#         self.width = width
#
#         meshgrid = np.meshgrid(range(self.width), range(self.height), indexing="xy")
#
#         self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
#         """
#         mershgrid是个list，需要stack成一个array
#         生成2*H*W的网格，格式为
#         0 1 ... W-1
#         0 1 ... W-1
#         0 1 ... W-1
#         第一维度
#         0 0 ... 0
#         1 1 ... 1
#         ...
#         H-1 H-1 ... H-1
#         第二维度
#         第1维是x坐标,第2维是y坐标
#         """
#         self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
#             requires_grad=False).cuda()
#
#         self.ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
#             requires_grad=False).cuda()
#         """
#         1*1*（H*W）的全1矩阵
#         """
#         self.pix_coords = torch.unsqueeze(torch.stack(
#             [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
#         """
#         这里的view(-1)是将2*H*W的网格展开成2*H*W的一维向量
#         0 1 ... W-1 0 1 ... W-1 0 1 ... W-1和0 0 ... 0 1 1 ... 1 ... H-1 H-1 ... H-1
#         之后拼接成2*（H*W）的网格
#         [0 1 ... W-1 0 1 ... W-1 0 1 ... W-1
#         0 0 ... 0 1 1 ... 1 ... H-1 H-1 ... H-1]
#         增加第三维度
#         [[0 1 ... W-1 0 1 ... W-1 0 1 ... W-1
#         0 0 ... 0 1 1 ... 1 ... H-1 H-1 ... H-1]]
#         1*2*（H*W）
#         """
#         self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
#             requires_grad=False)
#         """
#         得到1*3*（H*W）的网格，第一个纬度是为了和深度图对应
#         depth 的形状通常是 (B, 1, H, W)。
#         结尾形如：
#         [[0 1 ... W-1 0 1 ... W-1 0 1 ... W-1
#         0 0 ... 0 1 1 ... 1 ... H-1 H-1 ... H-1]
#         [1 1 ... 1 1 1 ... 1 ... 1 1 ... 1]]
#         沿着第二个纬度取可以得到（x,y,1）
#         """
#
#     def forward(self, depth, inv_K):
#         B, _, H, W = depth.shape
#         cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
#         """
#         inv_K[:, :3, :3]是B*3*3的矩阵，self.pix_coords是1*3*（H*W）的网格，K[:, :3, :3]是相机内参:
#         fx 0 cx
#         0 fy cy
#         0 0  1
#         含义是
#         x=1/Z*(fx*X+cx*Z)，y=1/Z*(fy*Y+cy*Z)，1=1/Z*Z,原理可见论文笔记的第二部分
#         改为矩阵形式就是
#         [x,y,1]=1/Z*K*[X,Y,Z]
#         这里认为pix_coords的单体是(x,y,1)，得到pix_coords*inv_K[:, :3, :3]*Z=[X,Y,Z]
#         """
#         cam_points = depth.view(B, 1, H * W) * cam_points  # 这里是把B*H*W→B*1*（H*W）
#         cam_points = torch.cat([cam_points, self.ones.expand(B, -1, -1)], 1)  # 这里是把B*3*（H*W）→B*4*（H*W）
#
#         return cam_points
#
#
# class Project3D(nn.Module):
#     """Layer which projects 3D points into a camera with intrinsics K and at position T
#     """
#
#     def __init__(self, height, width, eps=1e-7):
#         super(Project3D, self).__init__()
#
#         self.height = height
#         self.width = width
#         self.eps = eps
#
#     def forward(self, points, K, T):
#         B, _, HW = points.shape
#
#         P = torch.matmul(K, T)[:, :3, :]
#         """
#         这里的K是相机内参，T是相机外参，P是投影矩阵
#         可以根据3D的像素变化得到为什么可以*T来进行平移或旋转，之后再乘以K就可以得到二维的坐标
#         结果是B*3*4的矩阵
#         这里T是B*4*4的矩阵，K是B*4*4的矩阵
#         T的单体是
#         [1 0 0 +-0.1
#         0 1 0 0
#         0 0 1 0
#         0 0 0 1]
#         说明是在x轴上平移了0.1或者-0.1，代表stereo的两个相机的距离，所以说0.1代表54cm的来源是这里
#         """
#         cam_points = torch.matmul(P, points)
#         pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
#         """
#         point是B*4*（H*W）的矩阵,P是B*3*4的矩阵，所以P*point是B*3*（H*W）的矩阵,3代表x,y,z
#         之后除以z就可以得到x/z,y/z,1,这里只保留了x/z,y/z
#         下面view的作用是把B*2*（H*W）的矩阵变成B*2*H*W的矩阵
#         之后permute的作用是把B*2*H*W的矩阵变成B*H*W*2的矩阵
#
#         """
#         pix_coords = pix_coords.view(-1, 2, self.height, self.width)
#         pix_coords = pix_coords.permute(0, 2, 3, 1)
#         """
#         下面进行归一化，把像素坐标变成[0,1]的坐标
#         最后-0.5*2是为了把坐标变成[-1,1]的坐标
#         其目的是为了与 grid_sample 函数的要求相匹配
#         """
#         pix_coords[..., 0] /= self.width - 1
#         pix_coords[..., 1] /= self.height - 1
#         pix_coords = (pix_coords - 0.5) * 2
#         return pix_coords


class HomographyWarp(nn.Module):
    """Layer to transform a depth image into a point cloud
    """

    def __init__(self, height, width):
        super(HomographyWarp, self).__init__()

        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing="xy")
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
            requires_grad=False).cuda()

        self.ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
            requires_grad=False).cuda()

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
            requires_grad=False)

    def forward(self, d, n, T, K, inv_K):
        """
        d: B, N
        n: B, N, 3
        """
        B, N = d.shape
        d = d.reshape(B * N, 1, 1)
        n = n.reshape(B * N, 1, 3)
        pix_coords_t = self.pix_coords.expand(B * N, -1, -1)
        R = T[:, :3, :3]
        t = T[:, :3, 3:4]
        Rtnd = R + torch.matmul(t, n) / d
        # print(K[0, :3, :3], inv_K[0, :3, :3])
        H_s2t = torch.matmul(K[:, :3, :3], torch.matmul(Rtnd, inv_K[:, :3, :3]))
        H_t2s = torch.inverse(H_s2t)
        pix_coords = torch.matmul(H_t2s, pix_coords_t)

        padding_mask = (torch.matmul(inv_K[:, :3, :3], pix_coords_t) * torch.matmul(R, n[:, 0, :, None])).sum(1) > 0.
        z = pix_coords[:, 2:3, :]
        padding_mask = padding_mask * (z[:, 0] > 1e-7)
        padding_mask = padding_mask.reshape(B, N, 1, self.height, self.width)
        z[z < 1e-7] = 1e-7
        pix_coords = pix_coords[:, :2, :] / z
        pix_coords = pix_coords.view(B * N, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords, padding_mask


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


def get_smooth_loss_disp(disp, img, gamma=1):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-gamma * grad_img_x)
    grad_disp_y *= torch.exp(-gamma * grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


def get_smooth_loss_probability(probability, disp_layered, img, gamma=1):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(probability[:, :, :, :-1] - probability[:, :, :, 1:]) * (disp_layered[:, :, :, :-1] + disp_layered[:, :, :, 1:]) / 2.
    grad_disp_x = grad_disp_x.sum(1, True)
    grad_disp_y = torch.abs(probability[:, :, :-1, :] - probability[:, :, 1:, :]) * (disp_layered[:, :, :-1, :] + disp_layered[:, :, 1:, :]) / 2.
    grad_disp_y = grad_disp_y.sum(1, True)

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-gamma * grad_img_x)
    grad_disp_y *= torch.exp(-gamma * grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super(Embedder, self).__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], dim=1)


def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 2,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    # embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embedder_obj  # , embedder_obj.out_dim


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


# Define VGG19 from FALNet
class Vgg19_pc(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19_pc, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        vgg_pretrained_features = nn.DataParallel(vgg_pretrained_features.cuda())

        # This has Vgg config E:
        # partial convolution paper uses up to pool3
        # [64,'r', 64,r, 'M', 128,'r', 128,r, 'M', 256,'r', 256,r, 256,r, 256,r, 'M', 512,'r', 512,r, 512,r, 512,r]
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        n_new = 0
        for x in range(5):  # pool1,
            self.slice1.add_module(str(n_new), vgg_pretrained_features.module[x])
            n_new += 1
        for x in range(5, 10):  # pool2
            self.slice2.add_module(str(n_new), vgg_pretrained_features.module[x])
            n_new += 1
        for x in range(10, 19):  # pool3
            self.slice3.add_module(str(n_new), vgg_pretrained_features.module[x])
            n_new += 1
        for x in range(19, 28):  # pool4
            self.slice4.add_module(str(n_new), vgg_pretrained_features.module[x])
            n_new += 1
        for param in self.parameters():
            param.requires_grad = requires_grad
        # norm as torch
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        # norm as FalNet
        # self.normalize = torchvision.transforms.Normalize(mean=[0.411, 0.432, 0.45],
        #                         std=[1, 1, 1])

    def forward(self, x, full=False):
        x = self.normalize(x)
        h_relu1_2 = self.slice1(x)
        h_relu2_2 = self.slice2(h_relu1_2)
        h_relu3_4 = self.slice3(h_relu2_2)
        if full:
            h_relu4_4 = self.slice4(h_relu3_4)
            return h_relu1_2, h_relu2_2, h_relu3_4, h_relu4_4
        else:
            return h_relu1_2, h_relu2_2, h_relu3_4


class Resnet18_pc(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, requires_grad=False):
        super(Resnet18_pc, self).__init__()

        self.encoder = torchvision.models.resnet18(pretrained=True)

        for param in self.parameters():
            param.requires_grad = requires_grad

        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def forward(self, x):
        self.features = []
        x = self.normalize(x)
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        # self.features.append(self.encoder.layer3(self.features[-1]))
        # self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features


def gaussian(error, sigma):
    return torch.exp(-0.5 * error ** 2 / sigma ** 2) / sigma / (2 * np.pi) ** 0.5


def laplacian(error, b):
    return 0.5 * torch.exp(-(torch.abs(error) / b)) / b


def distribution(error, sigma, dist="gaussian"):
    return gaussian(error, sigma) if dist == "gaussian" else \
        laplacian(error, sigma)


def bimodal_loss(error0, error1, sigma0, sigma1, w0, w1, dist="gaussian"):
    return - torch.log(w0 * distribution(error0, sigma0, dist) + \
                       w1 * distribution(error1, sigma1, dist))


def multimodal_loss(error, sigma, pi, dist='gaussian'):
    return - torch.log(torch.sum(pi * distribution(error, sigma, dist), dim=1, keepdim=True) + 1e-7)


def create_camera_plane(height, width):
    K = np.array([[0.58, 0, 0.5, 0],
                  [0, 1.92, 0.5, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
    K[0, :] *= width
    K[1, :] *= height
    K = torch.Tensor(K).cuda()
    K_inv = torch.inverse(K)
    meshgrid = np.meshgrid(range(width), range(height), indexing="xy")
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = nn.Parameter(torch.from_numpy(id_coords),
        requires_grad=False)

    ones = nn.Parameter(torch.ones(1, 1, height * width),
        requires_grad=False)

    pix_coords = torch.unsqueeze(torch.stack(
        [id_coords[0].view(-1), id_coords[1].view(-1)], 0), 0)
    pix_coords = nn.Parameter(torch.cat([pix_coords, ones], 1),
        requires_grad=False).cuda()

    cam_points = torch.matmul(K_inv[None, :3, :3], pix_coords)
    cam_points = cam_points.reshape(1, 3, height, width)
    return cam_points
