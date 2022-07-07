import logging
import torch
from torch.autograd import Variable
import numpy as np


logger = logging.getLogger(__name__)


xyz_from_rgb = Variable(torch.FloatTensor([
    [0.412453, 0.357580, 0.180423],
    [0.212671, 0.715160, 0.072169],
    [0.019334, 0.119193, 0.950227]
]).cuda(), requires_grad=False)
rgb_from_xyz = Variable(torch.inverse(xyz_from_rgb.data).cuda(),
                        requires_grad=False)

lab_from_fxfyfz = Variable(torch.FloatTensor([
    [0.0, 500.0, 0.0],
    [116.0, -500.0, 200.0],
    [0.0, 0.0, -200.0],
]).t().cuda(), requires_grad=False)
fxfyfz_from_lab = Variable(torch.FloatTensor([
    [1/116.0, 1/116.0, 1/116.0],
    [1/500.0, 0.0, 0.0],
    [0.0, 0.0, -1/200.0],
]).t().cuda(), requires_grad=False)
d65_norm = Variable(torch.FloatTensor(
    [0.950456, 1.0, 1.088754]).cuda(),
    requires_grad=False)
lab_conv_bias = Variable(torch.FloatTensor([-16, 0, 0]).cuda(),
                         requires_grad=False)


lab_bias = Variable(torch.FloatTensor([0, 128, 128]).cuda(),
                    requires_grad=False)
lab_scale = Variable(torch.FloatTensor([100, 256, 256]).cuda(),
                     requires_grad=False)


def image_to_tensor(image):
    return Variable(torch.from_numpy(image.astype(dtype=np.float32)) \
        .permute(2, 0, 1) \
        .contiguous() \
        .unsqueeze(0).cuda(), requires_grad=False)


def tensor_to_image(tensor):
    if not torch.is_tensor(tensor):
        tensor = tensor.data
    return tensor.squeeze().cpu().permute(1, 2, 0).numpy()


def flatten_batch(batch):
    return batch.view(batch.size(0), batch.size(1), -1)


def srgb_to_rgb(srgb):
    srgb_pixels = flatten_batch(srgb)
    linear_mask = (srgb_pixels <= 0.04045).float()
    exponential_mask = (srgb_pixels > 0.04045).float()
    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
    return rgb_pixels.view(*srgb.size())


def rgb_to_srgb(rgb):
    rgb_pixels = flatten_batch(rgb)
    linear_mask = (rgb_pixels <= 0.0031308).float()
    exponential_mask = (rgb_pixels > 0.0031308).float()
    srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask
    return srgb_pixels.view(*rgb.size())


def rgb_to_xyz(rgb):
    rgb_pixels = flatten_batch(rgb)
    rgb_pixels = rgb_pixels.clamp(0, 1)
    xyz_pixels = torch.bmm(
        xyz_from_rgb.unsqueeze(0).expand(rgb_pixels.size(0), 3, 3),
        rgb_pixels)
    return xyz_pixels.view(*rgb.size())


def xyz_to_rgb(xyz):
    xyz_pixels = flatten_batch(xyz)
    rgb_pixels = torch.bmm(
        rgb_from_xyz.unsqueeze(0).expand(xyz_pixels.size(0), 3, 3),
        xyz_pixels)
    return rgb_pixels.view(*xyz.size())


def xyz_to_lab(xyz, normalized=False):
    xyz_pixels = flatten_batch(xyz)
    xyz_normalized_pixels = \
        xyz_pixels / d65_norm.view(1, 3, 1).expand(*xyz_pixels.size())

    epsilon = 6.0 / 29.0
    linear_mask = (xyz_normalized_pixels <= (epsilon ** 3)).float()
    exponential_mask = (xyz_normalized_pixels > (epsilon ** 3)).float()
    fxfyfz_pixels = (
        linear_mask * (xyz_normalized_pixels / (3 * epsilon ** 2) + 4 / 29)
        + exponential_mask * (xyz_normalized_pixels.pow(1/3)))

    lab_pixels = torch.baddbmm(
        lab_conv_bias.view(1, 3, 1).expand(*fxfyfz_pixels.size()),
        lab_from_fxfyfz.unsqueeze(0).expand(xyz_pixels.size(0), 3, 3),
        fxfyfz_pixels)

    if normalized:
        lab_pixels = normalize_lab(lab_pixels)

    return lab_pixels.view(xyz.size())


def lab_to_xyz(lab, normalized=False):
    lab_pixels = flatten_batch(lab)

    if normalized:
        lab_pixels = denormalize_lab(lab_pixels)

    lab_pixels = lab_pixels - lab_conv_bias.view(1, 3, 1).expand(*lab_pixels.size())
    fxfyfz_pixels = torch.bmm(
        fxfyfz_from_lab.unsqueeze(0).expand(lab_pixels.size(0), 3, 3),
        lab_pixels)

    # convert to xyz
    epsilon = 6.0 / 29.0
    linear_mask = (fxfyfz_pixels <= epsilon).float()
    exponential_mask = (fxfyfz_pixels > epsilon).float()
    xyz_pixels = (
        linear_mask * (3 * epsilon ** 2 * (fxfyfz_pixels - 4 / 29))
        + exponential_mask * (fxfyfz_pixels ** 3))

    xyz_pixels = \
        xyz_pixels * d65_norm.view(1, 3, 1).expand(*xyz_pixels.size())
    return xyz_pixels.view(*lab.size())


def rgb_to_lab(rgb, normalized=False):
    xyz = rgb_to_xyz(rgb)
    lab = xyz_to_lab(xyz, normalized=normalized)
    return lab


def lab_to_rgb(rgb, normalized=False):
    xyz = lab_to_xyz(rgb, normalized=normalized)
    rgb = xyz_to_rgb(xyz)
    return rgb


def normalize_lab(lab):
    lab = flatten_batch(lab)
    lab = ((lab + lab_bias.view(1, 3, 1).expand(*lab.size()))
            / lab_scale.view(1, 3, 1).expand(*lab.size()))
    return lab.view(*lab.size())


def denormalize_lab(lab):
    lab = flatten_batch(lab)
    lab = ((lab * lab_scale.view(1, 3, 1).expand(*lab.size())
            - lab_bias.view(1, 3, 1).expand(*lab.size())))
    return lab.view(*lab.size())
