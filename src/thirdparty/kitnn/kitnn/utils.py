import math
import logging
import numpy as np
import torch
from scipy.misc import imread, imresize
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from thirdparty.kitnn.kitnn.functions import StablePow

logger = logging.getLogger(__name__)


IMAGENET_MEAN = np.array([0.40760392, 0.45795686, 0.48501961])
SOBEL_KERNEL_X = Variable(torch.from_numpy(
    np.array([(1, 0, -1),
              (2, 0, -2),
              (1, 0, -1)]).astype(dtype=np.float32)),
                          requires_grad=False)
SOBEL_KERNEL_X = SOBEL_KERNEL_X.view(1, 1, *SOBEL_KERNEL_X.size())
SOBEL_KERNEL_Y = Variable(torch.from_numpy(
    np.array([(1, 2, 1),
              (0, 0, 0),
              (-1, -2, -1)]).astype(dtype=np.float32)),
                          requires_grad=False)
SOBEL_KERNEL_Y = SOBEL_KERNEL_Y.view(1, 1, *SOBEL_KERNEL_Y.size())


class SerializationMixin:
    def load_npy(self, path):
        raise NotImplementedError()

    def save_pth(self, path):
        with open(path, 'wb') as f:
            torch.save(self, f)


def make_batch(images, flatten=False):
    if len(images[0].shape) == 2:
        images = [i[:, :, None] for i in images]
    batch = np.stack(images, axis=3) \
        .transpose((3, 2, 0, 1)) \
        .astype(dtype=np.float32)
    batch = torch.from_numpy(batch).contiguous()
    if flatten:
        batch.resize_(*batch.size()[:2], batch.size(2) * batch.size(3))
    return batch


def load_module_npy(module, data):
    for name, child in module._modules.items():
        if name in data:
            logger.info("Loading {} => {}".format(name, child))
            weight_shape = tuple(child.weight.size())
            weights = data[name]['weights']
            if weight_shape != weights.shape:
                logger.info("\tReshaping weight {} => {}"
                      .format(weights.shape, weight_shape))
                weights = weights.reshape(weight_shape)
            weights = torch.from_numpy(weights)
            bias = data[name]['biases']
            bias = torch.from_numpy(bias)
            child.weight.data.copy_(weights)
            child.bias.data.copy_(bias)


def to_imagenet(image):
    if image.dtype == np.uint8:
        image = image.astype(np.float32)
        image /= 255.0
    image = image.astype(np.float32)
    image[:, :, :3] = image[:, :, [2, 1, 0]]
    image[:, :, :3] -= IMAGENET_MEAN[None, None, :]
    return image


def from_imagenet(image):
    image = image.copy()[:, :, :3]
    image[:, :, :3] += IMAGENET_MEAN[None, None, :]
    image = image[:, :, [2, 1, 0]]
    return image


def softmax2d(x):
    e_x = np.exp(x - np.max(x, axis=-1)[:, :, None])
    return e_x / e_x.sum(axis=-1)[:, :, None]


def batch_to_images(batch, dtype=None):
    batch = batch.cpu()
    if not torch.is_tensor(batch):
        batch = batch.data
    if len(batch.size()) == 4:
        array = batch.numpy().reshape(
            batch.size(0), batch.size(1), batch.size(-2), batch.size(-1))\
            .transpose((0, 2, 3, 1))
    else:
        array = batch.numpy().reshape(
            batch.size(0), batch.size(-2), batch.size(-1)).transpose((0, 1, 2))
    if dtype is not None:
        return array.astype(dtype=dtype)
    return array


def gradient_image(batch):
    grady = F.conv2d(batch, SOBEL_KERNEL_Y.cuda())
    gradx = F.conv2d(batch, SOBEL_KERNEL_X.cuda())
    return grady, gradx


def normalize_batch(batch):
    if torch.is_tensor(batch):
        mean = batch.mean()
        std = batch.std()
    else:
        mean = batch.mean().view(*(1 for _ in batch.size())).expand(batch.size())
        std = batch.std().view(*(1 for _ in batch.size())).expand(batch.size())
    batch = (batch - mean) / std
    return batch


def batch_frobenius_norm(batch):
    batch = batch.view(batch.size(0), batch.size(1), -1)
    return (batch ** 2).sum(dim=2).squeeze().sqrt()


def rotation_tensor(theta, phi, psi, n_comps):
    rot_x = Variable(torch.zeros(n_comps, 3, 3).cuda(), requires_grad=False)
    rot_y = Variable(torch.zeros(n_comps, 3, 3).cuda(), requires_grad=False)
    rot_z = Variable(torch.zeros(n_comps, 3, 3).cuda(), requires_grad=False)
    rot_x[:, 0, 0] = 1
    rot_x[:, 0, 1] = 0
    rot_x[:, 0, 2] = 0
    rot_x[:, 1, 0] = 0
    rot_x[:, 1, 1] = theta.cos()
    rot_x[:, 1, 2] = theta.sin()
    rot_x[:, 2, 0] = 0
    rot_x[:, 2, 1] = -theta.sin()
    rot_x[:, 2, 2] = theta.cos()

    rot_y[:, 0, 0] = phi.cos()
    rot_y[:, 0, 1] = 0
    rot_y[:, 0, 2] = -phi.sin()
    rot_y[:, 1, 0] = 0
    rot_y[:, 1, 1] = 1
    rot_y[:, 1, 2] = 0
    rot_y[:, 2, 0] = phi.sin()
    rot_y[:, 2, 1] = 0
    rot_y[:, 2, 2] = phi.cos()

    rot_z[:, 0, 0] = psi.cos()
    rot_z[:, 0, 1] = -psi.sin()
    rot_z[:, 0, 2] = 0
    rot_z[:, 1, 0] = psi.sin()
    rot_z[:, 1, 1] = psi.cos()
    rot_z[:, 1, 2] = 0
    rot_z[:, 2, 0] = 0
    rot_z[:, 2, 1] = 0
    rot_z[:, 2, 2] = 1
    return torch.bmm(rot_z, torch.bmm(rot_y, rot_x))
