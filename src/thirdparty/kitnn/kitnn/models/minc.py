import logging
from collections import OrderedDict

import numpy as np
from pydensecrf import densecrf
from skimage.color import rgb2lab
from torch import nn

from thirdparty.kitnn.kitnn.models.vgg import VGG16
from thirdparty.kitnn.kitnn.modules import SelectiveSequential, LRN
from thirdparty.kitnn.kitnn.utils import (SerializationMixin, load_module_npy, make_batch,
                         softmax2d)
from thirdparty.toolbox.toolbox.images import resize

logger = logging.getLogger(__name__)


SUBSTANCES = [
    'brick',
    'carpet',
    'ceramic',
    'fabric',
    'foliage',
    'food',
    'glass',
    'hair',
    'leather',
    'metal',
    'mirror',
    'other',
    'painted',
    'paper',
    'plastic',
    'polishedstone',
    'skin',
    'sky',
    'stone',
    'tile',
    'wallpaper',
    'water',
    'wood'
]
# chair and bed
REMAPPED_SUBSTANCES = [
    'fabric',
    'leather',
    'wood',
    'metal',
    'plastic',
    'background',
]
# table
# REMAPPED_SUBSTANCES = [
#     'fabric',
#     'stone',
#     'wood',
#     'metal',
#     'plastic',
#     'background',
# ]

# chair and bed
SUBST_MAPPING = OrderedDict([
    ('fabric', 'fabric'),
    ('carpet', 'fabric'),
    ('leather', 'leather'),
    ('wood', 'wood'),
    ('metal', 'metal'),
    ('plastic', 'plastic'),
])
# table
# SUBST_MAPPING = OrderedDict([
#     ('fabric', 'fabric'),
#     ('carpet', 'fabric'),
#     ('stone', 'stone'),
#     ('polishedstone', 'stone'),
#     ('tile', 'stone'),
#     ('wood', 'wood'),
#     ('metal', 'metal'),
#     ('plastic', 'plastic'),
# ])

RT2 = np.sqrt(2)


def compute_remapped_probs(probs: np.ndarray, fg_mask=None):
    fg_inds = [i for i, s in enumerate(REMAPPED_SUBSTANCES)
               if s != 'background']
    bg_idx = REMAPPED_SUBSTANCES.index('background')
    remapped = np.zeros((*probs.shape[:2], len(REMAPPED_SUBSTANCES)))
    for subst_old, subst_new in SUBST_MAPPING.items():
        old_idx = SUBSTANCES.index(subst_old)
        new_idx = REMAPPED_SUBSTANCES.index(subst_new)
        remapped[:, :, new_idx] += probs[:, :, old_idx]
    if fg_mask is None:
        remapped[remapped[:,:,:4].max(axis=2) < 0.2, bg_idx] = 1.0
    else:
        remapped[~fg_mask, bg_idx] = 2.0
        remapped[fg_mask, bg_idx] = -5
    remapped = softmax2d(remapped)
    return remapped


def preprocess_image(image):
    if image.max() > 1.0:
        logger.warning("Image has values larger than 1.0, this function"
                       " expects all images to be single between 0.0 and 1.0!")
    image = image.copy()
    processed = image.astype(np.float32) * 255.0
    processed = processed[:, :, [2, 1, 0]]
    processed[:, :, 0] -= 104.0
    processed[:, :, 1] -= 117.0
    processed[:, :, 2] -= 124.0
    return processed

def preprocess_images(images):
    if images.max() > 1.0:
        logger.warning("Image has values larger than 1.0, this function"
                       " expects all images to be single between 0.0 and 1.0!")
    images = images.copy()
    processeds = images.astype(np.float32) * 255.0
    processeds = processeds[:, :, :, [2, 1, 0]]
    processeds[:, :, :, 0] -= 104.0
    processeds[:, :, :, 1] -= 117.0
    processeds[:, :, :, 2] -= 124.0
    return processeds
        
    


def resize_image(image, scale=1.0, l_size=256, l_frac=0.233, order=2):
    small_dim_len = l_size / l_frac
    scale_mult = scale * small_dim_len / min(image.shape[:2])
    scale_shape = (int(image.shape[0] * scale_mult),
                   int(image.shape[1] * scale_mult))
    return resize(image, scale_shape, order=order)


def resize_images(images, scale=1.0, l_size=256, l_frac=0.233, order=2):
    small_dim_len = l_size / l_frac
    for i in range(images.shape[0]):
        image = np.squeeze(images[i])
        scale_mult = scale * small_dim_len / min(image.shape[:2])
        scale_shape = (int(image.shape[0] * scale_mult),
                       int(image.shape[1] * scale_mult))
        resized_image = resize(image, scale_shape, order=order)[np.newaxis,:]
        if i == 0:
            resized_images = resized_image
        else:
            resized_images = np.vstack((resized_images, resized_image))
    return resized_images


def combine_probs(prob_maps, image, remap=False, fg_mask=None):
    substances = REMAPPED_SUBSTANCES if remap else SUBSTANCES
    map_scale = 550 / min(image.shape[:2])
    map_sum = np.zeros((int(image.shape[0] * map_scale),
                        int(image.shape[1] * map_scale),
                        len(substances)))
    for prob_map in prob_maps:
        if remap:
            resized_fg_mask = resize(fg_mask, prob_map.shape, order=0)
            prob_map = compute_remapped_probs(
                prob_map, fg_mask=resized_fg_mask)
        prob_map = resize(prob_map, map_sum.shape[:2])
        map_sum += prob_map
    return map_sum / len(prob_maps)


def compute_probs_multiscale(
        image, mincnet, scales=list([RT2, 1.0, 1 / RT2]), use_cuda=True):
    prob_maps = []
    feat_dicts = []
    for scale in scales:
        image_scaled = resize_image(image, scale=scale)
        logger.info("\tProcessing scale={:.4}, shape={}"
                    .format(scale, image_scaled.shape))
        batch_arr = make_batch([image_scaled])
        if use_cuda:
            batch_arr = batch_arr.cuda()
        # batch = Variable(batch_arr, volatile=True)
        prob_map, sel_dict = mincnet(batch_arr, selection=['fc8-20', 'softmax'])
        prob_map_numpy = prob_map.cpu().data.numpy()[0].transpose((1, 2, 0))
        prob_maps.append(prob_map_numpy)
        feat_dicts.append(sel_dict)
    return prob_maps, feat_dicts


def compute_probs_multiscales(
        image, mincnet, scales=list([RT2, 1.0, 1 / RT2]), use_cuda=True):
    prob_maps = []
    feat_dicts = []
    for scale in scales:
        image_scaled = resize_image(image, scale=scale)
        logger.info("\tProcessing scale={:.4}, shape={}"
                    .format(scale, image_scaled.shape))
        batch_arr = make_batch([image_scaled])
        if use_cuda:
            batch_arr = batch_arr.cuda()
        # batch = Variable(batch_arr, volatile=True)
        prob_map, sel_dict = mincnet(batch_arr, selection=['fc8-20', 'softmax'])
        prob_map_numpy = prob_map.cpu().data.numpy()[0].transpose((1, 2, 0))
        prob_maps.append(prob_map_numpy)
        feat_dicts.append(sel_dict)
    return prob_maps, feat_dicts


def compute_probs_crf(
        image, prob_map, theta_p=0.1, theta_L=10.0, theta_ab=5.0):
    resized_im = np.clip(resize(image, prob_map.shape[:2], order=3), 0, 1)
    image_lab = rgb2lab(resized_im)

    p_y, p_x = np.mgrid[0:image_lab.shape[0], 0:image_lab.shape[1]]

    feats = np.zeros((5, *image_lab.shape[:2]), dtype=np.float32)
    d = min(image_lab.shape[:2])
    feats[0] = p_x / (theta_p * d)
    feats[1] = p_y / (theta_p * d)
    feats[2] = image_lab[:, :, 0] / theta_L
    feats[3] = image_lab[:, :, 1] / theta_ab
    feats[4] = image_lab[:, :, 2] / theta_ab
    crf = densecrf.DenseCRF2D(*prob_map.shape)
    unary = np.rollaxis(
        -np.log(prob_map), axis=-1).astype(dtype=np.float32, order='c')
    crf.setUnaryEnergy(np.reshape(unary, (prob_map.shape[-1], -1)))

    compat = 2*np.array((
        # f    l    w    m    p    b
        (0.0, 1.0, 1.0, 1.0, 1.0, 3.0),  # fabric
        (1.0, 0.0, 1.0, 1.0, 1.0, 3.0),  # leather
        (1.0, 1.0, 0.0, 1.0, 1.0, 3.0),  # wood
        (1.0, 1.0, 1.0, 0.0, 1.0, 3.0),  # metal
        (1.0, 1.0, 1.0, 1.0, 0.0, 3.0),  # plastic
        (1.5, 1.5, 1.5, 1.5, 1.5, 0.0),  # background
    ), dtype=np.float32)

    crf.addPairwiseEnergy(np.reshape(feats, (feats.shape[0], -1)),
                          compat=compat)

    Q = crf.inference(20)
    Q = np.array(Q).reshape((-1, *prob_map.shape[:2]))
    return np.rollaxis(Q, 0, 3)


class MincAlexNet(nn.Module, SerializationMixin):
    def __init__(self):
        super().__init__()
        self.features = AlexNet().features
        self.classifier = SelectiveSequential(OrderedDict([
            ('fc6', nn.Conv2d(256, 4096, kernel_size=6, stride=1, padding=3)),
            ('relu6', nn.ReLU(inplace=True)),
            ('fc7', nn.Conv2d(4096, 4096, kernel_size=1, stride=1)),
            ('relu7', nn.ReLU(inplace=True)),
            ('fc8-20', nn.Conv2d(4096, 23, kernel_size=1, stride=1)),
            ('softmax', nn.Softmax2d())
        ]))

    def forward(self, x):
        features = self.features(x, selection=['pool5'])[0]
        softmax = self.classifier(features, ['softmax'])[0]
        return softmax

    def load_npy(self, path):
        with open(path, 'rb') as f:
            data = np.load(f)[()]
        load_module_npy(self.features, data)
        load_module_npy(self.classifier, data)


class MincVGG(nn.Module, SerializationMixin):
    def __init__(self):
        super().__init__()
        self.features = VGG16().features
        self.classifier = SelectiveSequential(OrderedDict([
            ('fc6', nn.Conv2d(512, 4096, kernel_size=7, stride=1, padding=3)),
            ('relu6', nn.ReLU(True)),
            ('fc7', nn.Conv2d(4096, 4096, kernel_size=1, stride=1)),
            ('relu7', nn.ReLU(True)),
            ('fc8-20', nn.Conv2d(4096, 23, kernel_size=1, stride=1)),
            ('softmax', nn.Softmax2d())
        ]))

    def forward(self, x, selection=None):
        if selection is None:
            selection = ['softmax']
        feature_sel = {'pool5'}
        classifier_sel = set()
        for sel in selection:
            if sel in self.features.modules_dict:
                feature_sel.add(sel)
            elif sel in self.classifier.modules_dict:
                classifier_sel.add(sel)
            else:
                logger.warning('layer %s does not exist in modules', sel)
        x, features = self.features(x, selection=feature_sel)
        x, classifier = self.classifier(x, selection=classifier_sel)
        return x, {**features, **classifier}

    def load_npy(self, path):
        with open(path, 'rb') as f:
            data = np.load(f)[()]
        load_module_npy(self.features, data)
        load_module_npy(self.classifier, data)


class AlexNet(nn.Module, SerializationMixin):
    def __init__(self):
        super().__init__()
        self.features = SelectiveSequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)),
            ('relu1', nn.ReLU(inplace=True)),
            ('norm1', LRN(5, 0.0001, 0.75, 1)),
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2)),

            ('conv2', nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ('relu2', nn.ReLU(inplace=True)),
            ('norm2', LRN(5, 0.0001, 0.75, 1)),
            ('pool2', nn.MaxPool2d(kernel_size=3, stride=2)),

            ('conv3', nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ('relu3', nn.ReLU(inplace=True)),

            ('conv4', nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ('relu4', nn.ReLU(inplace=False)),

            ('conv5', nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ('relu5', nn.ReLU(inplace=True)),
            ('pool5', nn.MaxPool2d(kernel_size=3, stride=2)),
        ]))

    def forward(self, x, selection=list()):
        return self.features(x, selection)

    def load_npy(self, path):
        with open(path, 'rb') as f:
            data = np.load(f)[()]
        load_module_npy(self.features, data)
