import logging
import random
from typing import List, Tuple

import numpy as np
from skimage.transform import resize
from scipy.ndimage import zoom

from toolbox import images
from toolbox.images import crop, mask_bbox
from .poisson_disk import sample_poisson_uniform

logger = logging.getLogger(__name__)


class PatchType:
    S2F_MASKED_BLACK = 'cropped_scaled_to_fit'
    S2F_MASKED_WHITE = 'cropped_scaled_to_fit_white'
    S2F = 'scaled_to_fit'
    RANDOM = 'random2'


def sample_poisson_mask(mask, r, k):
    ymin, ymax, xmin, xmax = mask_bbox(mask)
    height = ymax - ymin
    width = xmax - xmin
    points = np.array(sample_poisson_uniform(height, width, r, k,
                                             mask[ymin:ymax, xmin:xmax]))
    points[:, 0] += ymin
    points[:, 1] += xmin
    points = np.floor(points).astype(int)
    return points


def generate_dense_bboxes(
        mask: np.ndarray,
        scale=0.23,
        min_dist=0.091):
    mask_height, mask_width = mask.shape
    min_length = min(mask_height, mask_width)
    patch_sample_size = scale * min_length
    centers = sample_poisson_mask(mask, min_length * min_dist, 1000)
    half = int(patch_sample_size / 2)
    bboxes = []
    for center in centers:
        ycent, xcent = center
        bbox = (ycent - half,
                ycent + half + 1,
                xcent - half,
                xcent + half + 1)
        if (bbox[0] >= 0 and bbox[1] < mask_height
            and bbox[2] >= 0 and bbox[3] < mask_width):
            bboxes.append(bbox)
    print('bboxes={} centers={}, mask_size={}, min_dist={}'.format(
        len(bboxes), len(centers), mask.shape, min_length * min_dist))
    return bboxes


def random_crops(image, patch_size, num_crops):
    border_mask = np.ones(image.shape[:2], dtype=bool)
    left = patch_size/2
    right = image.shape[1] - patch_size/2
    top = patch_size/2
    bottom = image.shape[0] - patch_size/2
    border_mask[:, :left] = False
    border_mask[:, right:] = False
    border_mask[:top, :] = False
    border_mask[bottom:, :] = False

    yinds, xinds = np.where(border_mask)

    bboxes = []
    for i in range(num_crops):
        point_idx = np.random.randint(0, len(yinds))
        ycent, xcent = yinds[point_idx], xinds[point_idx]
        half = int(patch_size / 2)

        # Just squash the patch if it's out of bounds.
        bbox = (ycent - half,
                ycent + half + 1,
                xcent - half,
                xcent + half + 1)
        bboxes.append(bbox)

    return bboxes_to_patches(image, bboxes, patch_size)


def generate_random_bboxes(mask: np.ndarray, scale_range=(1.0, 1.0),
                           num_patches=5, fixed_size=None):
    """
    Generates random bounding boxes at random scales with centroid within the
    mask.
    :param mask: The contrained area for the centroid of the patch.
    :param min_scale: The min scale (multiple of the minimum length of the
                      input mask) of the sampling.
    :param max_scale: The max scale (multiple of the minimum length of the
                      input mask) of the sampling.
    :param num_patches: Number of patches to generate.
    :return: Bounding boxes.
    """
    mask_height, mask_width = mask.shape[:2]
    min_length = min(mask_height, mask_width)

    yinds, xinds = np.where(mask)

    patch_bboxes = []
    patch_scales = []
    tries = 0
    while len(patch_bboxes) < num_patches:
        scale = random.uniform(*scale_range)
        patch_scales.append(scale)
        patch_size = scale * fixed_size if fixed_size else int(scale * min_length)
        point_idx = np.random.randint(0, len(yinds))
        ycent, xcent = yinds[point_idx], xinds[point_idx]
        half = int(patch_size / 2)

        # Just squash the patch if it's out of bounds.
        if (ycent - half < 0 or ycent + half > mask.shape[0] or
            xcent - half < 0 or xcent + half > mask.shape[1]):
            if tries < 100:
                tries += 1
                continue

        bbox = (max(ycent - half, 0),
                min(ycent + half + 1, mask.shape[0]),
                max(xcent - half, 0),
                min(xcent + half + 1, mask.shape[1]))
        patch_bboxes.append(bbox)

    return patch_bboxes, patch_scales


def bboxes_to_patches(im: np.ndarray,
                      bboxes: List[Tuple[int, int, int, int]],
                      patch_size: int, use_pil=False):
    """
    Converts bounding boxes to actual patches. Patches are all resized to the
    patch size regardless of the original bounding box size.
    :param im: To crop patch from.
    :param bboxes: Boxes defining the patch.
    :param patch_size: Patch size to return.
    :return: Image patches.
    """
    patches = []
    for bbox in bboxes:
        cropped = crop(im, bbox)
        if cropped.shape[0] != patch_size or cropped.shape[1] != patch_size:
            scale = [patch_size/cropped.shape[0], patch_size/cropped.shape[1]]
            if len(im.shape) == 3:
                scale.append(1.0)
            if use_pil:
                cropped = resize(cropped, (patch_size, patch_size)) \
                              .astype(dtype=np.float32)
            else:
                cropped = zoom(cropped, scale, im.dtype, order=1)
        patches.append(cropped)
    return patches


def compute_mask_tight_patch(im: np.ndarray,
                             mask: np.ndarray,
                             patch_size: int):
    """
    Computes a patch which contains all the pixels active in the mask scaled to
    the patch size.
    :param im:
    :param mask:
    :param patch_size:
    :return:
    """
    bbox = images.compute_mask_bbox(mask)
    cropped = images.crop(im, bbox)
    resized = imresize(cropped, (patch_size, patch_size, cropped.shape[2]))
    return resized


def compute_minmax_thickness(mask):
    max_width = 0
    max_height = 0
    for row_id in range(mask.shape[0]):
        row = mask[row_id, :]
        split_locs = np.where(np.diff(row) != 0)[0] + 1
        for segment in (np.split(row, split_locs)):
            if segment[0] != 0:
                max_width = max(max_width, len(segment))
    for col_id in range(mask.shape[1]):
        col = mask[:, col_id]
        split_locs = np.where(np.diff(col) != 0)[0] + 1
        for segment in (np.split(col, split_locs)):
            if segment[0] != 0:
                max_height = max(max_height, len(segment))

    return min(max_width, max_height), max(max_width, max_height)
