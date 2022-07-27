import logging

import numpy as np
from scipy.ndimage.filters import gaussian_filter

from skimage.color import rgb2lab
import skimage.io

import sys
sys.path.append("./src/thirdparty/toolbox/toolbox/")
import caching

logger = logging.getLogger(__name__)


def visualize_color(colors, size=50):
    n_colors = colors.shape[0]
    vis = np.zeros((size, n_colors*size, 3))
    for i in range(n_colors):
        vis[:, i*size:i*size+size] = colors[i]
    return vis


def normalize_lab(lab_values):
    return (lab_values - (50, 0, 0)) / (50, 128, 128)


def denormalize_lab(norm_lab_values):
    return np.array(norm_lab_values) * (50, 128, 128) + (50, 0, 0)


def lab_rgb_gamut_bin_mask(num_bins=(10, 10, 10)):
    """
    Computes a mask of the NxMxK CIE LAB colorspace histogram that can be
    represented by the standard RGB (sRGB) color gamut.

    :param num_bins:
    :return: 3-dimensional mask.
    """
    bin_edges_L = np.linspace(0, 100, num_bins[0] + 1, endpoint=True)
    bin_edges_a = np.linspace(-90, 100, num_bins[1] + 1, endpoint=True)
    bin_edges_b = np.linspace(-110, 100, num_bins[2] + 1, endpoint=True)
    edges = (bin_edges_L, bin_edges_a, bin_edges_b)

    cache_name = (f'lab_rgb_gamut_bin_mask'
                  f'_{num_bins[0]}_{num_bins[1]}_{num_bins[2]}.png')
    cache_path = caching.get_path(cache_name)

    if caching.exists(cache_name):
        valid_bin_mask = (skimage.io.imread(cache_path)
                          .reshape(num_bins)
                          .astype(bool))
    else:
        print(f'Computing {cache_name}')

        rgb_gamut = np.mgrid[:255, :255, :255].reshape(3, 1, -1).transpose(
            (2, 1, 0)).astype(np.uint8)
        lab_rgb_gamut = rgb2lab(rgb_gamut)

        lab_rgb_gamut_hist, lab_rgb_gamut_hist_edges = np.histogramdd(
            lab_rgb_gamut.reshape(-1, 3),
            (bin_edges_L, bin_edges_a, bin_edges_b))

        valid_bin_mask = lab_rgb_gamut_hist > 0

        print(f'Saving {cache_name}')
        skimage.io.imsave(cache_path,
                          (valid_bin_mask.astype(np.uint8) * 255).reshape(-1, 1))

    return valid_bin_mask, edges


def compute_lab_histogram(image_rgb, num_bins, sigma=0.5):
    if isinstance(image_rgb, tuple):
        image_rgb = np.array(image_rgb, dtype=np.uint8).reshape(1, 1, 3)

    image_lab = skimage.color.rgb2lab(image_rgb)

    valid_bin_mask, bin_edges = lab_rgb_gamut_bin_mask(num_bins)
    hist, hist_edges = np.histogramdd(image_lab.reshape(-1, 3), bin_edges)
    hist = gaussian_filter(hist, sigma=sigma)

    bin_dist = hist[valid_bin_mask]
    if bin_dist.sum() > 0:
        bin_dist /= bin_dist.sum()
    return bin_dist


def compute_rgb_histogram(image_rgb, num_bins, sigma=0.5):
    if isinstance(image_rgb, tuple):
        image_rgb = np.array(image_rgb).reshape(1, 1, 3)

    bin_edges = (
        np.linspace(0, 255, num_bins[0] + 1, endpoint=True),
        np.linspace(0, 255, num_bins[1] + 1, endpoint=True),
        np.linspace(0, 255, num_bins[2] + 1, endpoint=True),
    )

    hist, hist_edges = np.histogramdd(image_rgb.reshape(-1, 3), bin_edges)
    hist = gaussian_filter(hist, sigma=sigma)
    if hist.sum() > 0:
        hist /= hist.sum()

    return hist


def visualize_lab_color_hist(hist, num_bins):
    hist_vis = np.zeros(num_bins)
    valid_bin_mask, _ = lab_rgb_gamut_bin_mask(num_bins)
    hist_vis[valid_bin_mask] = hist
    return hist_vis

