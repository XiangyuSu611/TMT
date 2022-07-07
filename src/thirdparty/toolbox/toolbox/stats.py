import numpy as np
from numpy import linalg


EPS = 0.00000000000001


def normalize_to_range(array, lo, hi):
    if hi <= lo:
        raise ValueError('Range must be increasing but {} >= {}.'.format(
            lo, hi))
    min_val = array.min()
    max_val = array.max()
    scale = max_val - min_val if (min_val < max_val) else 1
    return (array - min_val) / scale * (hi - lo) + lo


def normalize_to_unit(points):
    return points / linalg.norm(points, axis=1)[:, np.newaxis]


def find_outliers(data, thres=3.5):
    """
    Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    data = np.array(data)
    median = np.median(data, axis=0)
    if data.ndim >= 2:
        diff = linalg.norm(data - median, axis=-1)
    else:
        diff = np.abs(data - median)
    med_abs_dev = np.median(diff)
    mean_abs_dev = np.mean(diff)
    if abs(med_abs_dev) > EPS:
        modified_z_score = 0.6745 * diff / med_abs_dev
    elif abs(mean_abs_dev) > EPS:
        modified_z_score = 0.6745 * diff / mean_abs_dev
    else:
        modified_z_score = np.zeros(diff.shape)
    return modified_z_score > thres


def reject_outliers(data, thres=3.5):
    inlier_mask = find_outliers(data, thres)
    return data[~inlier_mask]


def robust_mean(data, thres=3.5):
    return np.mean(reject_outliers(data, thres=thres))


def gaussian2d(mean, sigma, size=(100, 100)):
    x = np.linspace(-1, 1, num=size[1])
    y = np.linspace(-1, 1, num=size[0])
    gauss_y = np.exp(-((y - mean[0])**2)/(2*sigma[0]**2))[:, None]
    gauss_x = np.exp(-((x - mean[1])**2)/(2*sigma[1]**2))[:, None]
    return gauss_x.dot(gauss_y.T)
