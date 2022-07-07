import numpy as np


def mask_to_bbox(mask):
    yinds, xinds = np.where(mask)
    return np.min(yinds), np.max(yinds), np.min(xinds), np.max(xinds)


def fill_mask_bg(image, mask, fill=0):
    """
    Fills pixels outside the mask with a constant value.
    :param image: to apply the mask to.
    :param mask: binary mask with True values for pixels that are to be preserved.
    :param fill: fill value.
    :return: Masked image
    """
    masked = image.copy()
    masked[~mask] = fill
    return masked


def fill_mask_fg(image, mask, fill=0):
    """
    Fills pixels outside the mask with a constant value.
    :param image: to apply the mask to.
    :param mask: binary mask with True values for pixels that are to be preserved.
    :param fill: fill value.
    :return: Masked image
    """
    masked = image.copy()
    masked[mask] = fill
    return masked


def crop_bbox(image, bbox):
    return image[bbox[0]:bbox[1], bbox[2]:bbox[3]]
