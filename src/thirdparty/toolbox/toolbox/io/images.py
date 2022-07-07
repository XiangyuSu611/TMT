from pathlib import Path
from typing import Union

import numpy as np
from scipy import misc

from thirdparty.toolbox.toolbox.io import pfm


__all__ = [
    'save_image', 'load_image', 'save_hdr', 'load_hdr', 'save_arr', 'load_arr',
    'imread2', 'is_img',
]


def save_image(path, array):
    if array.dtype == np.uint8:
        array = array.astype(dtype=float)
        array /= 255.0
    array = np.round(np.clip(array, 0.0, 1.0) * 255.0).astype(dtype=np.uint8)
    misc.imsave(path, array)


def imread2(path):
    path = Path(path)
    if path.suffix in {'.hdr', '.exr'}:
        return load_hdr(path)
    return load_image(path)


def load_image(path, mode='RGB'):
    image = misc.imread(path, mode=mode)
    image = image.astype(dtype=np.float32) / 255.0
    return image


def save_arr(path, arr):
    np.savez(path, arr)


def load_arr(path):
    return np.load(path)['arr_0'][()]


def save_hdr(path: Path, image):
    if isinstance(path, str):
        path = Path(path)
    ext = path.suffix[1:]
    if ext == 'exr':
        if len(image.shape) == 3:
            image = image[:, :, [2, 1, 0]]
        import cv2
        cv2.imwrite(str(path), image)
    elif ext == 'pfm':
        pfm.pfm_write(str(path), image)
    else:
        raise RuntimeError("Unknown format {}".format(ext))


def load_hdr(path: Union[Path, str], ext=None):
    if isinstance(path, str):
        path = Path(path)

    if ext is None:
        ext = path.suffix[1:]
    if ext in {'exr', 'hdr'}:
        import cv2
        im = cv2.imread(str(path), -1)
        if len(im.shape) == 3:
            im = im[:, :, [2, 1, 0]]
    elif ext == 'pfm':
        im = pfm.pfm_read(path)
    else:
        raise RuntimeError("Unknown format {}".format(ext))
    return im


def is_img(path):
    img_types = ['png', 'tiff', 'tif', 'jpg', 'gif', 'jpeg']
    for t in img_types:
        if str(path).lower().endswith(t):
            return True
    return False