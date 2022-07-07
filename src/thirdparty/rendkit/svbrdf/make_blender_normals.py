import math
import argparse
import itertools
from pathlib import Path

import numpy as np
from scipy.misc import imsave

from rendkit.pfm import pfm_read
from svbrdf import beckmann
from toolbox.io.images import save_image, save_hdr, load_hdr
from toolbox.logging import init_logger

logger = init_logger(__name__)


def fix_nan(array):
    mask = np.isnan(array)
    yy, xx = np.where(mask)
    for y, x in zip(yy, xx):
        neigh = array[y-1:y+2, x-1:x+2]
        array[y, x] = neigh[~np.isnan(neigh)].mean()
    return array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='path', type=str)
    args = parser.parse_args()
    path = Path(args.path)

    logger.info("Fixing {}".format(path))
    orig_norm_path = Path('svbrdf-aittala', *path._parts[1:], 'out/reverse/map_normal.pfm')
    orig_norm = pfm_read(orig_norm_path)

    normal_map_blender = orig_norm.copy().astype(dtype=np.float32)
    # Normalize X and Y to [0, 1] to follow blender conventions.
    normal_map_blender[:, :, :2] += 1.0
    normal_map_blender[:, :, :2] /= 2.0
    normal_map_blender = np.round(255.0 * normal_map_blender).astype(np.uint8)
    imsave(path / beckmann.BLEND_NORMAL_MAP_NAME, normal_map_blender)


if __name__ == '__main__':
    main()
