import math
import argparse
import itertools
from pathlib import Path

import numpy as np

from rendkit.pfm import pfm_read
from svbrdf import beckmann
from toolbox.io.images import save_hdr, load_hdr
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
    spec_path = path / beckmann.SPEC_MAP_NAME

    orig_spec_path = Path('svbrdf-aittala', *path._parts[1:], 'out/reverse/map_spec.pfm')
    orig_spec = pfm_read(orig_spec_path)

    logger.info("[{}]".format(args.path))
    rough_map = load_hdr(path / 'map_roughness.exr')
    aniso_map = load_hdr(path / 'map_anisotropy.exr')

    # Fix NaNs
    if np.isnan(aniso_map).sum() > 0:
        logger.warning("Anisotropy map has NaNs. Fixing!")
        aniso_map = fix_nan(aniso_map)
        save_hdr(path / 'map_anisotropy.exr', aniso_map)

    if np.isnan(rough_map).sum() > 0:
        logger.warning("Roughness map has NaNs. Fixing!")
        rough_map = fix_nan(rough_map)
        save_hdr(path / 'map_roughness.exr', aniso_map)

    aniso_neg = aniso_map < 0
    aniso_pos = ~aniso_neg
    alpha_x_map = np.zeros(rough_map.shape)
    alpha_y_map = np.zeros(rough_map.shape)
    alpha_x_map[aniso_neg] = rough_map[aniso_neg] / (1.0 + aniso_map[aniso_neg])
    alpha_y_map[aniso_neg] = rough_map[aniso_neg] * (1.0 + aniso_map[aniso_neg])
    alpha_x_map[aniso_pos] = rough_map[aniso_pos] * (1.0 - aniso_map[aniso_pos])
    alpha_y_map[aniso_pos] = rough_map[aniso_pos] / (1.0 - aniso_map[aniso_pos])

    scale = (4.0 * math.pi * np.median(alpha_x_map * alpha_y_map))
    logger.info("Scaling by {:03f} (reciprocal {:03f})".format(scale, 1/scale))
    spec_map = orig_spec * scale

    logger.info("Saving specular")
    save_hdr(path / 'map_specular.exr', spec_map)
    # logger.info("Saving alpha x")
    # save_hdr(path / 'map_alpha_x.exr', alpha_x_map)
    # logger.info("Saving alpha y")
    # save_hdr(path / 'map_alpha_y.exr', alpha_y_map)


if __name__ == '__main__':
    main()
