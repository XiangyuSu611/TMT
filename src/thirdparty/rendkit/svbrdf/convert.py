import math
import argparse
import itertools
from pathlib import Path

import torch
import numpy as np
from scipy.interpolate import griddata
from tqdm import trange

from svbrdf.beckmann import BeckmannSVBRDF
from svbrdf.aittala import AittalaSVBRDF
from toolbox.logging import init_logger

logger = init_logger(__name__)


CROP_SIZE = (200, 200)
N_PHI = 500
N_THETA_MAX = 1000


def aittala_ndf(H, S, alpha=2):
    S = S.view(1, 2, 2).expand(H.size(0), 2, 2)
    h = H.view(-1, 2, 1)
    hT = H.view(-1, 1, 2)
    e = (torch.bmm(hT, torch.bmm(S, h)))
    e = e.abs() ** alpha
    return torch.exp(-e)


def beckmann_aniso_ndf(H, alpha_x, alpha_y):
    hx = H[0].view(-1, 1)
    hy = H[1].view(-1, 1)
    hz = H[2].view(-1, 1)

    if alpha_x == 0 or alpha_y == 0:
        # Avoid divide by zero by returning None. The actual NDF value should
        # be 1.0.
        return None

    slope_x = (-hx / (hz * alpha_x))
    slope_y = (-hy / (hz * alpha_y))
    cosThetaM = hz
    cosThetaM2 = cosThetaM * cosThetaM
    cosThetaM4 = cosThetaM2 * cosThetaM2
    cosThetaM4 = cosThetaM4.expand(*slope_x.size())
    D = torch.exp(-slope_x * slope_x - slope_y * slope_y) / cosThetaM4
    return D


def angles_to_half_vec(angles):
    """
    Converts spherical coordinates to half vectors.
    """
    H = torch.zeros(3, len(angles)).cuda()
    phi = angles[:, 0]
    theta = angles[:, 1]
    H[0] = torch.cos(theta) * torch.sin(phi)
    H[1] = torch.sin(theta) * torch.sin(phi)
    H[2] = torch.cos(phi)
    return H


def fit_beckmann(D_aittala, H, ones):
    ax_best = 0.5
    ay_best = 0.5
    ax_lo, ax_hi = 0, 1
    ay_lo, ay_hi = 0, 1
    best_err = float('inf')

    win_size = 3
    for branch in range(9):
        ax_cand = np.linspace(ax_lo, ax_hi, win_size)
        ay_cand = np.linspace(ay_lo, ay_hi, win_size)
        param_cand = itertools.product(ax_cand, ay_cand)
        for ax, ay in param_cand:
            if ax == 0 or ay == 0:
                D = ones
            else:
                D = beckmann_aniso_ndf(H, ax, ay)
            err = (D - D_aittala).abs()
            err = err.sum()
            if err < best_err:
                ax_best = ax
                ay_best = ay
                best_err = err
        ax_win = (ax_hi - ax_lo) / (win_size)
        ay_win = (ax_hi - ax_lo) / (win_size)
        ax_lo, ax_hi = max(0, ax_best - ax_win), min(1, ax_best + ax_win)
        ay_lo, ay_hi = max(0, ay_best - ay_win), min(1, ay_best + ay_win)
    return ax_best, ay_best, best_err


def fix_nan(array):
    mask = np.isnan(array)
    yy, xx = np.where(mask)
    for y, x in zip(yy, xx):
        neigh = array[y-1:y+2, x-1:x+2]
        array[y, x] = neigh[~np.isnan(neigh)].mean()
    return array


def rough_aniso_to_alpha(rough_map, aniso_map):
    aniso_neg = aniso_map < 0
    aniso_pos = ~aniso_neg
    alpha_x_map = np.zeros(rough_map.shape)
    alpha_y_map = np.zeros(rough_map.shape)
    alpha_x_map[aniso_neg] = rough_map[aniso_neg] / (1.0 + aniso_map[aniso_neg])
    alpha_y_map[aniso_neg] = rough_map[aniso_neg] * (1.0 + aniso_map[aniso_neg])
    alpha_x_map[aniso_pos] = rough_map[aniso_pos] * (1.0 - aniso_map[aniso_pos])
    alpha_y_map[aniso_pos] = rough_map[aniso_pos] / (1.0 - aniso_map[aniso_pos])
    return alpha_x_map, alpha_y_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--aittala', dest='aittala_path', type=str,
                        required=True)
    parser.add_argument('-o', '--out', dest='out_path', type=str,
                        required=True)
    args = parser.parse_args()

    if Path(args.out_path).exists():
        logger.error("{} already exists.".format(args.out_path))
        return

    svbrdf = AittalaSVBRDF(args.aittala_path)

    crop_spec_shape = svbrdf.spec_shape_map[:CROP_SIZE[0], :CROP_SIZE[1]].reshape((-1, 3))
    crop_spec_shape = crop_spec_shape[:, [0, 1, 1, 2]].reshape((-1, 2, 2))
    crop_spec_shape = torch.from_numpy(crop_spec_shape).cuda()

    logger.info("Generating sample angles..")
    phis = np.linspace(0.0, math.pi / 2, N_PHI)
    angles = []
    for phi in phis:
        n_theta = int(round(N_THETA_MAX * math.sin(phi) + 1))
        thetas = np.linspace(0, math.pi * 2, n_theta)
        for theta in thetas:
            angles.append((phi, theta))
    H = angles_to_half_vec(torch.FloatTensor(angles))
    H_tan = H[:2, :] / H[2].contiguous().view(1, -1).expand(*H[:2, :].size())
    H_tan = H_tan.t().contiguous()

    logger.info("Fitting Beckmann BRDF...")
    ones = torch.ones(H.size(1)).cuda()
    crop_rough_map = np.zeros(CROP_SIZE)
    crop_aniso_map = np.zeros(CROP_SIZE)
    for i in trange(crop_spec_shape.size(0)):
        row, col = i // CROP_SIZE[0], i % CROP_SIZE[1]
        D_aittala = aittala_ndf(H_tan, crop_spec_shape[i], alpha=svbrdf.alpha)
        ax, ay, err = fit_beckmann(D_aittala, H, ones)
        if ay > ax:
            aniso = 1 - math.sqrt(ax / ay)
            roughness = ay * (1 - aniso)
        else:
            aniso = math.sqrt(ay / ax) - 1
            roughness = ax * (1 + aniso)
        crop_aniso_map[row, col] = aniso
        crop_rough_map[row, col] = roughness

    logger.info("Interpolating sample fit to full texture...")
    crop_spec_shape = svbrdf.spec_shape_map[:CROP_SIZE[0],
                                            :CROP_SIZE[1]].reshape((-1, 3))
    rough_map = griddata(crop_spec_shape, crop_rough_map.flatten(),
                         svbrdf.spec_shape_map.reshape((-1, 3)),
                         method='nearest')
    rough_map = rough_map.reshape(svbrdf.spec_shape_map.shape[:2])
    aniso_map = griddata(crop_spec_shape, crop_aniso_map.flatten(),
                         svbrdf.spec_shape_map.reshape((-1, 3)),
                         method='nearest')
    aniso_map = aniso_map.reshape(svbrdf.spec_shape_map.shape[:2])
    if np.isnan(aniso_map).sum() > 0:
        logger.warning("Roughness map has NaNs. Fixing!")
        rough_map = fix_nan(rough_map)
    if np.isnan(rough_map).sum() > 0:
        logger.warning("Anisotropy map has NaNs. Fixing!")
        aniso_map = fix_nan(aniso_map)
    alpha_x_map, alpha_y_map = rough_aniso_to_alpha(rough_map, aniso_map)

    # Correct for energy conservation since Aittala and Beckmann use different
    # constants.
    spec_scale = (4.0 * math.pi * (alpha_x_map * alpha_y_map).mean())
    logger.info("Scaling specular albedo by {}".format(spec_scale))

    logger.info("Saving...")
    bsvbrdf = BeckmannSVBRDF(svbrdf.diffuse_map * math.pi,
                             svbrdf.specular_map * spec_scale,
                             svbrdf.normal_map,
                             rough_map,
                             aniso_map)
    bsvbrdf.save(args.out_path)


if __name__ == '__main__':
    main()
