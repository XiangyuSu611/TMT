"""
Computes aligning features for all ShapeNet shapes.
"""
import sys
sys.path.append('/home/code/TMT/src')
import config
import json
import numpy as np
import os
import utils


from config import SUBSTANCES
from pathlib import Path
from pydensecrf import densecrf
from skimage import transform
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.io import imread, imsave
from skimage.morphology import binary_closing, disk
from skimage.transform import resize
from tqdm import tqdm
from thirdparty.pyhog import pyhog
from thirdparty.kitnn.kitnn.colors import tensor_to_image, rgb_to_lab, image_to_tensor
from thirdparty.kitnn.kitnn.utils import softmax2d
from thirdparty.toolbox.toolbox.images import visualize_map, resize


def apply_flow(image, vx, vy):
    yy, xx = np.mgrid[0:vx.shape[0], 0:vx.shape[1]]
    yy, xx = np.round(yy + vy), np.round(xx + vx)
    yy = np.clip(yy, 0, vx.shape[0]-1).astype(dtype=int)
    xx = np.clip(xx, 0, vx.shape[1]-1).astype(dtype=int)
    warped = image[yy, xx]
    return warped


def visualize_segment_map(segment_map):
    return visualize_map(segment_map,
                         bg_value=-1,
                         values=range(segment_map.max() + 1))


def visualize_substance_map(substance_map):
    return visualize_map(substance_map,
                         bg_value=SUBSTANCES.index('background'),
                         values=range(len(SUBSTANCES)-1))


def compute_features(image, bin_size, im_shape):
    image = resize(image, im_shape, anti_aliasing=True, mode='constant')
    image = gaussian(image, sigma=0.1, multichannel=True)
    return _compute_hog_features(image, bin_size=bin_size)


def _compute_hog_features(image, bin_size=8):
    image = image.astype(dtype=np.float32)
    for c in range(3):
        image[:, :, c] -= image.mean()
        image[:, :, c] /= image.std()

    padded = np.dstack([np.pad(image[:, :, d], bin_size,
                               mode='constant', constant_values=image.mean())
                        for d in range(image.shape[-1])])
    feat = pyhog.compute_pedro(padded.astype(dtype=np.float64), bin_size)
    feat = feat[:, :, -8:]
    feat = feat.reshape((1, -1))
    return feat.astype(np.float32)


def compute_features(path):
    image = imread(path)
    return alignment.compute_features(
        image, bin_size=config.ALIGN_BIN_SIZE, im_shape=config.ALIGN_IM_SHAPE)


def apply_segment_crf(image, segment_map, theta_p=0.05, theta_L=5, theta_ab=5):
    image_lab = tensor_to_image(rgb_to_lab(image_to_tensor(image)))
    perc = np.percentile(np.unique(image[:, :, :3].min(axis=2)), 98)
    bg_mask = np.all(image > perc, axis=2)
    p_y, p_x = np.mgrid[0:image_lab.shape[0], 0:image_lab.shape[1]]
    feats = np.zeros((5, *image_lab.shape[:2]), dtype=np.float32)
    d = min(image_lab.shape[:2])
    feats[0] = p_x / (theta_p * d)
    feats[1] = p_y / (theta_p * d)
    feats[2] = image_lab[:, :, 0] / theta_L
    feats[3] = image_lab[:, :, 1] / theta_ab
    feats[4] = image_lab[:, :, 2] / theta_ab
    vals = np.unique(segment_map)
    probs = np.zeros((*segment_map.shape, len(vals)))
    for i, val in enumerate(vals):
        probs[:, :, i] = segment_map == val
    probs[bg_mask, 0] = 3
    probs[~bg_mask & (segment_map == -1)] = 1 / (len(vals))
    probs = softmax2d(probs)
    crf = densecrf.DenseCRF2D(*probs.shape)
    unary = np.rollaxis(
        -np.log(probs), axis=-1).astype(dtype=np.float32, order='c')
    crf.setUnaryEnergy(np.reshape(unary, (probs.shape[-1], -1)))
    crf.addPairwiseEnergy(np.reshape(feats, (feats.shape[0], -1)),
                          compat=3)
    Q = crf.inference(20)
    Q = np.array(Q).reshape((-1, *probs.shape[:2]))
    probs = np.rollaxis(Q, 0, 3)
    cleaned_seg_ind_map = probs.argmax(axis=-1)
    cleaned_seg_map = np.full(cleaned_seg_ind_map.shape,
                              fill_value=-1, dtype=int)
    for ind in np.unique(cleaned_seg_ind_map):
        cleaned_seg_map[cleaned_seg_ind_map == ind] = vals[ind]
    return cleaned_seg_map


def warp_renderings(pbar, pair):
    pair_root = Path(config.PAIR_ROOT, str(pair["id"]))
    # load flow npz.
    flow_path = Path(pair_root, 'numpy', config.FLOW_DATA_NAME)
    flow = np.load(flow_path)['arr_0']
    vx, vy = flow[:, :, 0], flow[:, :, 1]
    # load seg map.
    seg_map_path = Path(pair_root, 'images', config.SHAPE_REND_SEGMENT_MAP_NAME)
    shape_seg_map = imread(seg_map_path).astype(int) - 1
    warped_seg_map = apply_flow(shape_seg_map, vx, vy)
    # load exemplar
    exemplar_root = pair['exemplar']
    image = transform.resize(imread(exemplar_root + '/cropped.jpg'),
                             config.SHAPE_REND_SHAPE, anti_aliasing=True,
                             mode='reflect')
    crf_seg_map = apply_segment_crf(image, warped_seg_map)
    # load exemplar substance map.
    subst_map = imread(exemplar_root + '/image/' + str(config.EXEMPLAR_SUBST_MAP_NAME))
    subst_map = resize(subst_map, crf_seg_map.shape[:2], order=0)
    seg_subst_ids = utils.compute_substance_ids_by_segment(subst_map, crf_seg_map)
    proxy_subst_map = utils.compute_segment_substance_map(crf_seg_map, seg_subst_ids)
    shape_subst_map = utils.compute_segment_substance_map(shape_seg_map, seg_subst_ids)
    # visualization.
    shape_seg_vis = visualize_segment_map(shape_seg_map)
    warped_seg_vis = visualize_segment_map(warped_seg_map)
    crf_seg_vis = visualize_segment_map(crf_seg_map)
    exemplar_subst_vis = visualize_substance_map(subst_map)
    proxy_subst_vis = visualize_substance_map(proxy_subst_map)
    shape_subst_vis = visualize_substance_map(shape_subst_map)

    proxy_subst_map_path = Path(pair_root, 'images', config.PAIR_PROXY_SUBST_MAP_NAME)
    imsave(proxy_subst_map_path, proxy_subst_map.astype(np.uint8))
    proxy_subst_vis_path = Path(pair_root, 'images', config.PAIR_PROXY_SUBST_VIS_NAME)
    imsave(proxy_subst_vis_path, (proxy_subst_vis * 255).astype(np.uint8))
    shape_subst_map_path = Path(pair_root, 'images', config.PAIR_SHAPE_SUBST_MAP_NAME)
    imsave(shape_subst_map_path, shape_subst_map.astype(np.uint8))
    shape_subst_vis_path = Path(pair_root, 'images', config.PAIR_SHAPE_SUBST_VIS_NAME)
    imsave(shape_subst_vis_path, (shape_subst_vis * 255).astype(np.uint8))
    warped_seg_map_path = Path(pair_root, 'images', config.PAIR_SHAPE_WARPED_SEGMENT_MAP_NAME)
    imsave(warped_seg_map_path, (warped_seg_map + 1).astype(np.uint8))
    warped_seg_vis_path = Path(pair_root, 'images', config.PAIR_SHAPE_WARPED_SEGMENT_VIS_NAME)
    imsave(warped_seg_vis_path, (warped_seg_vis * 255).astype(np.uint8))
    shape_seg_vis_path = Path(pair_root, 'images', config.SHAPE_REND_SEGMENT_VIS_NAME)
    imsave(shape_seg_vis_path, (shape_seg_vis * 255).astype(np.uint8))
    crf_seg_vis_path = Path(pair_root, 'images', config.PAIR_SHAPE_CLEAN_SEGMENT_VIS_NAME)
    imsave(crf_seg_vis_path, (crf_seg_vis * 255).astype(np.uint8))
    crf_seg_vis_path = Path(pair_root, 'images', config.PAIR_SHAPE_CLEAN_SEGMENT_MAP_NAME)
    imsave(crf_seg_vis_path, (crf_seg_map + 1).astype(np.uint8))
    
    ov_fig_warped = (rgb2gray(image)[:, :, None].repeat(3, 2)+ visualize_map(warped_seg_map)) / 2.0
    ov_fig_original = (rgb2gray(image)[:, :, None].repeat(3, 2)+ visualize_map(shape_seg_map)) / 2.0
    ov_fig_clean = (rgb2gray(image)[:, :, None].repeat(3, 2)+ visualize_map(crf_seg_map)) / 2.0
    ov_fig = np.hstack((ov_fig_original, ov_fig_warped, ov_fig_clean))
    ov_fig_path = Path(pair_root, 'images', config.PAIR_SEGMENT_OVERLAY_NAME)
    imsave(ov_fig_path, (ov_fig * 255).astype(np.uint8))
    tqdm.write(f' * Saved for pair {pair["id"]}')


def main():
    print(f"Fetching pairs")
    with open(config.PAIRS_JSON_PATH, 'r') as f1:
        pairs = json.load(f1)
    print(f"Computing warped renderings for {len(pairs)} pairs.")
    pbar = tqdm(pairs)
    for pair in pbar:
        pbar.set_description(f'Pair {pair["id"]}, Exemplar {pair["exemplar"]}')
        warp_renderings(pbar, pair)

if __name__ == '__main__':
    main()