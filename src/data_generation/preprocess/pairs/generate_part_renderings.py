"""
Computes aligning features for all shapes.
"""
import sys
sys.path.append('/home/code/TMT/src')
import config
import json
import numpy as np
import os
import skimage
from pathlib import Path
from skimage.io import imread, imsave
from tqdm import tqdm
from thirdparty.toolbox.toolbox.images import mask_bbox, crop_tight_fg, visualize_map


def render_model_exemplars(pbar, pair, render_shape=config.SHAPE_REND_SHAPE):
    pair_image_dir = Path(config.DATA_ROOT, 'pairs', str(pair['id']), 'images')
    if not os.path.exists(Path(pair_image_dir, config.PAIR_FG_BBOX_NAME)):
        tqdm.write(f" * Rendering.")
        os.system(f'blender -b -P ./src/data_generation/preprocess/pairs/generate_part_rendering.py -- \
            {str(pair["shape"])}/models/uvmapped_v2.obj \
            {str(pair["id"])} \
            {str(pair["azimuth"])} \
            {str(pair["elevation"])}')
        segment_im = imread(Path(pair_image_dir, config.SHAPE_REND_SEGMENT_MAP_NAME)) - 1
        fg_mask = segment_im > -1 
        fg_bbox = mask_bbox(fg_mask)
        segment_im = crop_tight_fg(
            segment_im, render_shape,
            bbox=fg_bbox, fill=-1, order=0)
        segment_vis_path = Path(pair_image_dir, config.SHAPE_REND_SEGMENT_VIS_NAME)
        imsave(segment_vis_path, (visualize_map(segment_im + 1) * 255.0).astype(np.uint8))
        segment_map_path = Path(pair_image_dir, config.SHAPE_REND_SEGMENT_MAP_NAME)
        imsave(segment_map_path, (segment_im + 1).astype(np.uint8))   
        tqdm.write(f" * Saving bounding box.")
        raw_segment_map_path = Path(pair_image_dir, config.PAIR_RAW_SEGMENT_MAP_NAME)
        imsave(raw_segment_map_path, (segment_im).astype(np.uint8))
        raw_segment_map_path = Path(pair_image_dir, config.PAIR_FG_BBOX_NAME)
        raw_segment_map_path.parent.mkdir(parents=True, exist_ok=True)
        imsave(raw_segment_map_path, (fg_mask.astype(np.uint8) * 255).astype(np.uint8))
    else:
        fg_mask = imread(Path(pair_image_dir, config.PAIR_FG_BBOX_NAME))
        fg_bbox = mask_bbox(fg_mask)
    
    phong_im = imread(Path(pair_image_dir, config.SHAPE_REND_PHONG_NAME))
    phong_im = crop_tight_fg(phong_im, render_shape, bbox=fg_bbox)
    pbar.set_description(f' * Saving phone image.')
    imsave(Path(pair_image_dir, config.SHAPE_REND_PHONG_NAME), skimage.img_as_uint(phong_im))


def main():
    with open(config.PAIRS_JSON_PATH, 'r') as f1:
        pairs = json.load(f1)
        pbar = tqdm(pairs)
        for idx, pair in enumerate(pbar):
            render_model_exemplars(pbar, pair)


if __name__ == '__main__':
    main()
