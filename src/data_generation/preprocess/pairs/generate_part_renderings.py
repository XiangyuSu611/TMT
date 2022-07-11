"""
Computes aligning features for all shapes.
"""

import config, controllers
import numpy as np
import os
import skimage
import thirdparty.toolbox.toolbox.images

from skimage.io import imread
from tqdm import tqdm
from thirdparty.rendkit.rendkit.graphics_utils import compute_tight_clipping_planes
from thirdparty.rendkit.rendkit.shortcuts import (render_segments, render_wavefront_mtl, render_mesh_normals)
from thirdparty.toolbox.toolbox import cameras
from thirdparty.toolbox.toolbox.images import mask_bbox, crop_tight_fg, visualize_map
from thirdparty.vispy.vispy import app


app.use_app('glfw')


def parse_rend_filename(fname):
    fname, _ = os.path.splitext(fname)
    s = [s.split('=') for s in fname.split(',')]
    return {k: v for k, v in s}


def compute_features(path):
    image = imread(path)
    return alignment.compute_features(
        image, bin_size=config.ALIGN_BIN_SIZE, im_shape=config.ALIGN_IM_SHAPE)


def main():
    with open(config.PAIRS_JSON_PATH, 'r') as f1:
        pairs = json.load(f1)
        pbar = tqdm(pairs)
        for idx, pair in enumerate(pbar):
            render_model_exemplars(pbar, pair)
        

def render_model_exemplars(pbar, pair: ExemplarShapePair,
                           render_shape=config.SHAPE_REND_SHAPE):

    camera = cameras.spherical_coord_to_cam(
        pair['fov'], pair['azimuth'], pair['elevation'])

    pbar.set_description(f'Loading shape')
    mesh = wavefront.read_obj_file(Path(pair['shape'], 'models', 'uvmapped_v2.obj'))
    mesh.resize(100)
    materials = wavefront.read_mtl_file(Path(pair['shape'], 'models', 'uvmapped_v2.mtl'), mesh)
    camera.near, camera.far = compute_tight_clipping_planes(mesh, camera.view_mat())
    pbar.set_description(f'[{pair.id}] Rendering segments')
    if not os.path.exists(Path(config.DATA_ROOT, 'exemplars', pair['id'], 'images', config.PAIR_FG_BBOX_NAME)):
        print(type(camera))
        segment_im = render_segments(mesh, camera)
        fg_mask = segment_im > -1
        fg_bbox = mask_bbox(fg_mask)
        segment_im = crop_tight_fg(
            segment_im, render_shape,
            bbox=fg_bbox, fill=-1, order=0)
        segment_vis_path = Path(config.DATA_ROOT, 'pairs', pair['id'], 'images', config.SHAPE_REND_SEGMENT_VIS_NAME)
        segment_vis_path.parent.mkdir(parents=True, exist_ok=True)
        imsave(segment_vis_path, (visualize_map(segment_im) * 255.0).astype(np.uint8))
        segment_map_path = Path(config.DATA_ROOT, 'pairs', pair, 'images', config.SHAPE_REND_SEGMENT_MAP_NAME)
        segment_map_path.parent.mkdir(parents=True, exist_ok=True)
        imsave(segment_map_path, (segment_im + 1).astype(np.uint8))   
    
        tqdm.write(f" * Saving bounding box.")
        raw_segment_map_path = Path(config.DATA_ROOT, 'pairs', pair['id'], 'images', config.PAIR_RAW_SEGMENT_MAP_NAME)
        raw_segment_map_path.parent.mkdir(parents=True, exist_ok=True)
        imsave(raw_segment_map_path, (segment_im + 1).astype(np.uint8))

        raw_segment_map_path = Path(config.DATA_ROOT, 'pairs', pair['id'], 'images', config.PAIR_FG_BBOX_NAME)
        raw_segment_map_path.parent.mkdir(parents=True, exist_ok=True)
        imsave(raw_segment_map_path, (fg_mask.astype(np.uint8) * 255).astype(np.uint8))
    else:
        fg_mask = imread(Path(config.DATA_ROOT, 'exemplars', pair['id'], 'images', config.PAIR_FG_BBOX_NAME))
        fg_bbox = mask_bbox(fg_mask)
    rend_phong_path = Path(config.DATA_ROOT, 'exemplars', pair['id'], 'images', config.PAIR_FG_BBOX_NAME)
    rend_phong_path.parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(rend_phong_path):
        pbar.set_description('Rendering phong')
        phong_im = np.clip(
            render_wavefront_mtl(mesh, camera, materials,
                                 config.SHAPE_REND_RADMAP_PATH,
                                 gamma=2.2, ssaa=3, tonemap='reinhard'), 0, 1)
        phong_im = crop_tight_fg(phong_im, render_shape, bbox=fg_bbox)
        pbar.set_description(f'Saving data')
        imsave(rend_phong_path, skimage.img_as_uint(phong_im * 255.))

if __name__ == '__main__':
    main()
