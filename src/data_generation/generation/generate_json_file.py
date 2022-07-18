"""
generate original blender json file.
"""
import sys
sys.path.append('/home/code/TMT/src')
import config
import json
import numpy as np
import os
import random
from data_generation.preprocess.pairs import utils
from pathlib import Path
from thirdparty.rendkit.meshkit import wavefront

FOV_MIN = 50.0
FOV_MAX = 52.0

with open(config.CAMEA_POSE_PIROR_PATH,'r') as f1:
    blocks_dict = json.load(f1)


def main(rends_per_shape, max_dist):
    # get shapes with pairs.
    all_shapes = sorted(config.SHAPE_ROOT.iterdir())
    target_shapes = []
    target_pairs = []
    with open(config.PAIRS_JSON_PATH, 'r') as f1:
        all_pairs = json.load(f1)
    for pair in all_pairs:
        if pair["shape"] not in target_shapes:
            target_shapes.append(pair["shape"])
            # each shape only use one target pairs.
            # this code need to be improved to use pair with the smallest distance.
            target_pairs.append(pair["id"])
    
    for idx, shape in enumerate(target_shapes):
        final_pair = all_pairs[target_pairs[idx] - 1]
        # load shapes.
        mesh = wavefront.read_obj_file(final_pair["shape"] + '/models/uvmapped_v2.obj')
        mat_list = mesh.materials
        sorted(mat_list)
        mat_number = len(mesh.materials)
        # get original material assignment.
        seg_substances = utils.compute_partnet_segment_substances(final_pair)
        # deal with 'background'.
        for mat in mat_list:
            if mat not in seg_substances or seg_substances[mat] == 'background':
                seg_substances[mat] = 'null'  
        segment_ids = {name: i for i, name in enumerate(mat_list)}
        # sample camera pose.
        cam_angles = [] 
        for render_idx in range(rends_per_shape):
            randm_int = random.randint(0, 157184)
            for block in blocks_dict:
                if randm_int > blocks_dict[block]['sta_num'] and randm_int <= blocks_dict[block]['end_num']:
                    cam_angles.append((blocks_dict[block]['sta_azi'] + random.uniform(0,0.17), blocks_dict[block]['sta_ele'] + random.uniform(0,0.10)))
            cam_azimuth, cam_elevation = cam_angles[render_idx]
            cam_dist = random.uniform(1.3, 1.35)
            cam_fov = random.uniform(FOV_MIN, FOV_MAX)
            params = {
                'pair_id': pair["id"],
                'shape_id': int(pair["shape"][pair["shape"].rfind('/')+1:]),
                'exemplar_id': int(pair["exemplar"][pair["exemplar"].rfind('/')+1:]),
                'camera': {
                    'fov': cam_fov,
                    'azimuth': cam_azimuth,
                    'elevation': cam_elevation,
                    'distance': cam_dist,
                },
                'segment': {
                    'segment_ids': segment_ids,
                    'substances': seg_substances,
                },
            }
            with open(Path(config.BLENDER_JSON_PATH, str(pair["id"]) + '_' + str(render_idx) + '.json'), 'w') as f:
                json.dump(params, f, indent=2)

if __name__ == '__main__':
    main(5, 20)