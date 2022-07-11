# DISPLAY=:0 vglrun python -m data_preprocess.generate_data.generate_new_data --client-id 1 --host 127.0.0.1 --port 6667
import asyncio
import collections
import json
import random
import os
import attr
import math
import time
import visdom
import math
import numpy as np
import click as click
import sqlalchemy as sa
import thirdparty.toolbox.toolbox as toolbox
import thirdparty.vispy.vispy as vispy
import thirdparty.brender.brender as brender

from typing import List, Tuple, Dict
from sqlalchemy import orm
from thirdparty.brender.brender import Mesh
from thirdparty.brender.brender.material import NodesMaterial
from thirdparty.brender.brender.scene import BackgroundMode
from thirdparty.brender.brender.utils import suppress_stdout
from thirdparty.rendkit.meshkit import wavefront
from thirdparty.rendkit.rendkit import shortcuts

import config, models
from config import SUBSTANCES
from data_preprocess.pairs import utils
from data_preprocess.generate_data import collector
from models import partnet_Shape, ExemplarpartShapePair, Exemplar
from database import session_scope
from thirdparty.toolbox.toolbox import cameras
from data_preprocess.web.utils import make_http_client
from thirdparty.toolbox.toolbox.images import visualize_map
from thirdparty.toolbox.toolbox.logging import init_logger

vispy.use(app='glfw')  
logger = init_logger(__name__)
vis = visdom.Visdom(env='generate-data')
_TMP_MESH_PATH = './temp/_terial_generate_data_temp_mesh.obj'
_TMP_REND_PATH = './temp/_terial_generate_data_temp_rend.exr'
_REND_SHAPE = (500, 500)
pi = math.pi
FOV_MIN = 50.0
FOV_MAX = 52.0


with open('./newdata/camera_pose_piror.json','r') as f1:
    blocks_dict = json.load(f1)

with open('./newdata/material_find.json','r') as f2:
    mat_search_dict = json.load(f2)

mat_not_have = {}
for i in range(len(mat_search_dict.keys())):
    mat_not_have['material_' + str(i)] = 0


@attr.s()
class _Envmap(object):
    path: str = attr.ib()
    rotation: Tuple[float, float, float] = attr.ib()

@click.command()
@click.option('--category', default='chair')
@click.option('--client-id', required=True, type=str, default='3')
@click.option('--epoch-start', default=0)
@click.option('--rends-per-epoch', default=5, type=int)
@click.option('--dry-run', is_flag=True)
@click.option('--max-dist', default=12.0)
@click.option('--required-substances', type=str)
@click.option('--host', required=True)
@click.option('--port', default=6667)

def main(category, client_id, epoch_start, rends_per_epoch, dry_run,
         max_dist, host, port, required_substances):
    loop = asyncio.get_event_loop()
    http_sess = loop.run_until_complete(make_http_client())
    app = brender.Brender()
    collector_ctx = {
        'host': host,
        'port': port,
        'sess': http_sess,
        'client_id': client_id,
    }
    if required_substances:
        required_substances = set(required_substances.split(','))
        assert all(s in SUBSTANCES for s in required_substances)

    # get final shapes.
    shape_ids = []
    with session_scope() as sess:
       query = sess.query(ExemplarpartShapePair).filter(ExemplarpartShapePair.shape.has(category=category))
       all_pairs = query.all()
       count = query.count()
    for i in range(len(all_pairs)):
        shapes_id = all_pairs[i].shape_id
        if shapes_id not in shape_ids:
            shape_ids.append(shapes_id)
    shape_ids.sort()
    # shape_ids = shape_ids[]
    for epoch in shape_ids:
        # check save path exists or not.
        save_path = f'./newdata/masks/client={str(client_id)}/epoch={str(epoch)}/train'
        save_path2 = f'./newdata/masks/client={str(client_id)}/epoch={str(epoch)}/validation'
        if (os.path.exists(save_path) and len(os.listdir(save_path)) == (rends_per_epoch * 9)) \
            or (os.path.exists(save_path2) and len(os.listdir(save_path2)) == (rends_per_epoch * 9)):
            logger.info('Render for shape %d already exits!', epoch)
            continue
        logger.info('Render masks for shape %d', epoch)
        
        # load and get final pair. 
        with session_scope() as sess:
            pairs = (
                sess.query(ExemplarpartShapePair)
                    .join(partnet_Shape)
                    .join(Exemplar)
                    .filter(sa.and_(
                        partnet_Shape.split_set.isnot(None),
                        ExemplarpartShapePair.shape_id == epoch, 
                    ))
                    .options(orm.joinedload(ExemplarpartShapePair.exemplar),
                             orm.joinedload(ExemplarpartShapePair.shape))
                    .order_by(ExemplarpartShapePair.distance.asc())
                    .all())
            materials = sess.query(models.Material).all()
            envmaps = sess.query(models.Envmap).filter_by(enabled=True).all()
        final_pair = pairs[0]
        logger.info('Get pair id %d', final_pair.id)
        if not final_pair.data_exists(config.PAIR_SHAPE_CLEAN_SEGMENT_MAP_NAME):
            logger.warning('shape %d s pair %d not have enough information!', epoch, final_pair.id)
            continue

        # use camera poses piror.
        # use piror camera distribution from {count} pairs.
        cam_angles = [] 
        for render_num in range(rends_per_epoch):
            randm_int = random.randint(0, count)
            for block in blocks_dict:
                if randm_int > blocks_dict[block]['sta_num'] and randm_int <= blocks_dict[block]['end_num']:
                    cam_angles.append((blocks_dict[block]['sta_azi'] + random.uniform(0,0.17), blocks_dict[block]['sta_ele'] + random.uniform(0,0.10)))
                
        # get material substance.
        mats_by_subst = collections.defaultdict(list)
        for material in materials:
            mats_by_subst[material.substance].append(material)
        
        envmaps_by_split = {
            'train': [e for e in envmaps if e.split_set == 'train'],
            'validation': [e for e in envmaps if e.split_set == 'validation'],
        }

        # get original material number and names.
        mesh, _ = final_pair.shape.load()
        mat_list = mesh.materials
        mat_number = len(mesh.materials)
        
        # get original material assignment.
        ori_seg_substances = utils.compute_partnet_segment_substances(final_pair)
        tar_seg_substances = {}
        # deal with 'background'.
        for mat in mat_list:
            if mat not in ori_seg_substances or ori_seg_substances[mat] == 'background':
                print(mat)
                mat_not_have[mat] += 1
                mat_corr_list = mat_search_dict[mat]
                for i in range(mat_search_dict.keys()):
                    corr_mat = 'material_' + str(mat_corr_list[i])
                    if corr_mat in ori_seg_substances and ori_seg_substances[corr_mat]  != 'background':
                        tar_seg_substances[mat] = ori_seg_substances[corr_mat]
                        break
            else:
                tar_seg_substances[mat] = ori_seg_substances[mat]           
        app.init()
        try:
            loop.run_until_complete(
                process_pair(app,
                            final_pair,
                            cam_angles,
                            mats_by_subst,
                            envmaps_by_split=envmaps_by_split,
                            num_rends=rends_per_epoch,
                            is_dry_run=dry_run,
                            epoch=epoch,
                            collector_ctx=collector_ctx,
                            required_substances=required_substances,
                            substance=tar_seg_substances))
        except Exception as e:
            logger.exception('Uncaught exception', exc_info=True)
            continue
    with open('./data/training_data/mat_not_have.json', 'w') as f1:
        json.dump(mat_not_have, f1 , indent=2)
            
async def process_pair(app: brender.Brender,
                       pair: ExemplarpartShapePair,
                       cam_angles: List[Tuple[float, float]],
                       mats_by_subst: Dict[str, List[NodesMaterial]],
                       num_rends,
                       *,
                       envmaps_by_split,
                       is_dry_run,
                       epoch,
                       collector_ctx,
                       required_substances=None,
                       substance):
    rk_mesh, _ = pair.shape.load()
    rk_mesh.resize(1)
    rk_mesh_shapenet, _ = pair.shape.load_shapenet()
    rk_mesh_shapenet.resize(1)
    with open(_TMP_MESH_PATH, 'w') as f:
        wavefront.save_obj_file(f, rk_mesh)

    seg_substances = substance
    if required_substances:
        for subst in required_substances:
            if subst not in seg_substances.values():
                return

    logger.info('Predicted substance: %s', seg_substances)
    
    scene = brender.Scene(app, shape=_REND_SHAPE,
                          tile_size=(40, 40),
                          aa_samples=128,
                          diffuse_samples=3,
                          specular_samples=3,
                          background_mode=BackgroundMode.COLOR,
                          background_color=(1, 1, 1, 1))
    
    with suppress_stdout():
        mesh = Mesh.from_obj(scene, _TMP_MESH_PATH)
        # mesh.remove_doubles()
        mesh.enable_smooth_shading()
    
    for i in range(num_rends):
        bmats = []
        cam_angles_in = [cam_angles[i]]
        try:
            r = do_render(scene, pair, mesh, envmaps_by_split, cam_angles_in,
                      rk_mesh, rk_mesh_shapenet, seg_substances, mats_by_subst, bmats)
        finally:
            while len(bmats) > 0:
                bmat = bmats.pop()
                bmat.bobj.name = bmat.bobj.name
                del bmat
        
        if not is_dry_run:
            await collector.send_data(
                **collector_ctx,
                split_set=pair.shape.split_set,
                pair_id=pair.id,
                epoch=epoch,
                iteration=i,
                params=r['params'],
                seg_map=r['seg_map'],
                seg_vis=r['seg_vis'],
                shapenet_map=r['shapenet_map'],
                shapenet_vis=r['shapenet_vis'],
            )


def do_render(scene, pair, mesh, envmaps_by_split, cam_angles,
              rk_mesh, rk_mesh_shapenet, seg_substances, mats_by_subst, bmats):
    time_begin = time.time()

    # Jitter camera params.
    cam_azimuth, cam_elevation = random.choice(cam_angles)
    cam_dist = random.uniform(1.3, 1.35)
    cam_fov = random.uniform(FOV_MIN, FOV_MAX)
    rk_camera = cameras.spherical_coord_to_cam(
        cam_fov, cam_azimuth, cam_elevation, cam_dist=cam_dist,
        max_len=_REND_SHAPE[0]/2)

    # Jitter envmap params.
    envmap_scale = random.uniform(0.9, 1.2)
    envmap = random.choice(envmaps_by_split[pair.shape.split_set])
    envmap_rotation = (0, 0, (envmap.azimuth + pi/2 + cam_azimuth
                              + random.uniform(-pi/24, pi/24)))
    scene.set_envmap(
        envmap.get_data_path('hdr.exr'),
        scale=envmap_scale, rotation=envmap_rotation)

    if scene.camera is None:
        camera = brender.CalibratedCamera(
            scene, rk_camera.cam_to_world(), cam_fov)
        scene.set_active_camera(camera)
    else:
        scene.camera.set_params(rk_camera.cam_to_world(), cam_fov)

    # no need to generate seg map because it will be generated later
    seg_map = shortcuts.render_segments_partnet(rk_mesh, rk_camera)
    seg_vis = visualize_map(seg_map)[:, :, :3]
    # normal_map = shortcuts.render_mesh_normals(rk_mesh, rk_camera)
    
    # get part_map
    shapenet_map = shortcuts.render_segments(rk_mesh_shapenet, rk_camera)
    shapenet_vis = visualize_map(shapenet_map)[:, :, :3]
    

    segment_ids = {
        name: i for i, name in enumerate(rk_mesh.materials)
    }

    params = {
        'split_set': pair.shape.split_set,
        'pair_id': pair.id,
        'shape_id': pair.shape_id,
        'exemplar_id': pair.exemplar_id,
        'camera': {
            'fov': cam_fov,
            'azimuth': cam_azimuth,
            'elevation': cam_elevation,
            'distance': cam_dist,
        },
        'envmap': {
            'id': envmap.id,
            'name': envmap.name,
            'source': envmap.source,
            'scale': envmap_scale,
            'rotation': envmap_rotation,
        },
        'segment': {
            'segment_ids': segment_ids,
            'substances': seg_substances,
        },
    }

    return {
        'seg_map': (seg_map + 1).astype(np.uint8),
        'seg_vis': toolbox.images.to_8bit(seg_vis),
        'shapenet_map': (shapenet_map + 1).astype(np.uint8),
        'shapenet_vis': toolbox.images.to_8bit(shapenet_vis),
        'params': params,
    }


if __name__ == '__main__':
    main()
