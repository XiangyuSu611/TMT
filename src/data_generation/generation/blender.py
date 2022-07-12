import bpy
import config
import math
import shutil
import tempfile
from click import Path
import thirdparty.brender.brender as brender
from data_generation.generation import material_loader as loader
from thirdparty.brender.brender.material import DiffuseMaterial
from thirdparty.brender.brender.mesh import Mesh, Plane
from thirdparty.brender.brender.scene import BackgroundMode
from thirdparty.brender.brender.utils import suppress_stdout
from thirdparty.rendkit.meshkit import wavefront
from thirdparty.toolbox.toolbox import cameras


def construct_realimage_scene_chair(app: brender.Brender,
                              cocos_inference_dict,
                              mat_by_idx,
                              scene_type='inferred',
                              num_samples=256,
                              rend_shape=(1280, 1280),
                              tile_size=(512, 512),
                              frontal_camera=False,
                              diagonal_camera=False,
                              add_floor=True):
    
    if scene_type not in {'inferred', 'mtl'}:
        raise ValueError('Invalid scene type.')
    
    inference_dict = cocos_inference_dict
    shapeid = cocos_inference_dict['base_info']['shape']
    shape_root = config.SHAPE_ROOT
    obj_path = str(shape_root) + '/' + str(shapeid) + '/models/uvmapped_v2.obj'
    rk_mesh = wavefront.read_obj_file(obj_path)  
    rk_mesh.resize(1)
    scene = brender.Scene(app, shape=rend_shape,
                          num_samples=num_samples,
                          tile_size=tile_size,
                          background_mode=BackgroundMode.COLOR,
                          background_color=(1.0, 1.0, 1.0, 0))
    distance = 2.3
    fov = 50
    cam_loc = cocos_inference_dict['base_info']['camera']

    # Get exemplar camera parameters.
    rk_camera = cameras.location_to_cam(fov, cam_loc, max_len=rend_shape[0]/2)

    camera = brender.CalibratedCamera(scene, rk_camera.cam_to_world(), fov)

    scene.set_active_camera(camera)

    with suppress_stdout():
        # mesh = Mesh.from_obj(scene, tar_shape.resized_obj_path)
        mesh = Mesh.from_obj(scene, obj_path)
        mesh.make_normals_consistent()
        mesh.enable_smooth_shading()
    mesh.recenter()

    if add_floor:
        min_pos = mesh.compute_min_pos()
        floor_mat = DiffuseMaterial(diffuse_color=(1.0, 1.0, 1.0))
        floor_mesh = Plane(position=(0, 0, min_pos))
        floor_mesh.set_material(floor_mat)
    
    generate_type = 'final_result'
    for seg_id, seg_name in enumerate(rk_mesh.materials):
        if seg_name in inference_dict[generate_type]:
            mat_id = inference_dict[generate_type][seg_name]
            material = mat_by_idx[mat_id-1]
            uv_ref_scale = 2 ** (material["default_scale"] - 3)
            print(f'({seg_name}) to material {material["name"]}')
            for bobj in bpy.data.materials:
                if bobj.name == seg_name:
                    bmat = loader.material_to_brender(
                        material, bobj=bobj, uv_ref_scale=uv_ref_scale)
                    scene.add_bmat(bmat)
    print('Compute UV density...')
    mesh.compute_uv_density()

    return scene