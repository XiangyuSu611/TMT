import math
import copy
from typing import List, Dict

import numpy as np

from thirdparty.rendkit.meshkit import Mesh
from  thirdparty.rendkit.meshkit.wavefront import WavefrontMaterial
from  thirdparty.rendkit.rendkit import jsd
from  thirdparty.toolbox.toolbox import images
from  thirdparty.toolbox.toolbox.logging import init_logger
from .camera import ArcballCamera
from .jsd import JSDRenderer

logger = init_logger(__name__)


QUAL_COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (230, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (0, 0, 128),
    (128, 128, 128),
    (166, 206, 227),
    (31, 120, 180),
    (178, 223, 138),
    (51, 160, 44),
    (251, 154, 153),
    (227, 26, 28),
    (253, 191, 111),
    (255, 127, 0),
    (202, 178, 214),
    (106, 61, 154),
    (255, 255, 153),
    (177, 89, 40),
    (141, 211, 199),
    (255, 255, 179),
    (190, 186, 218),
    (251, 128, 114),
    (128, 177, 211),
    (253, 180, 98),
    (179, 222, 105),
    (252, 205, 229),
    (217, 217, 217),
    (188, 128, 189),
    (204, 235, 197),
    (255, 237, 111),
]


def svbrdf_plane_renderer(svbrdf, size=None, lights=list(), radmap=None,
                          mode='all', gamma=2.2, uv_scale=1.0, shape=None,
                          transpose=False, camera=None,
                          cam_lookat=(0.0, 0.0), cam_fov=90,
                          cam_dist=1.0, cam_up=(1.0, 0.0, 0.0), **kwargs):
    if shape is None:
        height, width, _ = svbrdf.diffuse_map.shape
    else:
        height, width = shape[:2]
    zeros = np.zeros(svbrdf.diffuse_map.shape, dtype=np.float32)
    ones = np.ones(svbrdf.diffuse_map.shape, dtype=np.float32)
    if mode != 'all':
        svbrdf = copy.copy(svbrdf)
    if mode == 'diffuse_only':
        svbrdf.specular_map = zeros
    elif mode == 'specular_only':
        svbrdf.diffuse_map = zeros
    elif mode == 'light_map':
        gamma = None
        svbrdf.specular_map = zeros
        svbrdf.diffuse_map = ones

    if transpose:
        cam_up = (0.0, 0.0, -1.0)
    if transpose:
        cam_lookat = tuple(reversed(cam_lookat))

    if size is None:
        size = (width, height)
        cam_dist = cam_dist * min(width, height)/max(width, height)

    if camera is None:
        camera = ArcballCamera(
            size=size, fov=cam_fov, near=0.1, far=1000.0,
            position=[0, cam_dist, 0],
            lookat=(cam_lookat[0], 0.0, -cam_lookat[1]), up=cam_up)
    else:
        camera = camera

    plane_size = 100

    jsd = {
        "mesh": {
            "scale": 1,
            "type": "inline",
            "vertices": [
                [width/height*plane_size, 0.0, -plane_size],
                [width/height*plane_size, 0.0, plane_size],
                [-width/height*plane_size, 0.0, plane_size],
                [-width/height*plane_size, 0.0, -plane_size]
            ],
            "uvs": [
                [uv_scale*plane_size, 0.0],
                [uv_scale*plane_size, uv_scale*plane_size],
                [0.0, uv_scale*plane_size],
                [0.0, 0.0]
            ],
            "normals": [
                [0.0, 1.0, 0.0]
            ],
            "materials": ["plane"],
            "faces": [
                {
                    "vertices": [0, 1, 2],
                    "uvs": [0, 1, 2],
                    "normals": [0, 0, 0],
                    "material": 0
                },
                {
                    "vertices": [0, 2, 3],
                    "uvs": [0, 2, 3],
                    "normals": [0, 0, 0],
                    "material": 0
                }
            ]
        },
        "lights": lights,
        "materials": {
            "plane": {
                "type": "svbrdf_inline",
                "diffuse_map": svbrdf.diffuse_map,
                "specular_map": svbrdf.specular_map,
                "spec_shape_map": svbrdf.spec_shape_map,
                "normal_map": svbrdf.normal_map,
                "alpha": svbrdf.alpha,
            }
        }
    }
    if radmap is not None:
        if isinstance(radmap, np.ndarray):
            radmap = dict(type='inline', array=radmap)
        jsd['radiance_map'] = radmap
    return JSDRenderer(jsd, camera, size=size, gamma=gamma,
                       **kwargs)


def render_full(jsd_dict, **kwargs):
    with jsd.JSDRenderer(jsd_dict, **kwargs) as r:
        r.camera.clear_color = (0.0, 0.0, 0.0)
        im = r.render_to_image()
    return im


def render_diffuse_lightmap(jsd_dict, **kwargs):
    jsd_dict = copy.deepcopy(jsd_dict)
    for mat_name in jsd_dict['materials']:
        mat_jsd = jsd_dict['materials'][mat_name]
        if mat_jsd['type'] == 'svbrdf_inline':
            mat_jsd['diffuse_map'][:] = 1.0
            mat_jsd['specular_map'][:] = 0.0
        elif mat_jsd['type'] == 'beckmann_inline':
            mat_jsd['params']['diffuse_map'][:] = 1.0
            mat_jsd['params']['specular_map'][:] = 0.0
        else:
            raise RuntimeError('Unknown mat type.')
    with jsd.JSDRenderer(jsd_dict, **kwargs) as r:
        r.camera.clear_color = (0.0, 0.0, 0.0)
        return r.render_to_image()


def render_specular_lightmap(jsd_dict, **kwargs):
    jsd_dict = copy.deepcopy(jsd_dict)
    for mat_name in jsd_dict['materials']:
        mat_jsd = jsd_dict['materials'][mat_name]
        if mat_jsd['type'] == 'svbrdf_inline':
            mat_jsd['diffuse_map'][:] = 0.0
            mat_jsd['specular_map'][:] = 1.0
        elif mat_jsd['type'] == 'beckmann_inline':
            mat_jsd['params']['diffuse_map'][:] = 0.0
            mat_jsd['params']['specular_map'][:] = 1.0
        else:
            raise RuntimeError('Unknown mat type.')
    with jsd.JSDRenderer(jsd_dict, **kwargs) as r:
        r.camera.clear_color = (0.0, 0.0, 0.0)
        return r.render_to_image()


def render_diffuse_albedo(jsd_dict, **kwargs):
    jsd_dict = copy.deepcopy(jsd_dict)
    new_mat_jsd = {}
    for mat_name, mat_jsd in jsd_dict['materials'].items():
        if mat_jsd['type'] == 'svbrdf_inline':
            new_mat_jsd[mat_name] = dict(type='basic_texture',
                                         texture=mat_jsd['diffuse_map'])
        elif mat_jsd['type'] == 'beckmann_inline':
            new_mat_jsd[mat_name] = dict(
                type='basic_texture', texture=mat_jsd['params']['diffuse_map'])
        elif mat_jsd['type'] == 'blinn_phong':
            new_mat_jsd[mat_name] = dict(type='basic',
                                         color=mat_jsd['diffuse'])
        else:
            logger.error("Diffuse albedo does not exist!")
            new_mat_jsd[mat_name] = dict(type='basic',
                                         color=(1.0, 0.0, 1.0))
    jsd_dict['materials'] = new_mat_jsd
    with jsd.JSDRenderer(jsd_dict, **kwargs) as r:
        r.camera.clear_color = (0.0, 0.0, 0.0)
        return r.render_to_image()


def render_specular_albedo(jsd_dict, **kwargs):
    jsd_dict = copy.deepcopy(jsd_dict)
    new_mat_jsd = {}
    for mat_name, mat_jsd in jsd_dict['materials'].items():
        if mat_jsd['type'] == 'svbrdf_inline':
            new_mat_jsd[mat_name] = dict(type='basic_texture',
                                         texture=mat_jsd['specular_map'])
        if mat_jsd['type'] == 'beckmann_inline':
            new_mat_jsd[mat_name] = dict(
                type='basic_texture', texture=mat_jsd['params']['specular_map'])
        elif mat_jsd['type'] == 'blinn_phong':
            new_mat_jsd[mat_name] = dict(type='basic',
                                         color=mat_jsd['specular'])
        else:
            new_mat_jsd[mat_name] = dict(type='basic',
                                         color=(0.0, 0.0, 0.0))
    jsd_dict['materials'] = new_mat_jsd
    with jsd.JSDRenderer(jsd_dict, **kwargs) as r:
        r.camera.clear_color = (0.0, 0.0, 0.0)
        return r.render_to_image()


def render_diff_component(jsd_dict):
    jsd_dict = copy.deepcopy(jsd_dict)
    for mat_name in jsd_dict['materials']:
        mat_jsd = jsd_dict['materials'][mat_name]
        if mat_jsd['type'] == 'svbrdf_inline':
            mat_jsd['specular_map'][:] = 0.0
    with jsd.JSDRenderer(jsd_dict, ssaa=3, gamma=None) as r:
        r.camera.clear_color = (0.0, 0.0, 0.0)
        return r.render_to_image()


def render_spec_component(jsd_dict):
    jsd_dict = copy.deepcopy(jsd_dict)
    for mat_name in jsd_dict['materials']:
        mat_jsd = jsd_dict['materials'][mat_name]
        if mat_jsd['type'] == 'svbrdf_inline':
            mat_jsd['diffuse_map'][:] = 0.0
    with jsd.JSDRenderer(jsd_dict, ssaa=3, gamma=None) as r:
        r.camera.clear_color = (0.0, 0.0, 0.0)
        return r.render_to_image()


def render_jsd(jsd_dict, format='rgb', **rend_opts):
    with jsd.JSDRenderer(jsd_dict, **rend_opts) as renderer:
        image = renderer.render_to_image(format=format)
    return image


def make_jsd(mesh, camera, clear_color=(1.0, 1.0, 1.0)):
    camera = copy.deepcopy(camera)
    camera.clear_color = clear_color
    jsd_dict = {
        "camera": camera.serialize(),
        "lights": [],
        "mesh": jsd.export_mesh_to_jsd(mesh),
        "materials": {key: {'type': 'depth'} for key in mesh.materials}
    }
    return jsd_dict


def render_depth(mesh, camera):
    jsd_dict = make_jsd(mesh, camera, clear_color=(0.0, 0.0, 0.0))
    jsd_dict["materials"] = {key: {'type': 'depth'} for key in mesh.materials}
    image = render_jsd(jsd_dict)[:, :, 0]
    return image


def render_mesh_normals(mesh, camera):
    jsd_dict = make_jsd(mesh, camera, clear_color=(0.0, 0.0, 0.0))
    jsd_dict["materials"] = {key: {'type': 'normal'} for key in mesh.materials}
    image = render_jsd(jsd_dict)
    return image


def render_tangents(mesh, camera):
    jsd_dict = make_jsd(mesh, camera, clear_color=(0.0, 0.0, 0.0))
    jsd_dict["materials"] = {key: {'type': 'tangent'} for key in mesh.materials}
    image = render_jsd(jsd_dict)
    return image


def render_bitangents(mesh, camera):
    jsd_dict = make_jsd(mesh, camera, clear_color=(0.0, 0.0, 0.0))
    jsd_dict["materials"] = {key: {'type': 'bitangent'} for key in mesh.materials}
    image = render_jsd(jsd_dict)
    return image

def render_part_segments(mesh: Mesh, camera,
                    segment_type='material') -> np.ndarray:
    if segment_type == 'material':
        segments = mesh.materials
        # print(segments)
    elif segment_type == 'group':
        segments = mesh.group_names
    elif segment_type == 'object':
        segments = mesh.object_names
    else:
        raise RuntimeError("Unknown segment type")

    mesh_jsd = jsd.export_mesh_to_jsd(mesh)
    mesh_jsd["materials"] = segments
    for face in mesh_jsd["faces"]:
        face["material"] = face[segment_type]

    camera = copy.deepcopy(camera)
    camera.clear_color = (0, 0, 0)
    jsd_dict = {
        "camera": camera.serialize(),
        "mesh": mesh_jsd,
        "lights": [],
        # define a certain color for different materials.
        "materials": 
        {'1_a': {'type': 'basic', 'color': [1.0, 1.0, 1.0]}, 
        '1_b': {'type': 'basic', 'color': [2.0, 2.0, 2.0]}, 
        '1_c': {'type': 'basic', 'color': [3.0, 3.0, 3.0]},
        '1_d': {'type': 'basic', 'color': [4.0, 4.0, 4.0]}}

    }
    # print(jsd_dict["materials"])
    image = render_jsd(jsd_dict)
    segment_image = (image - 1).astype(int)[:, :, 0]
    return segment_image


def render_segments(mesh: Mesh, camera,
                    segment_type='material') -> np.ndarray:
    if segment_type == 'material':
        segments = mesh.materials
    elif segment_type == 'group':
        segments = mesh.group_names
    elif segment_type == 'object':
        segments = mesh.object_names
    else:
        raise RuntimeError("Unknown segment type")

    mesh_jsd = jsd.export_mesh_to_jsd(mesh)
    mesh_jsd["materials"] = segments
    for face in mesh_jsd["faces"]:
        face["material"] = face[segment_type]

    camera = copy.deepcopy(camera)
    camera.clear_color = (0, 0, 0)
    jsd_dict = {
        "camera": camera.serialize(),
        "mesh": mesh_jsd,
        "lights": [],
        "materials": {
            key: {
                'type': 'basic',
                'color': np.full((3,), i+1, dtype=np.float32).tolist()
            } for i, key in enumerate(segments)}
    }
    image = render_jsd(jsd_dict)
    segment_image = (image - 1).astype(int)[:, :, 0]
    return segment_image

def render_segments_partnet(mesh: Mesh, camera,
                    segment_type='material') -> np.ndarray:
    if segment_type == 'material':
        segments = mesh.materials
    elif segment_type == 'group':
        segments = mesh.group_names
    elif segment_type == 'object':
        segments = mesh.object_names
    else:
        raise RuntimeError("Unknown segment type")

    mesh_jsd = jsd.export_mesh_to_jsd(mesh)
    mesh_jsd["materials"] = segments
    for face in mesh_jsd["faces"]:
        face["material"] = face[segment_type]

    camera = copy.deepcopy(camera)
    camera.clear_color = (0, 0, 0)
    jsd_dict = {
        "camera": camera.serialize(),
        "mesh": mesh_jsd,
        "lights": [],
        # define a certain color for different materials.
        "materials": 
        {'material_0': {'type': 'basic', 'color': [1.0, 1.0, 1.0]},
        'material_1': {'type': 'basic', 'color': [2.0, 2.0, 2.0]}, 
        'material_2': {'type': 'basic', 'color': [3.0, 3.0, 3.0]}, 
        'material_3': {'type': 'basic', 'color': [4.0, 4.0, 4.0]},
        'material_4': {'type': 'basic', 'color': [5.0, 5.0, 5.0]},
        'material_5': {'type': 'basic', 'color': [6.0, 6.0, 6.0]},
        'material_6': {'type': 'basic', 'color': [7.0, 7.0, 7.0]},
        'material_7': {'type': 'basic', 'color': [8.0, 8.0, 8.0]},
        'material_8': {'type': 'basic', 'color': [9.0, 9.0, 9.0]},
        'material_9': {'type': 'basic', 'color': [10.0, 10.0, 10.0]},
        'material_10': {'type': 'basic', 'color': [11.0, 11.0, 11.0]},
        'material_11': {'type': 'basic', 'color': [12.0, 12.0, 12.0]},
        'material_12': {'type': 'basic', 'color': [13.0, 13.0, 13.0]},
        'material_13': {'type': 'basic', 'color': [14.0, 14.0, 14.0]},
        'material_14': {'type': 'basic', 'color': [15.0, 15.0, 15.0]},
        'material_15': {'type': 'basic', 'color': [16.0, 16.0, 16.0]},
        'material_16': {'type': 'basic', 'color': [17.0, 17.0, 17.0]},
        'material_17': {'type': 'basic', 'color': [18.0, 18.0, 18.0]},
        'material_18': {'type': 'basic', 'color': [19.0, 19.0, 19.0]},
        'material_19': {'type': 'basic', 'color': [20.0, 20.0, 20.0]},
        'material_20': {'type': 'basic', 'color': [21.0, 21.0, 21.0]},
        'material_21': {'type': 'basic', 'color': [22.0, 22.0, 22.0]},
        'material_22': {'type': 'basic', 'color': [23.0, 23.0, 23.0]},
        'material_23': {'type': 'basic', 'color': [24.0, 24.0, 24.0]},
        'material_24': {'type': 'basic', 'color': [25.0, 25.0, 25.0]},
        'material_25': {'type': 'basic', 'color': [26.0, 26.0, 26.0]},
        'material_26': {'type': 'basic', 'color': [27.0, 27.0, 27.0]},
        'material_27': {'type': 'basic', 'color': [28.0, 28.0, 28.0]},
        'material_28': {'type': 'basic', 'color': [29.0, 29.0, 29.0]},
        'material_29': {'type': 'basic', 'color': [30.0, 30.0, 30.0]},
        'material_30': {'type': 'basic', 'color': [31.0, 31.0, 31.0]},
        'material_31': {'type': 'basic', 'color': [32.0, 32.0, 32.0]},
        'material_32': {'type': 'basic', 'color': [33.0, 33.0, 33.0]},
        'material_33': {'type': 'basic', 'color': [34.0, 34.0, 34.0]},
        'material_34': {'type': 'basic', 'color': [35.0, 35.0, 35.0]},
        'material_35': {'type': 'basic', 'color': [36.0, 36.0, 36.0]},
        'material_36': {'type': 'basic', 'color': [37.0, 37.0, 37.0]},
        'material_37': {'type': 'basic', 'color': [38.0, 38.0, 38.0]},
        'material_38': {'type': 'basic', 'color': [39.0, 39.0, 39.0]},
        'material_39': {'type': 'basic', 'color': [40.0, 40.0, 40.0]},
        'material_40': {'type': 'basic', 'color': [41.0, 41.0, 41.0]},
        'material_41': {'type': 'basic', 'color': [42.0, 42.0, 42.0]},
        'material_42': {'type': 'basic', 'color': [43.0, 43.0, 43.0]},
        'material_43': {'type': 'basic', 'color': [44.0, 44.0, 44.0]},
        'material_44': {'type': 'basic', 'color': [45.0, 45.0, 45.0]},
        'material_45': {'type': 'basic', 'color': [46.0, 46.0, 46.0]},
        'material_46': {'type': 'basic', 'color': [47.0, 47.0, 47.0]},
        'material_47': {'type': 'basic', 'color': [48.0, 48.0, 48.0]},
        'material_48': {'type': 'basic', 'color': [49.0, 49.0, 49.0]},
        'material_49': {'type': 'basic', 'color': [50.0, 50.0, 50.0]},
        'material_50': {'type': 'basic', 'color': [51.0, 51.0, 51.0]},
        'material_51': {'type': 'basic', 'color': [52.0, 52.0, 52.0]},
        'material_52': {'type': 'basic', 'color': [53.0, 53.0, 53.0]},
        'material_53': {'type': 'basic', 'color': [54.0, 54.0, 54.0]},
        'material_54': {'type': 'basic', 'color': [55.0, 55.0, 55.0]},
        'material_55': {'type': 'basic', 'color': [56.0, 56.0, 56.0]},
        'material_56': {'type': 'basic', 'color': [57.0, 57.0, 57.0]},
        'material_57': {'type': 'basic', 'color': [58.0, 58.0, 58.0]},
        'material_58': {'type': 'basic', 'color': [59.0, 59.0, 59.0]},
        'material_59': {'type': 'basic', 'color': [60.0, 60.0, 60.0]},
        'material_60': {'type': 'basic', 'color': [61.0, 61.0, 61.0]},
        'material_61': {'type': 'basic', 'color': [62.0, 62.0, 62.0]},
        'material_62': {'type': 'basic', 'color': [63.0, 63.0, 63.0]},
        'material_63': {'type': 'basic', 'color': [64.0, 64.0, 64.0]},
        'material_64': {'type': 'basic', 'color': [65.0, 65.0, 65.0]},
        'material_65': {'type': 'basic', 'color': [66.0, 66.0, 66.0]},
        'material_66': {'type': 'basic', 'color': [67.0, 67.0, 67.0]},
        'material_67': {'type': 'basic', 'color': [68.0, 68.0, 68.0]},
        'material_68': {'type': 'basic', 'color': [69.0, 69.0, 69.0]},
        'material_69': {'type': 'basic', 'color': [70.0, 70.0, 70.0]},
        'material_70': {'type': 'basic', 'color': [71.0, 71.0, 71.0]},
        'material_71': {'type': 'basic', 'color': [72.0, 72.0, 72.0]},
        'material_72': {'type': 'basic', 'color': [73.0, 73.0, 73.0]},
        'material_73': {'type': 'basic', 'color': [74.0, 74.0, 74.0]},
        'material_74': {'type': 'basic', 'color': [75.0, 75.0, 75.0]},
        'material_75': {'type': 'basic', 'color': [76.0, 76.0, 76.0]},
        'material_76': {'type': 'basic', 'color': [77.0, 77.0, 77.0]},
        'material_77': {'type': 'basic', 'color': [78.0, 78.0, 78.0]},
        'material_78': {'type': 'basic', 'color': [79.0, 79.0, 79.0]},
        'material_79': {'type': 'basic', 'color': [80.0, 80.0, 80.0]},
        'material_80': {'type': 'basic', 'color': [81.0, 81.0, 81.0]},
        'material_81': {'type': 'basic', 'color': [82.0, 82.0, 82.0]},
        'material_82': {'type': 'basic', 'color': [83.0, 83.0, 83.0]},
        }
        
    }
    # print(jsd_dict["materials"])
    image = render_jsd(jsd_dict)
    segment_image = (image - 1).astype(int)[:, :, 0]
    return segment_image


def render_wavefront_mtl(mesh: Mesh, camera,
                         materials: Dict[str, WavefrontMaterial],
                         radmap_path=None,
                         **rend_opts):
    jsd_dict = make_jsd(mesh, camera)
    jsd_dict["materials"] = {
        mtl.name: {
            'type': 'blinn_phong',
            'diffuse': mtl.diffuse_color,
            'specular': mtl.specular_color,
            'roughness': 1.0 / math.sqrt((mtl.specular_exponent + 2) / 2.0),
        } for mtl_name, mtl in materials.items()
    }
    if radmap_path is not None:
        jsd_dict['radiance_map'] = {
            'path': radmap_path
        }
    return render_jsd(jsd_dict, **rend_opts)


def render_material_dicts(mesh: Mesh, camera,
                     materials: Dict[str, dict],
                     radmap_path=None,
                     format='rgb',
                     **rend_opts):
    jsd_dict = make_jsd(mesh, camera)
    jsd_dict["materials"] = {
        mtl_name: mtl for mtl_name, mtl in materials.items()
    }
    if radmap_path is not None:
        jsd_dict['radiance_map'] = {
            'path': radmap_path
        }
    return render_jsd(jsd_dict, format=format, **rend_opts)


def render_preview(mesh: Mesh, camera,
                   radmap_path=None,
                   **rend_opts):
    jsd_dict = make_jsd(mesh, camera)
    jsd_dict["materials"] = {
        mtl_name: {
            'type': 'blinn_phong',
            'diffuse': tuple(c/255/1.2
                             for c in QUAL_COLORS[mtl_id % len(QUAL_COLORS)]),
            'specular': (0.044, 0.044, 0.044),
            'roughness': 0.1,
        } for mtl_id, mtl_name in enumerate(mesh.materials)
    }
    if radmap_path is not None:
        jsd_dict['radiance_map'] = {
            'path': radmap_path
        }
    return render_jsd(jsd_dict, **rend_opts)


def render_world_coords(mesh, camera):
    jsd_dict = make_jsd(mesh, camera)
    jsd_dict["materials"] = {key: {'type': 'world'} for key in mesh.materials}
    return render_jsd(jsd_dict)


def render_median_colors(mesh, image, camera):
    pixel_segment_ids = render_segments(mesh, camera)
    median_colors = images.compute_segment_median_colors(
        image, pixel_segment_ids)
    median_image = np.ones(image.shape)
    for segment_id in range(len(median_colors)):
        mask = pixel_segment_ids == segment_id
        median_image[mask, :] = median_colors[segment_id, :]

    return median_image


def render_uvs(mesh, camera):
    if len(mesh.uvs) == 0:
        raise RuntimeError('Mesh does not have UVs')
    jsd_dict = make_jsd(mesh, camera, clear_color=(0, 0, 0, 0))
    jsd_dict["materials"] = {key: {'type': 'uv'} for key in mesh.materials}
    return render_jsd(jsd_dict)[:, :, :2]
