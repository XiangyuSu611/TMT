import copy
import logging
from typing import Dict, List, Union

import numpy as np
from thirdparty.rendkit import rendkit as rendkit
from thirdparty.rendkit.rendkit import materials
from thirdparty.rendkit.meshkit import Mesh
from thirdparty.rendkit.meshkit import wavefront
from thirdparty.rendkit.rendkit import shapes
from thirdparty.rendkit.rendkit.envmap import EnvironmentMap
from thirdparty.rendkit.rendkit.envmap.io import load_envmap
from thirdparty.rendkit.rendkit.lights import Light, PointLight, DirectionalLight
from thirdparty.rendkit.rendkit.materials import (GLSLProgram, AittalaMaterial, BlinnPhongMaterial,
                               BasicMaterial, NormalMaterial,
                               WorldCoordMaterial,
                               DepthMaterial, UVMaterial, UnwrapToUVMaterial,
                               TangentMaterial, BitangentMaterial,
                               BasicTextureMaterial, BeckmannMaterial)
from thirdparty.rendkit.rendkit.renderers import SceneRenderer
from thirdparty.rendkit.rendkit.scene import Scene
from thirdparty.rendkit.svbrdf.aittala import AittalaSVBRDF
from thirdparty.rendkit.svbrdf.beckmann import BeckmannSVBRDF
from .camera import CalibratedCamera, PerspectiveCamera, ArcballCamera

logger = logging.getLogger(__name__)


class JSDRenderer(SceneRenderer):
    def __init__(self, jsd_dict, camera=None, show_floor=False,
                 shadows=False, *args, **kwargs):
        if camera is None:
            camera = import_jsd_camera(jsd_dict)
        scene = import_jsd_scene(jsd_dict, show_floor, shadows)
        super().__init__(scene=scene, camera=camera, *args, **kwargs)


def import_jsd_scene(jsd_dict, show_floor=False, shadows=False):
    scene = Scene(lights=import_jsd_lights(jsd_dict),
                         materials=import_jsd_materials(jsd_dict))
    scene.add_mesh(import_jsd_mesh(jsd_dict["mesh"]))

    if show_floor:
        floor_pos = scene.meshes[0].vertices[:, 1].min()
        floor_mesh = shapes.make_plane(10000, 10000, 'floor')
        floor_mesh.name = 'floor'
        scene.add_mesh(floor_mesh, (0, floor_pos, 0))
        scene.put_material(
            'floor', BlinnPhongMaterial((1.0, 1.0, 1.0),
                                        (0.0, 0.0, 0.0), 0.0))

    if 'radiance_map' in jsd_dict:
        scene.set_radiance_map(import_radiance_map(
            jsd_dict['radiance_map']),
            add_shadows=shadows)
    return scene


def import_jsd_camera(jsd_dict):
    if 'camera' in jsd_dict:
        jsd_cam = jsd_dict['camera']
        type = jsd_cam['type']
        clear_color = jsd_cam.get('clear_color', (1.0, 1.0, 1.0))
        if type == 'perspective':
            return PerspectiveCamera(
                size=jsd_cam['size'],
                near=jsd_cam['near'],
                far=jsd_cam['far'],
                fov=jsd_cam['fov'],
                position=jsd_cam['position'],
                lookat=jsd_cam['lookat'],
                up=jsd_cam['up'],
                clear_color=clear_color)
        elif type == 'arcball':
            return ArcballCamera(
                size=jsd_cam['size'],
                near=jsd_cam['near'],
                far=jsd_cam['far'],
                fov=jsd_cam['fov'],
                position=jsd_cam['position'],
                lookat=jsd_cam['lookat'],
                up=jsd_cam['up'],
                clear_color=clear_color)
        elif type == 'calibrated':
            return CalibratedCamera(
                size=jsd_cam['size'],
                near=jsd_cam['near'],
                far=jsd_cam['far'],
                extrinsic=np.array(jsd_cam['extrinsic']),
                intrinsic=np.array(jsd_cam['intrinsic']),
                clear_color=clear_color)
        else:
            raise RuntimeError('Unknown camera type {}'.format(type))

    logger.warning('Camera undefined, returning default camera.')
    return ArcballCamera(
        size=(1024, 768), fov=75, near=10, far=1000.0,
        position=[50, 50, 50],
        lookat=(0.0, 0.0, -0.0),
        up=(0.0, 1.0, 0.0))


def import_jsd_lights(jsd_dict) -> List[Light]:
    jsd_lights = jsd_dict.get('lights', None)
    if jsd_lights is None:
        return []

    lights = []
    for jsd_light in jsd_lights:
        lights.append(import_jsd_light(jsd_light))

    return lights


def import_radiance_map(jsd_radmap) -> Union[EnvironmentMap, None]:
    scale = jsd_radmap['scale'] if ('scale' in jsd_radmap) else 1.0

    if 'path' in jsd_radmap:
        path = jsd_radmap['path']
        logger.info('Importing radiance map from {} with scale={}'
                    .format(path, scale))
        cube_faces = load_envmap(path)
    elif jsd_radmap['type'] == 'inline':
        logger.info("Importing inline radiance map with shape={}".format(
            jsd_radmap['array'].shape))
        cube_faces = np.array(jsd_radmap['array'], dtype=np.float32)
    else:
        raise RuntimeError('Unknown radiance map type {}!'.format(
            jsd_radmap['type']))
    assert cube_faces.shape[0] == 6
    if 'max' in jsd_radmap:
        cube_faces = np.clip(cube_faces, 0, jsd_radmap['max'])
    logger.info('Radiance range: ({}, {})'
                .format(cube_faces.min(), cube_faces.max()))
    print('ready to return environmentmap')
    return EnvironmentMap(cube_faces, scale)


def import_jsd_light(jsd_light) -> Light:
    if jsd_light['type'] == 'point':
        return PointLight(jsd_light['position'],
                          jsd_light['intensity'])
    elif jsd_light['type'] == 'directional':
        return DirectionalLight(jsd_light['position'],
                                jsd_light['intensity'])
    else:
        raise RuntimeError('Unknown light type {}'.format(jsd_light['type']))


def list_material_names(jsd_dict):
    return [m for m in jsd_dict['materials'].keys()]


def import_jsd_materials(jsd_dict) -> Dict[str, GLSLProgram]:
    materials = {}
    for name, jsd_material in jsd_dict["materials"].items():
        materials[name] = import_jsd_material(jsd_material)
    return materials


def import_jsd_material(jsd_material) -> rendkit.materials.GLSLProgram:
    mat_type = jsd_material['type']
    if mat_type == 'svbrdf' or mat_type == 'aittala':
        transposed = False
        if 'transposed' in jsd_material:
            transposed = bool(jsd_material['transposed'])
        return AittalaMaterial(AittalaSVBRDF(jsd_material['path'], transposed=transposed))
    elif mat_type == 'beckmann':
        return BeckmannMaterial(BeckmannSVBRDF.from_path(jsd_material['path']))
    elif mat_type == 'beckmann_inline':
        return BeckmannMaterial(BeckmannSVBRDF(**jsd_material['params']))
    elif mat_type == 'basic':
        return BasicMaterial(jsd_material['color'])
    elif mat_type == 'svbrdf_inline':
        return AittalaMaterial(AittalaSVBRDF(**jsd_material['params']))
    elif mat_type == 'basic_texture':
        return BasicTextureMaterial(jsd_material['texture'])
    elif mat_type == 'blinn_phong':
        return BlinnPhongMaterial(
            jsd_material['diffuse'],
            jsd_material['specular'],
            jsd_material['roughness'])
    elif mat_type == 'uv':
        return UVMaterial()
    elif mat_type == 'depth':
        return DepthMaterial()
    elif mat_type == 'normal':
        return NormalMaterial()
    elif mat_type == 'tangent':
        return TangentMaterial()
    elif mat_type == 'bitangent':
        return BitangentMaterial()
    elif mat_type == 'world':
        return WorldCoordMaterial()
    elif mat_type == 'unwrap_to_uv':
        return UnwrapToUVMaterial(
            jsd_material['image'],
            jsd_material['depth'])


def import_jsd_mesh(jsd_mesh):
    if jsd_mesh['type'] == 'wavefront':
        mesh = wavefront.read_obj_file(jsd_mesh['path'])
    elif jsd_mesh['type'] == 'inline':
        vertices = jsd_mesh['vertices']
        normals = jsd_mesh['normals'] if 'normals' in jsd_mesh else []
        uvs = jsd_mesh['uvs'] if 'uvs' in jsd_mesh else []
        mesh = Mesh(np.array(vertices),
                    np.array(normals),
                    np.array(uvs),
                    jsd_mesh['faces'],
                    jsd_mesh['materials'], [], [])
    else:
        raise RuntimeError("Unknown mesh type {}".format(jsd_mesh['type']))

    if 'size' in jsd_mesh:
        mesh.resize(jsd_mesh['size'])
    elif 'scale' in jsd_mesh:
        mesh.rescale(jsd_mesh['scale'])

    if 'uv_scale' in jsd_mesh:
        uv_scale = float(jsd_mesh['uv_scale'])
        logger.info("UV scale is set to {:.04f} for mesh"
                    .format(uv_scale))
        mesh.uv_scale = uv_scale

    return mesh


def export_mesh_to_jsd(mesh: Mesh):
    return {
        "size": mesh.size,
        "type": "inline",
        "vertices": mesh.vertices.tolist(),
        "uvs": mesh.uvs.tolist() if len(mesh.uvs) > 0 else [],
        "normals": mesh.normals.tolist(),
        "materials": list(mesh.materials),
        "faces": copy.copy(mesh.faces),
    }


def cache_inline(jsd_dict):
    new_dict = copy.deepcopy(jsd_dict)
    for mat_name, jsd_mat in new_dict['materials'].items():
        if jsd_mat['type'] == 'svbrdf':
            brdf = AittalaSVBRDF(jsd_mat['path'])
            new_dict['materials'][mat_name] = brdf.to_jsd()
    return new_dict
