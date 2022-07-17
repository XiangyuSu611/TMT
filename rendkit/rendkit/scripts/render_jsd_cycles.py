from pathlib import Path

import bpy
import json

from meshkit import wavefront
import argparse

from svbrdf import beckmann
from toolbox.logging import init_logger

logger = init_logger(__name__)


def reset_blender():
    bpy.ops.wm.read_factory_settings()

    for scene in bpy.data.scenes:
        for obj in scene.objects:
            scene.objects.unlink(obj)

    # only worry about data in the startup scene
    for bpy_data_iter in (
            bpy.data.objects,
            bpy.data.meshes,
            bpy.data.lamps,
            bpy.data.cameras,
    ):
        for id_data in bpy_data_iter:
            bpy_data_iter.remove(id_data)


def look_at(obj_camera, point):
    loc_camera = obj_camera.matrix_world.to_translation()

    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')

    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()


def assign_material(jsd_mat, bpy_mat):
    if jsd_mat['type'] != 'beckmann':
        logger.error("Only beckmann is supported.")
        return

    brdf_path = Path(jsd_mat['path'])
    logger.info("Assigning {}".format(jsd_mat['path']))
    diff_tex_path = brdf_path / beckmann.DIFF_MAP_NAME
    spec_tex_path = brdf_path / beckmann.SPEC_MAP_NAME
    normal_tex_path = brdf_path / beckmann.BLEND_NORMAL_MAP_NAME
    rough_tex_path = brdf_path / beckmann.ROUGH_MAP_NAME
    aniso_tex_path = brdf_path / beckmann.ANISO_MAP_NAME

    bpy_mat.use_nodes = True
    nodes = bpy_mat.node_tree.nodes
    links = bpy_mat.node_tree.links

    tex_coord_node = nodes.new(type="ShaderNodeTexCoord")
    mapping_node = nodes.new(type="ShaderNodeMapping")
    mapping_node.vector_type = 'TEXTURE'
    mapping_node.scale[0] = 0.1
    mapping_node.scale[1] = 0.1
    mapping_node.scale[2] = 0.1
    links.new(mapping_node.inputs[0], tex_coord_node.outputs[2])

    normal_tex_node = nodes.new(type="ShaderNodeTexImage")
    normal_tex_node.color_space = 'NONE'
    normal_tex_node.image = bpy.data.images.load(filepath=str(normal_tex_path))
    normal_map_node = nodes.new(type="ShaderNodeNormalMap")
    normal_map_node.space = 'TANGENT'
    links.new(normal_tex_node.inputs[0], mapping_node.outputs[0])
    links.new(normal_map_node.inputs[1], normal_tex_node.outputs[0])

    diff_tex_node = nodes.new(type="ShaderNodeTexImage")
    diff_tex_node.image = bpy.data.images.load(filepath=str(diff_tex_path))
    links.new(diff_tex_node.inputs[0], mapping_node.outputs[0])

    spec_tex_node = nodes.new(type="ShaderNodeTexImage")
    spec_tex_node.image = bpy.data.images.load(filepath=str(spec_tex_path))
    links.new(spec_tex_node.inputs[0], mapping_node.outputs[0])

    aniso_tex_node = nodes.new(type="ShaderNodeTexImage")
    aniso_tex_node.color_space = 'NONE'
    aniso_tex_node.image = bpy.data.images.load(filepath=str(aniso_tex_path))
    links.new(aniso_tex_node.inputs[0], mapping_node.outputs[0])

    rough_tex_node = nodes.new(type="ShaderNodeTexImage")
    rough_tex_node.color_space = 'NONE'
    rough_tex_node.image = bpy.data.images.load(filepath=str(rough_tex_path))
    links.new(rough_tex_node.inputs[0], mapping_node.outputs[0])

    diff_bsdf_node = nodes.new(type="ShaderNodeBsdfDiffuse")
    links.new(diff_bsdf_node.inputs[0], diff_tex_node.outputs[0])
    links.new(diff_bsdf_node.inputs[2], normal_map_node.outputs[0])

    spec_bsdf_node = nodes.new(type="ShaderNodeBsdfAnisotropic")
    spec_bsdf_node.distribution = 'BECKMANN'
    links.new(spec_bsdf_node.inputs[0], spec_tex_node.outputs[0])
    links.new(spec_bsdf_node.inputs[1], rough_tex_node.outputs[0])
    links.new(spec_bsdf_node.inputs[2], aniso_tex_node.outputs[0])
    links.new(spec_bsdf_node.inputs[4], normal_map_node.outputs[0])

    mix_node = nodes.new(type="ShaderNodeMixShader")
    links.new(mix_node.inputs[1], spec_bsdf_node.outputs[0])
    links.new(mix_node.inputs[2], diff_bsdf_node.outputs[0])

    output_node = nodes["Material Output"]
    links.new(output_node.inputs[0], mix_node.outputs[0])


def set_envmap(path):
    path = path.replace('cross', 'pano')
    logger.info("Setting envmap to {}".format(str(path)))
    scene = bpy.context.scene
    scene.world.use_nodes = True
    nodes = scene.world.node_tree.nodes
    links = scene.world.node_tree.links
    env_image = bpy.data.images.load(filepath=path)
    env_tex_node = nodes.new(type="ShaderNodeTexEnvironment")
    env_tex_node.image = env_image
    background_node = nodes.new(type="ShaderNodeBackground")
    links.new(background_node.inputs[0], env_tex_node.outputs[0])
    output_node = nodes['World Output']
    links.new(output_node.inputs[0], background_node.outputs[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='jsd_path', type=str)
    parser.add_argument('--rend-path', dest='rend_path', type=str,
                        required=False)
    parser.add_argument('--blend-path', dest='blend_path', type=str,
                        required=False)
    args = parser.parse_args()

    with open(args.jsd_path, 'r') as f:
        jsd_dict = json.load(f)

    reset_blender()

    logger.info("Importing mesh {}".format(jsd_dict['mesh']['path']))
    # mesh = wavefront.read_obj_file(jsd_dict['mesh']['path'])
    # mesh.resize(1)
    # mesh_tmp_path = '/tmp/_jsd_to_cycles.tmp.obj'
    # wavefront.save_obj_file(mesh_tmp_path, mesh)

    bpy.ops.import_scene.obj(filepath=jsd_dict['mesh']['path'],
                             use_edges=False, use_smooth_groups=False,
                             use_split_objects=False, use_split_groups=False,
                             use_groups_as_vgroups=False,
                             use_image_search=True)

    bpy.ops.object.camera_add()
    scene = bpy.context.scene
    set_envmap(jsd_dict['radiance_map']['path'])

    scene.render.resolution_x = 2000
    scene.render.resolution_y = 2000
    scene.camera = bpy.context.object
    scene.camera.location = (0.60, 1.0, 0.50)
    scene.camera.rotation_euler = (1.109, 0, 2.617)
    if scene.render.engine != 'CYCLES':
        logger.info("Setting renderer engine {} -> CYCLES"
                    .format(scene.render.engine))
        scene.render.engine = 'CYCLES'

    for bpy_mat in bpy.data.materials:
        mat_name = bpy_mat.name
        if mat_name in jsd_dict['materials']:
            logger.info("Processing material {}".format(mat_name))
            assign_material(jsd_dict['materials'][mat_name], bpy_mat)

    floor_pos = 0
    for bpy_mesh in bpy.data.meshes:
        floor_pos = min(floor_pos, min(p.co.y for p in bpy_mesh.vertices))
    logger.info("Making floor plane at {}".format(floor_pos))
    bpy.ops.mesh.primitive_plane_add(radius=100, location=(0, 0, floor_pos))

    if args.rend_path:
        scene.cycles.device = 'GPU'
        scene.cycles.samples = 128
        prefs = bpy.context.user_preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        prefs.devices[0].use = True
        bpy.data.scenes['Scene'].render.filepath = args.rend_path
        bpy.ops.render.render(write_still=True)

    if args.blend_path:
        bpy.ops.wm.save_as_mainfile(filepath=args.blend_path)


if __name__ == '__main__':
    main()
