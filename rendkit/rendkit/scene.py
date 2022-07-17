from collections import OrderedDict
from typing import List, Dict

import numpy as np
from scipy import misc

from thirdparty.rendkit.meshkit import Mesh
from thirdparty.rendkit.rendkit import vector_utils, util
from thirdparty.rendkit.rendkit.camera import OrthographicCamera
from thirdparty.rendkit.rendkit.core import logger, mesh_to_renderables, ContextProvider, draw_depth
from thirdparty.rendkit.rendkit.envmap import EnvironmentMap
from thirdparty.rendkit.rendkit.envmap.prefilter import find_shadow_sources
from thirdparty.rendkit.rendkit.glsl import GLSLProgram
from thirdparty.rendkit.rendkit.lights import Light
from thirdparty.rendkit.rendkit.materials import PLACEHOLDER_MATERIAL
from vispy import gloo
# from thirdparty.vispy.vispy import gloo
from vispy.util import transforms
# from thirdparty.vispy.vispy.util import transforms


class Scene:
    def __init__(self,
                 lights: List[Light]=None,
                 materials: Dict[str, GLSLProgram]=None):
        self.lights = [] if lights is None else lights
        self.radiance_map = None
        self.materials = {} if materials is None else materials
        self.meshes = []
        self.renderables_by_mesh = OrderedDict()
        self.shadow_sources = []
        self._version = 0

    def reset(self):
        for material in self.materials.values():
            material._instances = []
            material.init_uniforms()
        for renderable in self.renderables:
            renderable._program = None
        if self.radiance_map:
            self.radiance_map.reset()

    def get_material(self, name):
        if name in self.materials:
            return self.materials[name]
        logger.info('Material {} is not defined! Rendering with'
                       ' placeholder'.format(name))
        return PLACEHOLDER_MATERIAL

    def put_material(self, name: str, material: GLSLProgram):
        self.materials[name] = material
        self.mark_updated()

    def remove_material(self, name):
        material = self.materials[name]
        material.clear()
        del self.materials[name]
        self.mark_updated()

    def add_light(self, light: Light):
        self.lights.append(light)
        self.mark_updated()

    def set_radiance_map(self, radiance_map, add_shadows=False):
        if radiance_map is None:
            logger.error("Radiance map is None.")
            return
        self.radiance_map = radiance_map

        if add_shadows:
            self._init_radmap_shadows()
        self.mark_updated()

    def _init_radmap_shadows(self):
        shadow_dirs = find_shadow_sources(
            self.radiance_map.radiance_faces)
        logger.info("[shadows ON] Rendering {} shadow maps."
                    .format(len(shadow_dirs)))
        with ContextProvider((1024, 1024)):
            for i, shadow_dir in enumerate(shadow_dirs):
                position = vector_utils.normalized(shadow_dir)
                up = np.roll(position, 1) * (1, 1, -1)
                camera = OrthographicCamera(
                    (200, 200), -150, 150, position=position,
                    lookat=(0, 0, 0), up=up)
                rend_target = util.create_rend_target((1024, 1024))
                draw_depth(camera, self.renderables, rend_target)
                with rend_target[0]:
                    # gloo.read_pixels flips the output.
                    depth = np.flipud(np.copy(gloo.read_pixels(
                        format='depth', out_type=np.float32)))
                    # Opengl ES doesn't support border clamp value
                    # specification so hack it like this.
                    depth[[0, -1], :] = 1.0
                    depth[:, [0, -1]] = 1.0
                self.shadow_sources.append((camera, depth))

    def add_mesh(self, mesh: Mesh, position=(0, 0, 0)):
        model_mat = transforms.translate(position).T
        self.meshes.append(mesh)
        self.renderables_by_mesh[mesh] = mesh_to_renderables(mesh, model_mat)
        self.mark_updated()

    def remove_mesh(self, mesh: Mesh):
        self.meshes.remove(mesh)
        del self.renderables_by_mesh[mesh]
        self.mark_updated()

    def clear_meshes(self):
        self.meshes.clear()
        self.renderables_by_mesh.clear()
        self.mark_updated()

    def clear_materials(self):
        for material in self.materials.values():
            material.clear()
        self.materials.clear()
        self.mark_updated()

    def clear(self):
        self.clear_meshes()
        self.clear_materials()

    def set_mesh_transform(self, mesh: Mesh, transform_mat: np.ndarray,
                           apply_to_existing=False):
        if transform_mat.shape != (4, 4):
            raise ValueError("Invalid transformation matrix (must be 4x4).")
        for renderable in self.renderables_by_mesh[mesh]:
            if apply_to_existing:
                renderable.model_mat = transform_mat @ renderable.model_mat
            else:
                renderable.model_mat = transform_mat

    @property
    def renderables(self):
        result = []
        for mesh_renderables in self.renderables_by_mesh.values():
            for renderable in mesh_renderables:
                result.append(renderable)
        return result

    @property
    def version(self):
        return self._version

    def mark_updated(self):
        self._version += 1
