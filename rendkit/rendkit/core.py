import logging
from typing import Dict

import numpy as np

from thirdparty.rendkit.meshkit import Mesh
from thirdparty.rendkit.rendkit.materials import DepthMaterial
from vispy import gloo,app
# from thirdparty.vispy.vispy import gloo, app
# from thirdparty.vispy.vispy.gloo import gl
from vispy.gloo import gl

logger = logging.getLogger(__name__)


class Renderable:
    def __init__(self,
                 material_name: str,
                 attributes: Dict[str, np.ndarray],
                 model_mat=np.eye(4),
                 uv_scale=1.0):
        self.model_mat = model_mat
        self.material_name = material_name
        self._attributes = attributes
        self._uv_scale = uv_scale

        self._current_scene = None
        self._program = None
        self._scene_version = -1

    def set_uv_scale(self, scale):
        self._uv_scale = scale
        if 'a_uv' in self._attributes:
            if self._program is not None:
                self._program['u_uv_scale'] = self._uv_scale

    def scale_uv_scale(self, v):
        self._uv_scale *= v
        if 'a_uv' in self._attributes:
            if self._program is not None:
                self._program['u_uv_scale'] = self._uv_scale

    def activate(self, scene, camera):
        material = scene.get_material(self.material_name)
        if self._program is None or scene != self._current_scene:
            self._current_scene = scene
            self._scene_version = -1
        if self._scene_version != scene.version:
            self._current_scene = scene
            self._scene_version = scene.version
            self._program = material.compile(
                num_lights=len(scene.lights),
                num_shadow_sources=len(scene.shadow_sources),
                use_radiance_map=scene.radiance_map is not None)
            material.upload_attributes(self._program, self._attributes, self._uv_scale)
            material.upload_radmap(self._program, scene.radiance_map)
            material.upload_shadow_sources(self._program, scene.shadow_sources)
            material.upload_lights(self._program, scene.lights)

        material.upload_camera(self._program, camera)
        self._program['u_model'] = self.model_mat.T

        return self._program


def mesh_to_renderables(mesh: Mesh, model_mat):
    renderables = []
    # For now each renderable represents a submesh with the same materials.
    for material_id, material_name in enumerate(mesh.materials):
        filter = {'material': material_id}
        vertex_positions = mesh.expand_face_vertices(filter)
        vertex_normals = mesh.expand_face_normals(filter)
        vertex_tangents, vertex_bitangents = mesh.expand_tangents(
            filter)
        vertex_uvs = mesh.expand_face_uvs(filter)
        if len(vertex_positions) < 3:
            logger.warning('Material {} not visible.'.format(material_name))
            continue
        attributes = dict(
            a_position=vertex_positions,
            a_normal=vertex_normals,
            a_tangent=vertex_tangents,
            a_bitangent=vertex_bitangents,
            a_uv=vertex_uvs
        )
        renderables.append(Renderable(material_name, attributes, model_mat,
                                      uv_scale=mesh.uv_scale))
    return renderables


class DummyRenderer(app.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        gloo.set_viewport(0, 0, *self.size)

    def __enter__(self):
        self._backend._vispy_warmup()
        return self


class ContextProvider:
    def __init__(self, size):
        self.size = size
        canvas = gloo.get_current_canvas()
        self.context_exists = canvas is not None and not canvas._closed
        if self.context_exists:
            logger.debug("Using existing OpenGL context.")
            self.provider = gloo.get_current_canvas()
            self.previous_size = self.provider.size
        else:
            logger.debug("Providing temporary context with DummyRenderer.")
            self.provider = DummyRenderer(size=size)

    def __enter__(self):
        gloo.set_viewport(0, 0, *self.size)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.context_exists:
            self.provider.__exit__(exc_type, exc_val, exc_tb)
        else:
            gloo.set_viewport(0, 0, *self.previous_size)


def draw_depth(camera, renderables, rend_target):
    rendfb, rendtex, _ = rend_target

    material = DepthMaterial()
    program = DepthMaterial().compile()

    with rendfb:
        gloo.clear(color=camera.clear_color)
        gloo.set_state(depth_test=True)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_FRONT)
        for renderable in renderables:
            material.upload_camera(program, camera)
            material.upload_attributes(program, renderable._attributes)
            program['u_model'] = renderable.model_mat.T
            program.draw(gl.GL_TRIANGLES)
        gl.glCullFace(gl.GL_BACK)
        gl.glDisable(gl.GL_CULL_FACE)
