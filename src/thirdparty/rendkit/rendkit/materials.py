import logging

import math
import numpy as np
from skimage.color import rgb2lab

from thirdparty.rendkit.rendkit.glsl import GLSLProgram, GLSLTemplate, glsl_bool
from thirdparty.rendkit.svbrdf.aittala import AittalaSVBRDF
from thirdparty.rendkit.svbrdf.beckmann import BeckmannSVBRDF
from thirdparty.vispy.vispy.gloo import Texture2D

logger = logging.getLogger(__name__)


class BasicMaterial(GLSLProgram):
    def __init__(self, color: np.ndarray):
        super().__init__(GLSLTemplate.fromfile('default.vert.glsl'),
                         GLSLTemplate.fromfile('basic.frag.glsl'),
                         use_uvs=False,
                         use_lights=False,
                         use_cam_pos=False,
                         use_normals=False,
                         use_tangents=False)
        self.color = color
        self.uniforms = {
            'u_color': self.color
        }


class BasicTextureMaterial(GLSLProgram):
    def __init__(self, texture: np.ndarray):
        super().__init__(GLSLTemplate.fromfile('default.vert.glsl'),
                         GLSLTemplate.fromfile('basic_texture.frag.glsl'),
                         use_uvs=True,
                         use_lights=False,
                         use_cam_pos=False,
                         use_normals=False,
                         use_tangents=False)
        self.texture = texture
        self.uniforms = {}
        self.init_uniforms()

    def init_uniforms(self):
        self.uniforms['u_texture'] = Texture2D(
            self.texture,
            interpolation=('linear_mipmap_linear', 'linear'),
            wrapping='repeat',
            mipmap_levels=10,
            internalformat='rgb32f')


class BlinnPhongMaterial(GLSLProgram):
    def __init__(self, diff_color, spec_color, roughness):
        super().__init__(GLSLTemplate.fromfile('default.vert.glsl'),
                         GLSLTemplate.fromfile('blinn_phong.frag.glsl'),
                         use_uvs=False,
                         use_lights=True,
                         use_cam_pos=True,
                         use_normals=True,
                         use_radiance_map=True)
        self.diff_color = diff_color
        self.spec_color = spec_color
        self.roughness = roughness
        self.uniforms = {
            'u_diff': self.diff_color,
            'u_spec': self.spec_color,
            'u_roughness': self.roughness,
        }


class UVMaterial(GLSLProgram):
    def __init__(self):
        super().__init__(GLSLTemplate.fromfile('default.vert.glsl'),
                         GLSLTemplate.fromfile('uv.frag.glsl'),
                         use_uvs=True,
                         use_cam_pos=False,
                         use_lights=False,
                         use_normals=False)


class DepthMaterial(GLSLProgram):
    def __init__(self):
        super().__init__(GLSLTemplate.fromfile('depth.vert.glsl'),
                         GLSLTemplate.fromfile('depth.frag.glsl'),
                         use_uvs=False,
                         use_cam_pos=False,
                         use_lights=False,
                         use_normals=False,
                         use_near_far=True)


class WorldCoordMaterial(GLSLProgram):
    def __init__(self):
        super().__init__(GLSLTemplate.fromfile('world_coord.vert.glsl'),
                         GLSLTemplate.fromfile('world_coord.frag.glsl'),
                         use_uvs=False,
                         use_cam_pos=False,
                         use_lights=False,
                         use_normals=False)


class NormalMaterial(GLSLProgram):
    def __init__(self):
        super().__init__(GLSLTemplate.fromfile('default.vert.glsl'),
                         GLSLTemplate.fromfile('normal.frag.glsl'),
                         use_uvs=False,
                         use_cam_pos=False,
                         use_lights=False,
                         use_normals=True)


class TangentMaterial(GLSLProgram):
    def __init__(self):
        super().__init__(GLSLTemplate.fromfile('default.vert.glsl'),
                         GLSLTemplate.fromfile('tangent.frag.glsl'),
                         use_uvs=False,
                         use_cam_pos=False,
                         use_lights=False,
                         use_tangents=True,
                         use_bitangents=False)


class BitangentMaterial(GLSLProgram):
    def __init__(self):
        super().__init__(GLSLTemplate.fromfile('default.vert.glsl'),
                         GLSLTemplate.fromfile('bitangent.frag.glsl'),
                         use_uvs=False,
                         use_cam_pos=False,
                         use_lights=False,
                         use_tangents=False,
                         use_bitangents=True)


class BeckmannMaterial(GLSLProgram):
    def __init__(self, svbrdf: BeckmannSVBRDF):
        super().__init__(GLSLTemplate.fromfile('default.vert.glsl'),
                         GLSLTemplate.fromfile('beckmann.frag.glsl'),
                         use_uvs=True,
                         use_cam_pos=True,
                         use_lights=True,
                         use_normals=True,
                         use_tangents=True,
                         use_bitangents=True,
                         use_radiance_map=True)
        self.diffuse_map = svbrdf.diffuse_map.astype(np.float32)
        self.specular_map = svbrdf.specular_map.astype(np.float32)
        self.normal_map = svbrdf.normal_map.astype(np.float32)
        self.roughness_map = svbrdf.roughness_map.astype(np.float32)
        self.anisotropy_map = svbrdf.anisotropy_map.astype(np.float32)

        self.frag_tpl_vars['change_color'] = glsl_bool(False)
        self.diff_map_lab = None
        self.diff_old_mean = None
        self.diff_old_std = None
        self.diff_new_mean = None
        self.diff_new_std = None

        self.uniforms = {}
        self.init_uniforms()

    def change_color(self, new_mean, new_std=None):
        self.frag_tpl_vars['change_color'] = glsl_bool(True)
        if self.diff_map_lab is None:
            self.diff_map_lab = rgb2lab(np.clip(self.diffuse_map / math.pi, 0, 1))
            self.diff_old_mean = self.diff_map_lab.mean(axis=(0, 1))
            self.diff_old_std = self.diff_map_lab.std(axis=(0, 1))
        self.diff_new_mean = new_mean
        self.diff_new_std = new_std
        self.uniforms['u_mean_old'] = self.diff_old_mean
        self.uniforms['u_std_old'] = self.diff_old_std
        self.uniforms['u_mean_new'] = self.diff_new_mean
        if new_std:
            self.uniforms['u_std_new'] = self.diff_new_std
        else:
            self.uniforms['u_std_new'] = self.diff_old_std
        self.update_instances()

    def init_uniforms(self):
        self.uniforms['u_diff_map'] = Texture2D(
            self.diffuse_map,
            interpolation=('linear_mipmap_linear', 'linear'),
            wrapping='repeat',
            mipmap_levels=10,
            internalformat='rgb32f')
        self.uniforms['u_spec_map'] = Texture2D(
            self.specular_map,
            interpolation=('linear_mipmap_linear', 'linear'),
            wrapping='repeat',
            mipmap_levels=10,
            internalformat='rgb32f')
        # TODO: Mipmapping here causes artifacts, but not doing it makes the
        # opject too specular. How can we fix this?
        self.uniforms['u_rough_map'] = Texture2D(
            self.roughness_map,
            interpolation=('linear_mipmap_linear', 'linear'),
            wrapping='repeat',
            mipmap_levels=10,
            internalformat='r32f')
        self.uniforms['u_aniso_map'] = Texture2D(
            self.anisotropy_map,
            interpolation=('linear_mipmap_linear', 'linear'),
            wrapping='repeat',
            mipmap_levels=10,
            internalformat='r32f')
        self.uniforms['u_normal_map'] = Texture2D(
            self.normal_map,
            interpolation=('linear_mipmap_linear', 'linear'),
            wrapping='repeat',
            mipmap_levels=10,
            internalformat='rgb32f')


class AittalaMaterial(GLSLProgram):
    def __init__(self, svbrdf: AittalaSVBRDF):
        super().__init__(GLSLTemplate.fromfile('default.vert.glsl'),
                         GLSLTemplate.fromfile('aittala.frag.glsl'),
                         use_uvs=True,
                         use_cam_pos=True,
                         use_lights=True,
                         use_normals=True,
                         use_tangents=True,
                         use_bitangents=True,
                         use_radiance_map=True)
        self.alpha = svbrdf.alpha
        self.diff_map = svbrdf.diffuse_map.astype(np.float32)
        self.spec_map = svbrdf.specular_map.astype(np.float32)
        self.spec_shape_map = svbrdf.spec_shape_map.astype(np.float32)
        self.normal_map = svbrdf.normal_map.astype(np.float32)

        self.sigma_min = svbrdf.sigma_min
        self.sigma_max = svbrdf.sigma_max
        self.cdf_sampler = svbrdf.cdf_sampler
        self.pdf_sampler = svbrdf.pdf_sampler

        self.frag_tpl_vars['change_color'] = glsl_bool(False)
        self.diff_map_lab = None
        self.diff_old_mean = None
        self.diff_old_std = None
        self.diff_new_mean = None
        self.diff_new_std = None

        self.uniforms = {}
        self.init_uniforms()

    def change_color(self, new_mean, new_std=None):
        self.frag_tpl_vars['change_color'] = glsl_bool(True)
        if self.diff_map_lab is None:
            self.diff_map_lab = rgb2lab(np.clip(self.diff_map, 0, 1))
            self.diff_old_mean = self.diff_map_lab.mean(axis=(0, 1))
            self.diff_old_std = self.diff_map_lab.std(axis=(0, 1))
        self.diff_new_mean = new_mean
        self.diff_new_std = new_std
        self.uniforms['u_mean_old'] = self.diff_old_mean
        self.uniforms['u_std_old'] = self.diff_old_std
        self.uniforms['u_mean_new'] = self.diff_new_mean
        if new_std:
            self.uniforms['u_std_new'] = self.diff_new_std
        else:
            self.uniforms['u_std_new'] = self.diff_old_std
        self.update_instances()

    def init_uniforms(self):
        self.uniforms['u_alpha'] = self.alpha
        self.uniforms['u_diff_map'] = Texture2D(
            self.diff_map,
            interpolation=('linear_mipmap_linear', 'linear'),
            wrapping='repeat',
            mipmap_levels=10,
            internalformat='rgb32f')
        self.uniforms['u_spec_map'] = Texture2D(
            self.spec_map,
            interpolation=('linear_mipmap_linear', 'linear'),
            wrapping='repeat',
            mipmap_levels=10,
            internalformat='rgb32f')
        # TODO: Mipmapping here causes artifacts, but not doing it makes the
        # opject too specular. How can we fix this?
        self.uniforms['u_spec_shape_map'] = Texture2D(
            self.spec_shape_map,
            interpolation=('linear_mipmap_linear', 'linear'),
            wrapping='repeat',
            mipmap_levels=10,
            internalformat='rgb32f')
        self.uniforms['u_normal_map'] = Texture2D(
            self.normal_map,
            interpolation=('linear_mipmap_linear', 'linear'),
            wrapping='repeat',
            mipmap_levels=10,
            internalformat='rgb32f')
        self.uniforms['u_cdf_sampler'] = Texture2D(
            self.cdf_sampler.astype(np.float32),
            interpolation='linear',
            wrapping='clamp_to_edge',
            internalformat='r32f')
        self.uniforms['u_pdf_sampler'] = Texture2D(
            self.pdf_sampler.astype(np.float32),
            interpolation='linear',
            wrapping='clamp_to_edge',
            internalformat='r32f')
        self.uniforms['u_sigma_range'] = (self.sigma_min, self.sigma_max)


class UnwrapToUVMaterial(GLSLProgram):
    def __init__(self, image, depth_im):
        super().__init__(GLSLTemplate.fromfile('unwrap_to_uv.vert.glsl'),
                         GLSLTemplate.fromfile('unwrap_to_uv.frag.glsl'),
                         use_uvs=True)
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if depth_im.dtype != np.float32:
            depth_im = depth_im.astype(np.float32)
        self.input_tex = Texture2D(image,
                                   interpolation='linear',
                                   wrapping='clamp_to_edge',
                                   internalformat='rgb32f')
        self.input_depth = Texture2D(depth_im,
                                     interpolation='linear',
                                     wrapping='clamp_to_edge',
                                     internalformat='r32f')
        self.uniforms = {
            'input_tex': self.input_tex,
            'input_depth': self.input_depth,
        }


class DummyMaterial(GLSLProgram):
    def __init__(self):
        super().__init__(GLSLTemplate.fromfile('default.vert.glsl'),
                         GLSLTemplate.fromfile('dummy.frag.glsl'),
                         use_lights=False,
                         use_radiance_map=False)


PLACEHOLDER_MATERIAL = BlinnPhongMaterial([1.0, 0.0, 1.0], [1, 0, 1], 0.01)
