import uuid
import bpy
import json

from pathlib import Path

import math

import warnings

import thirdparty.brender.mdl as mdl
from thirdparty.brender.brender.mesh import PASS_INDEX_UV_DENSITY_MULT
from thirdparty.toolbox.toolbox.io.images import load_image, load_hdr


class BlenderMaterial(object):
    _next_id = 0

    def __init__(self, bobj=None, name=None, **kwargs):
        if name and not bobj:
            self.name = name
        elif bobj:
            self.name = bobj.name
        else:
            self.name = f'brender_material_{BlenderMaterial._next_id}_{uuid.uuid4()}'

        if bobj is None:
            self._bobj_was_given = False
            if self.name in bpy.data.materials:
                raise ValueError(f'Blender material {self.name!r} already '
                                 f'exists')
            BlenderMaterial._next_id += 1
            bpy.data.materials.new(name=self.name)
            self._bobj = bpy.data.materials[self.name]
            print(f'Created new material {self.name}')
        else:
            self._bobj_was_given = True
            self._bobj = bobj

    @property
    def bobj(self):
        return self._bobj

    def mean_roughness(self) -> float:
        raise NotImplementedError

    def __del__(self):
        pass

class NodesMaterial(BlenderMaterial):
    def __init__(self,
                 uv_ref_scale=1.0,
                 uv_translation=(0, 0),
                 uv_rotation=0,
                 **kwargs):
        super().__init__(**kwargs)
        self.bobj.use_nodes = True
        self.uv_ref_scale = uv_ref_scale
        self.uv_translation = uv_translation
        self.uv_rotation = uv_rotation

        # self.clear_nodes()
        self.uv_ref_scale_node = None

        nodes = self.bobj.node_tree.nodes
        if 'Material Output' not in nodes:
            output_node = nodes.new(type="ShaderNodeOutput")
            output_node.name = "Material Output"

        self.init_nodes()

    def clear_nodes(self):
        self.bobj.node_tree.nodes.clear()
        self.bobj.node_tree.links.clear()
        nodes = self.bobj.node_tree.nodes
        output_node = nodes.new(type="ShaderNodeOutput")
        output_node.name = "Material Output"

    @property
    def has_uvs(self):
        return self.uv_ref_scale_node is not None

    def connect_scale_node(self, map_nodes):
        nodes = self.bobj.node_tree.nodes
        links = self.bobj.node_tree.links

        uv_node = self.init_uv_node(nodes, links)

        for map_node in map_nodes:
            links.new(map_node.inputs[0], uv_node.outputs[0])

    def set_uv_ref_scale(self, uv_ref_scale):
        self.uv_ref_scale = uv_ref_scale
        self.update_uv_node()

    def update_uv_node(self):
        if self.uv_ref_scale_node is None:
            raise ValueError('UV scale node has not been initialized yet.')

        self.uv_ref_scale_node.outputs[0].default_value = self.uv_ref_scale

    def init_uv_node(self, nodes, links):
        self.uv_ref_scale_node = nodes.new(type='ShaderNodeValue')
        self.update_uv_node()

        obj_info_node = nodes.new(type='ShaderNodeObjectInfo')
        obj_info_to_uv_density_node = nodes.new(type='ShaderNodeMath')
        obj_info_to_uv_density_node.operation = 'DIVIDE'
        obj_info_to_uv_density_node.inputs[1].default_value = \
            PASS_INDEX_UV_DENSITY_MULT
        links.new(obj_info_to_uv_density_node.inputs[0], obj_info_node.outputs[1])

        # Divide our desired scale by the object UV density:
        #   uv_scale = uv_ref_scale / uv_density
        uv_scale_node = nodes.new(type='ShaderNodeMath')
        uv_scale_node.operation = 'DIVIDE'
        links.new(uv_scale_node.inputs[0], self.uv_ref_scale_node.outputs[0])
        links.new(uv_scale_node.inputs[1],
                  obj_info_to_uv_density_node.outputs[0])

        uv_map_node = nodes.new(type="ShaderNodeUVMap")
        separate_uv_node = nodes.new(type="ShaderNodeSeparateXYZ")
        combine_uv_node = nodes.new(type="ShaderNodeCombineXYZ")

        links.new(separate_uv_node.inputs[0], uv_map_node.outputs[0])

        for i in [0, 1, 2]:
            mult_node = nodes.new(type="ShaderNodeMath")
            mult_node.operation = 'MULTIPLY'
            links.new(mult_node.inputs[0], separate_uv_node.outputs[i])
            links.new(mult_node.inputs[1], uv_scale_node.outputs[0])
            links.new(combine_uv_node.inputs[i], mult_node.outputs[0])

        # Translate (or possible rotate) UVs.
        mapping_node = nodes.new(type="ShaderNodeMapping")
        mapping_node.vector_type = 'TEXTURE'
        mapping_node.translation[0] = self.uv_translation[0]
        mapping_node.translation[1] = self.uv_translation[1]
        mapping_node.rotation[2] = self.uv_rotation
        links.new(mapping_node.inputs[0], combine_uv_node.outputs[0])

        return mapping_node
        # return combine_uv_node

    def init_nodes(self):
        raise NotImplementedError

    @classmethod
    def load_annotations(cls, path):
        annot_path = path / 'annotations.json'
        if annot_path.exists():
            with open(annot_path, 'r') as f:
                scale = json.load(f).get('scale', 1.0)
                print(f'Loaded scale={scale} from {annot_path}')
        else:
            scale = 1.0
        return scale

class InvisibleMaterial(NodesMaterial):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_nodes(self):
        nodes = self.bobj.node_tree.nodes
        links = self.bobj.node_tree.links

        diffuse_node = nodes.new(type="ShaderNodeBsdfTransparent")
        diffuse_node.inputs[0].default_value = (0, 0, 0, 0)
        output_node = nodes["Material Output"]
        links.new(output_node.inputs[0], diffuse_node.outputs[0])

class DiffuseMaterial(NodesMaterial):
    def __init__(self,
                 diffuse_color=(0.3, 0.3, 0.3),
                 roughness=0.0,
                 **kwargs):
        self.diffuse_color = diffuse_color
        self.roughness = roughness
        super().__init__(**kwargs)

    def init_nodes(self):
        nodes = self.bobj.node_tree.nodes
        links = self.bobj.node_tree.links

        diffuse_node = nodes.new(type="ShaderNodeBsdfDiffuse")
        diffuse_node.inputs[0].default_value = (*self.diffuse_color, 1.0)
        diffuse_node.inputs[1].default_value = self.roughness

        output_node = nodes["Material Output"]
        links.new(output_node.inputs[0], diffuse_node.outputs[0])

    def mean_roughness(self):
        return 1.0

class RawTextureMaterial(NodesMaterial):
    def __init__(self, path, **kwargs):
        self.path = path
        super().__init__(**kwargs)

    def init_nodes(self):
        nodes = self.bobj.node_tree.nodes
        links = self.bobj.node_tree.links

        tex_coord_node = nodes.new(type="ShaderNodeTexCoord")
        mapping_node = nodes.new(type="ShaderNodeMapping")
        mapping_node.vector_type = 'TEXTURE'
        links.new(mapping_node.inputs[0], tex_coord_node.outputs[2])

        tex_node = nodes.new(type="ShaderNodeTexImage")
        tex_node.color_space = 'COLOR'
        tex_node.image = bpy.data.images.load(filepath=str(self.path))

        self.connect_scale_node([tex_node])

        emit_node = nodes.new(type="ShaderNodeEmission")

        links.new(tex_node.inputs[0], mapping_node.outputs[0])
        links.new(emit_node.inputs[0], tex_node.outputs[0])

        lightpath_node = nodes.new(type="ShaderNodeLightPath")
        emitmix_node = nodes.new(type="ShaderNodeMixShader")
        links.new(emitmix_node.inputs[0], lightpath_node.outputs[0])
        links.new(emitmix_node.inputs[2], emit_node.outputs[0])

        output_node = nodes["Material Output"]
        links.new(output_node.inputs[0], emitmix_node.outputs[0])

class SVBRDFMaterial(NodesMaterial):
    def __init__(self, diffuse_map_path, gloss_map_path, normal_map_path,
                 specular_map_path, **kwargs):
        self.diffuse_map_path = diffuse_map_path
        self.specular_map_path = specular_map_path
        self.gloss_map_path = gloss_map_path
        self.normal_map_path = normal_map_path
        super().__init__(**kwargs)

    @property
    def substance(self):
        return self.diffuse_map_path.parent.parent.name

    @property
    def source_name(self):
        return self.diffuse_map_path.parent.name

    def init_nodes(self):
        nodes = self.bobj.node_tree.nodes
        links = self.bobj.node_tree.links

        normal_tex_node = nodes.new(type="ShaderNodeTexImage")
        normal_tex_node.color_space = 'NONE'
        normal_tex_node.image = bpy.data.images.load(
            filepath=str(self.normal_map_path))
        normal_map_node = nodes.new(type="ShaderNodeNormalMap")
        normal_map_node.space = 'TANGENT'
        links.new(normal_map_node.inputs[1], normal_tex_node.outputs[0])

        diff_tex_node = nodes.new(type="ShaderNodeTexImage")
        diff_tex_node.image = bpy.data.images.load(
            filepath=str(self.diffuse_map_path))

        spec_tex_node = nodes.new(type="ShaderNodeTexImage")
        spec_tex_node.image = bpy.data.images.load(
            filepath=str(self.specular_map_path))

        gloss_tex_node = nodes.new(type="ShaderNodeTexImage")
        gloss_tex_node.color_space = 'NONE'
        gloss_tex_node.image = bpy.data.images.load(
            filepath=str(self.gloss_map_path))

        self.connect_scale_node([normal_tex_node,
                                 diff_tex_node,
                                 spec_tex_node,
                                 gloss_tex_node])

        gloss_invert_node = nodes.new(type='ShaderNodeInvert')
        links.new(gloss_invert_node.inputs[0], gloss_tex_node.outputs[0])

        diff_bsdf_node = nodes.new(type="ShaderNodeBsdfDiffuse")
        links.new(diff_bsdf_node.inputs[0], diff_tex_node.outputs[0])
        links.new(diff_bsdf_node.inputs[2], normal_map_node.outputs[0])

        spec_bsdf_node = nodes.new(type="ShaderNodeBsdfAnisotropic")
        spec_bsdf_node.distribution = 'GGX'
        # links.new(spec_bsdf_node.inputs[0], spec_output.outputs[0])
        links.new(spec_bsdf_node.inputs[0], spec_tex_node.outputs[0])
        links.new(spec_bsdf_node.inputs[1], gloss_invert_node.outputs[0])
        links.new(spec_bsdf_node.inputs[4], normal_map_node.outputs[0])

        mix_node = nodes.new(type="ShaderNodeMixShader")
        links.new(mix_node.inputs[1], spec_bsdf_node.outputs[0])
        links.new(mix_node.inputs[2], diff_bsdf_node.outputs[0])

        output_node = nodes["Material Output"]
        links.new(output_node.inputs[0], mix_node.outputs[0])

    def mean_roughness(self):
        if (self.gloss_map_path.name.endswith('.exr')
            or self.gloss_map_path.name.endswith('.hdr')):
            gloss_map = load_hdr(self.gloss_map_path)
        else:
            gloss_map = load_image(self.gloss_map_path)

        if len(gloss_map.shape) == 3:
            gloss_map = gloss_map[:, :, 0]

        return 1.0 - gloss_map.mean()

class MDLMaterial(NodesMaterial):
    BASE_COLOR_INP_ID = 0
    SUBSURF_COLOR_INP_ID = 3
    METALLIC_INP_ID = 4
    SPECULAR_INP_ID = 5
    ROUGHNESS_INP_ID = 7
    IOR_INP_ID = 14
    TRANSMISSION_INP_ID = 15
    NORMAL_INP_ID = 16

    @classmethod
    def from_path(cls, mat_path, **kwargs):
        if not mat_path.exists():
            raise FileNotFoundError(mat_path)

        parsed_dict = mdl.parse_mdl(mat_path)
        # print(parsed_dict)

        for k, v in parsed_dict.items():
            if isinstance(v, str):
                kwargs[k] = Path(mat_path.parent, v)
            elif isinstance(v, float):
                kwargs[k] = v
            elif isinstance(v, tuple):
                kwargs[k] = v
            elif v is None:
                warnings.warn(f'Value for {k} is None')
            else:
                raise ValueError(f'Unknown value type: {v}')

        return cls(mdl_path=mat_path, **kwargs)

    def __init__(self,
                 *,
                 mdl_path,
                 base_color,
                 roughness,
                 opacity=1.0,
                 normal=None,
                 metallic=0.0,
                 glow=0.0,
                 translucence=0.0,
                 subsurface_color=None,
                 ior=1.450,
                 height=None,
                 height_scale=1.0,
                 **kwargs):
        self.mdl_path = mdl_path
        self.base_color = base_color
        self.opacity = opacity
        self.roughness = roughness
        self.normal = normal
        self.metallic = metallic
        self.translucence = translucence
        self.glow = glow
        self.subsurface_color = subsurface_color
        self.ior = ior
        self.height = height
        self.height_scale = height_scale

        self.tex_nodes = []
        super().__init__(**kwargs)

    @property
    def substance(self):
        return self.mdl_path.parent.parent.name

    @property
    def source_name(self):
        return self.mdl_path.stem

    def _add_input(self, value, input_id, color_space, bsdf_node,
                   invert=False):
        if not value:
            raise ValueError('Value is None')

        nodes = self.bobj.node_tree.nodes
        links = self.bobj.node_tree.links

        if isinstance(value, Path):
            tex_node = nodes.new(type="ShaderNodeTexImage")
            tex_node.color_space = color_space
            tex_node.image = bpy.data.images.load(
                filepath=str(value))
            self.tex_nodes.append(tex_node)
            if invert:
                opacity_invert_node = nodes.new(type='ShaderNodeInvert')
                links.new(opacity_invert_node.inputs[0], tex_node.outputs[0])
                links.new(bsdf_node.inputs[input_id],
                          opacity_invert_node.outputs[0])
            else:
                links.new(bsdf_node.inputs[input_id], tex_node.outputs[0])
        elif isinstance(value, tuple):
            if invert:
                bsdf_node.inputs[input_id].default_value = \
                    tuple(1 - v for v in value)
            else:
                bsdf_node.inputs[input_id].default_value = value
        elif isinstance(value, float):
            bsdf_node.inputs[input_id].default_value = \
                (1 - value) if invert else value
        else:
            RuntimeError(f'Invalid type {value!r}')

    def init_nodes(self):
        self.tex_nodes.clear()
        nodes = self.bobj.node_tree.nodes
        links = self.bobj.node_tree.links

        bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf_node.distribution = 'MULTI_GGX'

        if self.normal:
            normal_map_node = nodes.new(type="ShaderNodeNormalMap")
            normal_map_node.space = 'TANGENT'

            if isinstance(self.normal, Path):
                normal_tex_node = nodes.new(type="ShaderNodeTexImage")
                normal_tex_node.color_space = 'NONE'
                normal_tex_node.image = bpy.data.images.load(
                    filepath=str(self.normal))
                self.tex_nodes.append(normal_tex_node)
                links.new(normal_map_node.inputs[1], normal_tex_node.outputs[0])
            elif isinstance(self.normal, tuple):
                normal_map_node.inputs[1].default_value = self.normal

        if not self.base_color:
            raise RuntimeError('Base color must be defined.')
        else:
            self._add_input(self.base_color, self.BASE_COLOR_INP_ID, 'COLOR',
                            bsdf_node)

        if self.subsurface_color:
            self._add_input(self.subsurface_color, self.SUBSURF_COLOR_INP_ID,
                            'COLOR', bsdf_node)

        if self.roughness:
            self._add_input(self.roughness, self.ROUGHNESS_INP_ID, 'NONE',
                            bsdf_node)

        if self.metallic:
            self._add_input(self.metallic, self.METALLIC_INP_ID, 'NONE',
                            bsdf_node)

        # if self.opacity:
        #     self._add_input(self.opacity, self.SPECULAR_INP_ID, 'NONE',
        #                     bsdf_node, invert=False)

        if self.ior:
            self._add_input(self.ior, self.IOR_INP_ID, 'NONE', bsdf_node)

        output_node = nodes["Material Output"]

        if self.height:
            bump_node = nodes.new(type="ShaderNodeBump")
            bump_node.inputs[0].default_value = self.height_scale
            if self.normal:
                links.new(bump_node.inputs[3], normal_map_node.outputs[0])

            if isinstance(self.height, Path):
                height_tex_node = nodes.new(type="ShaderNodeTexImage")
                height_tex_node.color_space = 'NONE'
                height_tex_node.image = bpy.data.images.load(
                    filepath=str(self.height))
                self.tex_nodes.append(height_tex_node)
                links.new(bump_node.inputs[0],
                          height_tex_node.outputs[0])
            elif isinstance(self.height, float):
                bump_node.inputs[0].default_value = self.height

            links.new(bsdf_node.inputs[self.NORMAL_INP_ID],
                      bump_node.outputs[0])
        elif self.normal:
            links.new(bsdf_node.inputs[self.NORMAL_INP_ID],
                      normal_map_node.outputs[0])

        self.connect_scale_node(self.tex_nodes)

        links.new(output_node.inputs[0], bsdf_node.outputs[0])

    def mean_roughness(self):
        if isinstance(self.roughness, Path):
            if (self.roughness.name.endswith('.exr')
                    or self.roughness.name.endswith('.hdr')):
                roughness_map = load_hdr(self.roughness)
            else:
                roughness_map = load_image(self.roughness)
            if len(roughness_map.shape) == 3:
                roughness_map = roughness_map[:, :, 0]

            return roughness_map.mean()
        elif isinstance(self.roughness, float):
            return self.roughness

        raise RuntimeError('Invalid roughness value')

class PoliigonMaterial(SVBRDFMaterial):
    @classmethod
    def from_path(cls, mat_path, **kwargs):
        if not mat_path.exists():
            raise FileNotFoundError(mat_path)

        try:
            if len(list(mat_path.glob('*COL_VAR2*.jpg'))) == 0:
                diffuse_map_path = list(mat_path.glob('*COL_VAR2*.png'))[0]
            else:
                diffuse_map_path = list(mat_path.glob('*COL_VAR2*.jpg'))[0]
        except IndexError:
            if len(list(mat_path.glob('*COL*.jpg'))) == 0:
                diffuse_map_path = list(mat_path.glob('*COL*.png'))[0]
            else:
                diffuse_map_path = list(mat_path.glob('*COL*.jpg'))[0]
        if len(list(mat_path.glob('*_REFL*.jpg'))) == 0:
            specular_map_path = list(mat_path.glob('*_REFL*.png'))[0]
        else:
            specular_map_path = list(mat_path.glob('*_REFL*.jpg'))[0]
        if len(list(mat_path.glob('*_NRM*.jpg'))) == 0:
            normal_map_path = list(mat_path.glob('*_NRM*.png'))[0]
        else:
            normal_map_path = list(mat_path.glob('*_NRM*.jpg'))[0]
        if len(list(mat_path.glob('*_GLOSS*.jpg'))) == 0:
            gloss_map_path = list(mat_path.glob('*_GLOSS*.png'))[0]
        else:
            gloss_map_path = list(mat_path.glob('*_GLOSS*.jpg'))[0]
        # only need diffuse(COLOR), specular(REFL), normal(NRM), gloss(GLOSS)
        return cls(diffuse_map_path=diffuse_map_path,
                   specular_map_path=specular_map_path,
                   gloss_map_path=gloss_map_path,
                   normal_map_path=normal_map_path,
                   **kwargs)

class TextureMaterial(NodesMaterial):
    # aledo, height(displacement, we don't use at this time), normal, roughness, metallic(only for matel)
    @classmethod
    def from_path(cls, mat_path, **kwargs):
        if not mat_path.exists():
            raise FileNotFoundError(mat_path)
        
        albedo_map_path = list(mat_path.glob('*albedo*.tif'))[0]
        normal_map_path = list(mat_path.glob('*normal*.tif'))[0]
        rough_map_path = list(mat_path.glob('*roughness*.tif'))[0]
        try:
            metallic_map_path = list(mat_path.glob('*metallic*.tif'))[0]
        except:
            metallic_map_path = []
        
        
        return cls(albedo_map_path=albedo_map_path,
                    normal_map_path=normal_map_path,
                    rough_map_path=rough_map_path,
                    metallic_map_path=metallic_map_path,
                    **kwargs)
    
    def __init__(self,
                 albedo_map_path,
                 normal_map_path,
                 rough_map_path,
                 metallic_map_path,
                 **kwargs):
        self.albedo_map_path = albedo_map_path
        self.normal_map_path = normal_map_path
        self.rough_map_path = rough_map_path
        self.metallic_map_path = metallic_map_path

        self.tex_nodes = []
        super().__init__(**kwargs)

    @property
    def substance(self):
        return self.base_map_path.parent.parent.name

    @property
    def source_name(self):
        return self.base_map_path.parent.name

    def init_nodes(self):
        self.tex_nodes.clear()
        nodes = self.bobj.node_tree.nodes
        links = self.bobj.node_tree.links

        bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf_node.distribution = 'MULTI_GGX'

        if self.albedo_map_path:
            base_node = nodes.new(type='ShaderNodeTexImage')
            base_node.color_space = 'COLOR'
            base_node.image = bpy.data.images.load(filepath=str(self.albedo_map_path))
            self.tex_nodes.append(base_node)
            links.new(bsdf_node.inputs[0], base_node.outputs[0])
        
        if self.normal_map_path:
            normal_tex_node = nodes.new(type='ShaderNodeTexImage')
            normal_tex_node.color_space = 'NONE'
            normal_tex_node.image = bpy.data.images.load(filepath=str(self.normal_map_path))
            self.tex_nodes.append(normal_tex_node)
            normal_map_node = nodes.new(type='ShaderNodeNormalMap')
            links.new(normal_map_node.inputs[1], normal_tex_node.outputs[0])
            links.new(bsdf_node.inputs['Normal'], normal_map_node.outputs[0])
             
        if self.rough_map_path:
            rough_node = nodes.new(type='ShaderNodeTexImage')
            rough_node.color_space = 'NONE'
            rough_node.image = bpy.data.images.load(filepath=str(self.rough_map_path))
            self.tex_nodes.append(rough_node)
            links.new(bsdf_node.inputs[7], rough_node.outputs[0])
        
        if self.metallic_map_path:
            metal_node = nodes.new(type='ShaderNodeTexImage')
            metal_node.color_space = 'NONE'
            metal_node.image = bpy.data.images.load(filepath=str(self.metallic_map_path))
            self.tex_nodes.append(metal_node)
            links.new(bsdf_node.inputs[4], metal_node.outputs[0])


        output_node = nodes['Material Output']
        self.connect_scale_node(self.tex_nodes)
        links.new(output_node.inputs[0], bsdf_node.outputs[0])
        links.new(bsdf_node.inputs['Normal'], normal_map_node.outputs[0])

    def mean_roughness(self):
        if isinstance(self.roughness, Path):
            if (self.roughness.name.endswith('.exr')
                    or self.roughness.name.endswith('.hdr')):
                roughness_map = load_hdr(self.roughness)
            else:
                roughness_map = load_image(self.roughness)
            if len(roughness_map.shape) == 3:
                roughness_map = roughness_map[:, :, 0]

            return roughness_map.mean()
        elif isinstance(self.roughness, float):
            return self.roughness

        raise RuntimeError('Invalid roughness value')
    
class TextureHarvenMaterial(NodesMaterial):    
    # basecolor(diff/col) nor(nor/bump) rough spec
    @classmethod
    def from_path(cls, mat_path, **kwargs):
        if not mat_path.exists():
            raise FileNotFoundError(mat_path)
        try:
            col_map_path = list(mat_path.glob('*_col*.jpg'))[0]
        except:
            col_map_path = []
        try:
            dif_map_path = list(mat_path.glob('*_diff*.jpg'))[0]
        except:
            dif_map_path = []
        try:
            nor_map_path = list(mat_path.glob('*_nor*.jpg'))[0]
        except:
            nor_map_path = []
        try:
            bum_map_path = list(mat_path.glob('*_bump*.jpg'))[0]
        except:
            bum_map_path = []
        try:
            rough_map_path = list(mat_path.glob('*_rough*.jpg'))[0]
        except:
            rough_map_path = []
        try:
            spec_map_path = list(mar_path.glob('*_spec*.jpg'))[0]
        except:
            spec_map_path = []    
        
        return cls(col_map_path=col_map_path,
                    dif_map_path=dif_map_path,
                    nor_map_path=nor_map_path,
                    bum_map_path=bum_map_path,
                    rough_map_path=rough_map_path,
                    spec_map_path=spec_map_path,
                    **kwargs)
    def __init__(self,
                 col_map_path,
                 dif_map_path,
                 nor_map_path,
                 bum_map_path,
                 rough_map_path,
                 spec_map_path,
                 **kwargs):
        self.col_map_path = col_map_path
        self.dif_map_path = dif_map_path
        self.nor_map_path = nor_map_path
        self.bum_map_path = bum_map_path
        self.rough_map_path = rough_map_path
        self.spec_map_path = spec_map_path

        self.tex_nodes = []
        super().__init__(**kwargs)

    @property
    def substance(self):
        return self.base_map_path.parent.parent.name

    @property
    def source_name(self):
        return self.base_map_path.parent.name

    def init_nodes(self):
        self.tex_nodes.clear()
        nodes = self.bobj.node_tree.nodes
        links =  self.bobj.node_tree.links

        bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf_node.distribution = 'MULTI_GGX'

        if self.col_map_path:
            base_node = nodes.new(type='ShaderNodeTexImage')
            base_node.color_space = 'COLOR'
            base_node.image = bpy.data.images.load(filepath=str(self.col_map_path))
            self.tex_nodes.append(base_node)
            links.new(bsdf_node.inputs[0], base_node.outputs[0])
        elif self.dif_map_path:
            base_node = nodes.new(type='ShaderNodeTexImage')
            base_node.color_space = 'COLOR'
            base_node.image = bpy.data.images.load(filepath=str(self.dif_map_path))
            self.tex_nodes.append(base_node)
            links.new(bsdf_node.inputs[0], base_node.outputs[0])
        
        if self.bum_map_path:
            normal_tex_node = nodes.new(type='ShaderNodeTexImage')
            normal_tex_node.color_space = 'NONE'
            normal_tex_node.image = bpy.data.images.load(filepath=str(self.bum_map_path))
            self.tex_nodes.append(normal_tex_node)
            bump_node = nodes.new(type='ShaderNodeTexImage')
            bump_node.color_space = 'NONE'
            bump_node.image = bpy.data.images.load(filepath=str(self.bum_map_path))
            links.new(bump_node.inputs[3], normal_tex_node.outputs[0])
            links.new(bsdf_node.inputs['Normal'], bump_node.outputs[0])
        elif self.nor_map_path:
            normal_tex_node = nodes.new(type='ShaderNodeTexImage')
            normal_tex_node.color_space = 'NONE'
            normal_tex_node.image = bpy.data.images.load(filepath=str(self.nor_map_path))
            self.tex_nodes.append(normal_tex_node)
            normal_map_node = nodes.new(type='ShaderNodeNormalMap')
            links.new(normal_map_node.inputs[1], normal_tex_node.outputs[0])
            links.new(bsdf_node.inputs['Normal'], normal_map_node.outputs[0])
             
        if self.rough_map_path:
            rough_node = nodes.new(type='ShaderNodeTexImage')
            rough_node.color_space = 'NONE'
            rough_node.image = bpy.data.images.load(filepath=str(self.rough_map_path))
            self.tex_nodes.append(rough_node)
            links.new(bsdf_node.inputs[7], rough_node.outputs[0])
        
        if self.spec_map_path:
            spec_node = nodes.new(type='ShaderNodeTexImage')
            spec_node.color_space = 'NONE'
            spec_node.image = bpy.data.image.load(filepath=str(str.spec_map_path))
            self.tex_nodes.apppend(spec_node)
            links.new(bsdf_node.inputs[5], spec_node.outputs[0])

        output_node = nodes['Material Output']
        self.connect_scale_node(self.tex_nodes)
        links.new(output_node.inputs[0], bsdf_node.outputs[0])
        if self.bum_map_path:
            links.new(bsdf_node.inputs['Normal'], bump_node.outputs[0])
        elif self.nor_map_path:
            links.new(bsdf_node.inputs['Normal'], normal_map_node.outputs[0])

    def mean_roughness(self):
        if isinstance(self.roughness, Path):
            if (self.roughness.name.endswith('.exr')
                    or self.roughness.name.endswith('.hdr')):
                roughness_map = load_hdr(self.roughness)
            else:
                roughness_map = load_image(self.roughness)
            if len(roughness_map.shape) == 3:
                roughness_map = roughness_map[:, :, 0]

            return roughness_map.mean()
        elif isinstance(self.roughness, float):
            return self.roughness

        raise RuntimeError('Invalid roughness value')
    
class Texture3DMaterial(NodesMaterial):
    
    # diff, displacement (we don't use at this time), normal, roughness, metallic(only for matel)
    @classmethod
    def from_path(cls, mat_path, **kwargs):
        if not mat_path.exists():
            raise FileNotFoundError(mat_path)
        
        diff_map_path = list(mat_path.glob('*COLOR*'))[0]
        normal_map_path = list(mat_path.glob('*NORM*'))[0]
        try:
            rough_map_path = list(mat_path.glob('*ROUGH*'))[0]
        except:
            rough_map_path = []
        try:
            metallic_map_path = list(mat_path.glob('*METAL*'))[0]
        except:
            metallic_map_path = []
        try:
            spec_map_path = list(mat_path.glob('*SPEC*'))[0]
        except:
            spec_map_path = []
        
        return cls(diff_map_path=diff_map_path,
                    normal_map_path=normal_map_path,
                    rough_map_path=rough_map_path,
                    metallic_map_path=metallic_map_path,
                    spec_map_path = spec_map_path,
                    **kwargs)
    
    def __init__(self,
                 diff_map_path,
                 normal_map_path,
                 rough_map_path,
                 metallic_map_path,
                 spec_map_path,
                 **kwargs):
        self.diff_map_path = diff_map_path
        self.normal_map_path = normal_map_path
        self.rough_map_path = rough_map_path
        self.metallic_map_path = metallic_map_path
        self.spec_map_path = spec_map_path
        
        self.tex_nodes = []
        super().__init__(**kwargs)

    @property
    def substance(self):
        return self.base_map_path.parent.parent.name

    @property
    def source_name(self):
        return self.base_map_path.parent.name

    def init_nodes(self):
        self.tex_nodes.clear()
        nodes = self.bobj.node_tree.nodes
        links = self.bobj.node_tree.links

        bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf_node.distribution = 'MULTI_GGX'

        if self.diff_map_path:
            base_node = nodes.new(type='ShaderNodeTexImage')
            base_node.color_space = 'COLOR'
            base_node.image = bpy.data.images.load(filepath=str(self.diff_map_path))
            self.tex_nodes.append(base_node)
            links.new(bsdf_node.inputs[0], base_node.outputs[0])
        
        if self.normal_map_path:
            normal_tex_node = nodes.new(type='ShaderNodeTexImage')
            normal_tex_node.color_space = 'NONE'
            normal_tex_node.image = bpy.data.images.load(filepath=str(self.normal_map_path))
            self.tex_nodes.append(normal_tex_node)
            normal_map_node = nodes.new(type='ShaderNodeNormalMap')
            links.new(normal_map_node.inputs[1], normal_tex_node.outputs[0])
            links.new(bsdf_node.inputs['Normal'], normal_map_node.outputs[0])
             
        if self.rough_map_path:
            rough_node = nodes.new(type='ShaderNodeTexImage')
            rough_node.color_space = 'NONE'
            rough_node.image = bpy.data.images.load(filepath=str(self.rough_map_path))
            self.tex_nodes.append(rough_node)
            links.new(bsdf_node.inputs[7], rough_node.outputs[0])
        
        if self.metallic_map_path:
            metal_node = nodes.new(type='ShaderNodeTexImage')
            metal_node.color_space = 'NONE'
            metal_node.image = bpy.data.images.load(filepath=str(self.metallic_map_path))
            self.tex_nodes.append(metal_node)
            links.new(bsdf_node.inputs[4], metal_node.outputs[0])

        if self.spec_map_path:
            spec_node = nodes.new(type='ShaderNodeTexImage')
            spec_node.color_space = 'NONE'
            spec_node.image = bpy.data.images.load(filepath=str(self.spec_map_path))
            self.tex_nodes.append(spec_node)
            links.new(bsdf_node.inputs[5], spec_node.outputs[0])
        
        output_node = nodes['Material Output']
        self.connect_scale_node(self.tex_nodes)
        links.new(output_node.inputs[0], bsdf_node.outputs[0])
        links.new(bsdf_node.inputs['Normal'], normal_map_node.outputs[0])

    def mean_roughness(self):
        if isinstance(self.roughness, Path):
            if (self.roughness.name.endswith('.exr')
                    or self.roughness.name.endswith('.hdr')):
                roughness_map = load_hdr(self.roughness)
            else:
                roughness_map = load_image(self.roughness)
            if len(roughness_map.shape) == 3:
                roughness_map = roughness_map[:, :, 0]

            return roughness_map.mean()
        elif isinstance(self.roughness, float):
            return self.roughness

        raise RuntimeError('Invalid roughness value')

class ShareTextureMaterial(NodesMaterial):
    # diff, displacement (we don't use at this time), normal, roughness
    @classmethod
    def from_path(cls, mat_path, **kwargs):
        if not mat_path.exists():
            raise FileNotFoundError(mat_path)
        
        try:
            diff_map_path = list(mat_path.glob('*diffuse*'))[0]
        except:
            diff_map_path = []
        try:
            base_map_path = list(mat_path.glob('*olor*'))[0]
        except:
            base_map_path = []
        try:
            normal_map_path = list(mat_path.glob('*ormal*'))[0]
        except:
            normal_map_path = []
        try:
            rough_map_path = list(mat_path.glob('*oughness*'))[0]
        except:
            rough_map_path = []
        try:
            spec_map_path = list(mat_path.glob('*pecular*'))[0]
        except:
            spec_map_path = []
        try:
            metal_map_path = list(mat_path.glob('*etallic*'))[0]
        except:
            metal_map_path = []
        
        return cls(diff_map_path=diff_map_path,
                    base_map_path=base_map_path,
                    normal_map_path=normal_map_path,
                    rough_map_path=rough_map_path,
                    spec_map_path=spec_map_path,
                    metal_map_path=metal_map_path,
                    **kwargs)
    
    def __init__(self,
                 diff_map_path,
                 base_map_path,
                 normal_map_path,
                 rough_map_path,
                 spec_map_path,
                 metal_map_path,
                 **kwargs):
        self.diff_map_path = diff_map_path
        self.base_map_path = base_map_path
        self.normal_map_path = normal_map_path
        self.rough_map_path = rough_map_path
        self.spec_map_path = spec_map_path
        self.metal_map_path = metal_map_path

        self.tex_nodes = []
        super().__init__(**kwargs)

    @property
    def substance(self):
        return self.base_map_path.parent.parent.name

    @property
    def source_name(self):
        return self.base_map_path.parent.name

    def init_nodes(self):
        self.tex_nodes.clear()
        nodes = self.bobj.node_tree.nodes
        links = self.bobj.node_tree.links

        bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf_node.distribution = 'MULTI_GGX'

        if self.base_map_path:
            base_node = nodes.new(type='ShaderNodeTexImage')
            base_node.color_space = 'COLOR'
            base_node.image = bpy.data.images.load(filepath=str(self.base_map_path))
            self.tex_nodes.append(base_node)
            links.new(bsdf_node.inputs[0], base_node.outputs[0])
        elif self.diff_map_path:
            base_node = nodes.new(type='ShaderNodeTexImage')
            base_node.color_space = 'COLOR'
            base_node.image = bpy.data.images.load(filepath=str(self.diff_map_path))
            self.tex_nodes.append(base_node)
            links.new(bsdf_node.inputs[0], base_node.outputs[0])
        
        if self.normal_map_path:
            normal_tex_node = nodes.new(type='ShaderNodeTexImage')
            normal_tex_node.color_space = 'NONE'
            normal_tex_node.image = bpy.data.images.load(filepath=str(self.normal_map_path))
            self.tex_nodes.append(normal_tex_node)
            normal_map_node = nodes.new(type='ShaderNodeNormalMap')
            links.new(normal_map_node.inputs[1], normal_tex_node.outputs[0])
            links.new(bsdf_node.inputs['Normal'], normal_map_node.outputs[0])
             
        if self.rough_map_path:
            rough_node = nodes.new(type='ShaderNodeTexImage')
            rough_node.color_space = 'NONE'
            rough_node.image = bpy.data.images.load(filepath=str(self.rough_map_path))
            self.tex_nodes.append(rough_node)
            links.new(bsdf_node.inputs[7], rough_node.outputs[0])

        if self.spec_map_path:
            spec_node = nodes.new(type='ShaderNodeTexImage')
            spec_node.color_space = 'NONE'
            spec_node.image = bpy.data.images.load(filepath=str(self.spec_map_path))
            self.tex_nodes.append(spec_node)
            links.new(bsdf_node.inputs[5], spec_node.outputs[0])
        
        if self.metal_map_path:
            metal_node = nodes.new(type='ShaderNodeTexImage')
            metal_node.color_space = 'NONE'
            metal_node.image = bpy.data.images.load(filepath=str(self.metal_map_path))
            self.tex_nodes.append(metal_node)
            links.new(bsdf_node.inputs[4], metal_node.outputs[0])

        output_node = nodes['Material Output']
        self.connect_scale_node(self.tex_nodes)
        links.new(output_node.inputs[0], bsdf_node.outputs[0])
        links.new(bsdf_node.inputs['Normal'], normal_map_node.outputs[0])

    def mean_roughness(self):
        if isinstance(self.roughness, Path):
            if (self.roughness.name.endswith('.exr')
                    or self.roughness.name.endswith('.hdr')):
                roughness_map = load_hdr(self.roughness)
            else:
                roughness_map = load_image(self.roughness)
            if len(roughness_map.shape) == 3:
                roughness_map = roughness_map[:, :, 0]

            return roughness_map.mean()
        elif isinstance(self.roughness, float):
            return self.roughness

        raise RuntimeError('Invalid roughness value')

class CC0textureMaterial(NodesMaterial):
    # include Normal, Displacement, Color, Roughness, Metalness(only for mateiral textures)
    @classmethod
    def from_path(cls, mat_path, **kwargs):
        if not mat_path.exists():
            raise FileNotFoundError(mat_path)
        color_map_path = list(mat_path.glob('*_Color*.jpg'))[0]
        dis_map_path = list(mat_path.glob('*_Displacement*.jpg'))[0]
        normal_map_path = list(mat_path.glob('*_Normal*.jpg'))[0]
        roughness_map_path  = list(mat_path.glob('*_Roughness*.jpg'))[0]
        try:
            metallic_map_path = list(mat_path.glob('*_Metalness*,jpg'))[0]
        except:
            metallic_map_path = []
        
        return cls(color_map_path=color_map_path,
                    dis_map_path=dis_map_path,
                    normal_map_path=normal_map_path,
                    roughness_map_path=roughness_map_path,
                    metallic_map_path=metallic_map_path,
                    **kwargs)
    
    def __init__(self,
                color_map_path,
                dis_map_path,
                normal_map_path,
                roughness_map_path,
                metallic_map_path,
                **kwargs):
        self.color_map_path = color_map_path
        self.dis_map_path = dis_map_path
        self.normal_map_path = normal_map_path
        self.roughness_map_path = roughness_map_path
        self.metallic_map_path = metallic_map_path
    
        self.tex_nodes = []
        super().__init__(**kwargs)

    @property
    def substance(self):
        return self.color_map_path.parent.parent.name

    @property
    def source_name(self):
        return self.color_map_path.parent.name

    def init_nodes(self):
        self.tex_nodes.clear()
        nodes = self.bobj.node_tree.nodes
        links =  self.bobj.node_tree.links

        bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf_node.distribution = 'MULTI_GGX'

        if self.color_map_path:
            base_node = nodes.new(type='ShaderNodeTexImage')
            base_node.color_space = 'COLOR'
            base_node.image = bpy.data.images.load(filepath=str(self.color_map_path))
            self.tex_nodes.append(base_node)
            links.new(bsdf_node.inputs[0], base_node.outputs[0])

        if self.normal_map_path:
            # nor_node = nodes.new(type='ShaderNodeNormalMap')
            # nor_node.space = 'TANGENT'
            normal_tex_node = nodes.new(type="ShaderNodeTexImage")
            normal_tex_node.color_space = 'NONE'
            normal_tex_node.image = bpy.data.images.load(filepath=str(self.normal_map_path))
            self.tex_nodes.append(normal_tex_node)
            sep_RGB_node = nodes.new(type='ShaderNodeSeparateRGB')
            links.new(sep_RGB_node.inputs[0], normal_tex_node.outputs[0])
            revert_green_node = nodes.new(type='ShaderNodeInvert')
            links.new(revert_green_node.inputs[1], sep_RGB_node.outputs[1])
            com_RGB_node = nodes.new(type='ShaderNodeCombineRGB')
            links.new(com_RGB_node.inputs[0], sep_RGB_node.outputs[0])
            links.new(com_RGB_node.inputs[1], revert_green_node.outputs[0])
            links.new(com_RGB_node.inputs[2], sep_RGB_node.outputs[2])
            nor_map_node = nodes.new(type='ShaderNodeNormalMap')
            links.new(bsdf_node.inputs[16], nor_map_node.outputs[0])
            links.new(nor_map_node.outputs[0], bsdf_node.inputs[16])
            links.new(nor_map_node.inputs[1], com_RGB_node.outputs[0])
            
            
        if self.roughness_map_path:
            rough_node = nodes.new(type='ShaderNodeTexImage')
            rough_node.color_space = 'NONE'
            rough_node.image = bpy.data.images.load(filepath=str(self.roughness_map_path))
            self.tex_nodes.append(rough_node)
            links.new(bsdf_node.inputs[7], rough_node.outputs[0])

        if self.metallic_map_path:
            metal_node = nodes.new(type='ShaderNodeTexImage')
            metal_node.color_space = 'NONE'
            metal_node.image = bpy.data.images.load(filepath=str(self.metallic_map_path))
            self.tex_nodes.append(metal_node)
            links.new(bsdf_node.inputs[4], metal_node.outputs[0])

        output_node = nodes['Material Output']
        self.connect_scale_node(self.tex_nodes)
        links.new(output_node.inputs[0], bsdf_node.outputs[0])
        links.new(bsdf_node.inputs['Normal'], nor_map_node.outputs[0])
    def mean_roughness(self):
        if isinstance(self.roughness, Path):
            if (self.roughness.name.endswith('.exr')
                    or self.roughness.name.endswith('.hdr')):
                roughness_map = load_hdr(self.roughness)
            else:
                roughness_map = load_image(self.roughness)
            if len(roughness_map.shape) == 3:
                roughness_map = roughness_map[:, :, 0]

            return roughness_map.mean()
        elif isinstance(self.roughness, float):
            return self.roughness

        raise RuntimeError('Invalid roughness value')

class VRayMaterial(SVBRDFMaterial):
    @classmethod
    def from_path(cls, mat_path, **kwargs):
        if not mat_path.exists():
            raise FileNotFoundError(mat_path)

        diffuse_map_path = mat_path / 'output.VRayDiffuseFilter.0000.exr'
        specular_map_path = mat_path / 'output.VRayReflectionFilter.0000.exr'
        normal_map_path = mat_path / 'output.VRaySamplerInfo.0000.exr'
        gloss_map_path = mat_path / 'output.VRayMtlReflectGlossiness.0000.exr'

        return cls(diffuse_map_path=diffuse_map_path,
                   specular_map_path=specular_map_path,
                   gloss_map_path=gloss_map_path,
                   normal_map_path=normal_map_path,
                   **kwargs)

class AittalaMaterial(NodesMaterial):

    @classmethod
    def from_path(cls, mat_path, **kwargs):
        if not mat_path.exists():
            raise FileNotFoundError(mat_path)

        diffuse_map_path = mat_path / 'map_diffuse.exr'
        specular_map_path = mat_path / 'map_specular.exr'
        normal_map_path = mat_path / 'map_normal_blender.png'
        roughness_map_path = mat_path / 'map_roughness.exr'
        anisotropy_map_path = mat_path / 'map_anisotropy.exr'

        return cls(diffuse_map_path=diffuse_map_path,
                   specular_map_path=specular_map_path,
                   roughness_map_path=roughness_map_path,
                   normal_map_path=normal_map_path,
                   anisotropy_map_path=anisotropy_map_path,
                   **kwargs)

    def __init__(self,
                 diffuse_map_path,
                 specular_map_path,
                 roughness_map_path,
                 normal_map_path,
                 anisotropy_map_path,
                 **kwargs):
        self.diffuse_map_path = diffuse_map_path
        self.specular_map_path = specular_map_path
        self.roughness_map_path = roughness_map_path
        self.normal_map_path = normal_map_path
        self.anisotropy_map_path = anisotropy_map_path
        super().__init__(**kwargs)

    @property
    def substance(self):
        return self.diffuse_map_path.parent.parent.name

    @property
    def source_name(self):
        return self.diffuse_map_path.parent.name

    def init_nodes(self):
        nodes = self.bobj.node_tree.nodes
        links = self.bobj.node_tree.links

        normal_tex_node = nodes.new(type="ShaderNodeTexImage")
        normal_tex_node.color_space = 'NONE'
        normal_tex_node.image = bpy.data.images.load(
            filepath=str(self.normal_map_path))
        normal_map_node = nodes.new(type="ShaderNodeNormalMap")
        normal_map_node.space = 'TANGENT'
        links.new(normal_map_node.inputs[1], normal_tex_node.outputs[0])

        diff_tex_node = nodes.new(type="ShaderNodeTexImage")
        diff_tex_node.image = bpy.data.images.load(
            filepath=str(self.diffuse_map_path))

        spec_tex_node = nodes.new(type="ShaderNodeTexImage")
        spec_tex_node.image = bpy.data.images.load(
            filepath=str(self.specular_map_path))

        aniso_tex_node = nodes.new(type="ShaderNodeTexImage")
        aniso_tex_node.color_space = 'NONE'
        aniso_tex_node.image = bpy.data.images.load(
            filepath=str(self.anisotropy_map_path))

        rough_tex_node = nodes.new(type="ShaderNodeTexImage")
        rough_tex_node.color_space = 'NONE'
        rough_tex_node.image = bpy.data.images.load(
            filepath=str(self.roughness_map_path))

        self.connect_scale_node([normal_tex_node,
                                 diff_tex_node,
                                 spec_tex_node,
                                 rough_tex_node,
                                 aniso_tex_node])

        diff_scale_node = nodes.new(type="ShaderNodeHueSaturation")
        diff_scale_node.inputs[2].default_value = 0.3
        links.new(diff_scale_node.inputs[4], diff_tex_node.outputs[0])
        diff_output = clamp_image_node(
            nodes, links, diff_scale_node, max_value=5.0)

        spec_scale_node = nodes.new(type="ShaderNodeHueSaturation")
        spec_scale_node.inputs[2].default_value = 0.3
        links.new(spec_scale_node.inputs[4], spec_tex_node.outputs[0])
        spec_output = clamp_image_node(
            nodes, links, spec_scale_node, max_value=2.0)

        diff_bsdf_node = nodes.new(type="ShaderNodeBsdfDiffuse")
        links.new(diff_bsdf_node.inputs[0], diff_output)
        links.new(diff_bsdf_node.inputs[2], normal_map_node.outputs[0])

        spec_bsdf_node = nodes.new(type="ShaderNodeBsdfAnisotropic")
        spec_bsdf_node.distribution = 'BECKMANN'
        links.new(spec_bsdf_node.inputs[0], spec_output)
        links.new(spec_bsdf_node.inputs[1], rough_tex_node.outputs[0])
        links.new(spec_bsdf_node.inputs[2], aniso_tex_node.outputs[0])
        links.new(spec_bsdf_node.inputs[4], normal_map_node.outputs[0])

        mix_node = nodes.new(type="ShaderNodeMixShader")
        links.new(mix_node.inputs[1], spec_bsdf_node.outputs[0])
        links.new(mix_node.inputs[2], diff_bsdf_node.outputs[0])

        output_node = nodes["Material Output"]
        links.new(output_node.inputs[0], mix_node.outputs[0])

    def mean_roughness(self):
        if (self.roughness_map_path.name.endswith('.exr')
                or self.roughness_map_path.name.endswith('.hdr')):
            roughness_map = load_hdr(self.roughness_map_path)
        else:
            roughness_map = load_image(self.roughness_map_path)

        if len(roughness_map.shape) == 3:
            roughness_map = roughness_map[:, :, 0]

        return roughness_map.mean()

class BlinnPhongMaterial(NodesMaterial):
    """
    Represents a Blinn-Phong parameterized material and renders it in Blender
    through the Beckmann BSDF.
    """

    def __init__(self, diffuse_albedo, specular_albedo,
                 shininess=None,
                 roughness=None,
                 **kwargs):
        if len(diffuse_albedo) == 3:
            diffuse_albedo = (*diffuse_albedo, 1.0)
        if len(specular_albedo) == 3:
            specular_albedo = (*specular_albedo, 1.0)

        if roughness is None and shininess is None:
            raise ValueError("Either shininess or roughness must be set.")

        self._roughness = roughness

        self.diffuse_albedo = diffuse_albedo
        self.specular_albedo = specular_albedo
        self.shininess = shininess
        super().__init__(**kwargs)

    @property
    def roughness(self):
        """
        Follows from power = roughness^2 - 2

        http://graphicrants.blogspot.de/2013/08/specular-brdf-reference.html
        :return:
        """
        if self._roughness is None:
            return math.sqrt(2 / (self.shininess + 2))
        else:
            return self._roughness

    def init_nodes(self):
        nodes = self.bobj.node_tree.nodes
        links = self.bobj.node_tree.links

        diff_bsdf_node = nodes.new(type="ShaderNodeBsdfDiffuse")
        diff_bsdf_node.inputs[0].default_value = self.diffuse_albedo

        spec_bsdf_node = nodes.new(type="ShaderNodeBsdfGlossy")
        spec_bsdf_node.distribution = 'BECKMANN'
        spec_bsdf_node.inputs[0].default_value = self.specular_albedo
        spec_bsdf_node.inputs[1].default_value = self.roughness

        mix_node = nodes.new(type="ShaderNodeMixShader")
        links.new(mix_node.inputs[1], spec_bsdf_node.outputs[0])
        links.new(mix_node.inputs[2], diff_bsdf_node.outputs[0])

        output_node = nodes["Material Output"]
        links.new(output_node.inputs[0], mix_node.outputs[0])

    def mean_roughness(self):
        return self.roughness

class PrincipledMaterial(NodesMaterial):
    """
    Represents a Blinn-Phong parameterized material and renders it in Blender
    through the Beckmann BSDF.
    """

    def __init__(self,
                 diffuse_color,
                 specular,
                 metallic,
                 roughness,
                 anisotropy=0.0,
                 anisotropic_rotation=0.0,
                 clearcoat=0.0,
                 clearcoat_roughness=0.03,
                 ior=1.45,
                 **kwargs):
        if len(diffuse_color) == 3:
            diffuse_color = (*diffuse_color, 1.0)

        self.distribution = 'MULTI_GGX'
        self.diffuse_color = diffuse_color
        self.specular = specular
        self.metallic = metallic
        self.roughness = roughness
        self.anisotropy = anisotropy
        self.anisotropic_rotation = anisotropic_rotation
        self.clearcoat = clearcoat
        self.clearcoat_roughness = clearcoat_roughness
        self.ior = ior

        super().__init__(**kwargs)

    def init_nodes(self):
        nodes = self.bobj.node_tree.nodes
        links = self.bobj.node_tree.links

        # diff_scale_node = nodes.new(type="ShaderNodeHueSaturation")
        # diff_scale_node.inputs[2].default_value = 1/0.3
        # diff_scale_node.inputs[4].default_value = self.diffuse_albedo
        # spec_scale_node = nodes.new(type="ShaderNodeHueSaturation")
        # spec_scale_node.inputs[2].default_value = 1/0.3
        # spec_scale_node.inputs[4].default_value = self.specular_albedo

        bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf_node.distribution = 'MULTI_GGX'
        bsdf_node.inputs[0].default_value = self.diffuse_color
        bsdf_node.inputs[4].default_value = self.metallic
        bsdf_node.inputs[5].default_value = self.specular
        bsdf_node.inputs[7].default_value = self.roughness
        bsdf_node.inputs[8].default_value = self.anisotropy
        bsdf_node.inputs[9].default_value = self.anisotropic_rotation
        bsdf_node.inputs[12].default_value = self.clearcoat
        bsdf_node.inputs[13].default_value = self.clearcoat_roughness
        bsdf_node.inputs[14].default_value = self.ior

        output_node = nodes["Material Output"]
        links.new(output_node.inputs[0], bsdf_node.outputs[0])

    def serialize(self):
        return {
            'name': self.name,
            'distribution': self.distribution,
            'diffuse_color': self.diffuse_color,
            'specular': self.specular,
            'metallic': self.metallic,
            'roughness': self.roughness,
            'anisotropy': self.anisotropy,
            'anisotropic_rotation': self.anisotropic_rotation,
            'clearcoat': self.clearcoat,
            'clearcoat_roughness': self.clearcoat_roughness,
            'ior': self.ior,
        }

    def mean_roughness(self):
        return self.roughness


def clamp_image_node(nodes, links, input, max_value):
    rgb_separate = nodes.new(type="ShaderNodeSeparateRGB")
    links.new(rgb_separate.inputs[0], input.outputs[0])

    r_math = nodes.new(type="ShaderNodeMath")
    r_math.operation = 'MINIMUM'
    r_math.inputs[1].default_value = max_value
    g_math = nodes.new(type="ShaderNodeMath")
    g_math.operation = 'MINIMUM'
    g_math.inputs[1].default_value = max_value
    b_math = nodes.new(type="ShaderNodeMath")
    b_math.operation = 'MINIMUM'
    b_math.inputs[1].default_value = max_value

    links.new(r_math.inputs[0], rgb_separate.outputs[0])
    links.new(g_math.inputs[0], rgb_separate.outputs[1])
    links.new(b_math.inputs[0], rgb_separate.outputs[2])

    rgb_combine = nodes.new(type="ShaderNodeCombineRGB")
    links.new(rgb_combine.inputs[0], r_math.outputs[0])
    links.new(rgb_combine.inputs[1], g_math.outputs[0])
    links.new(rgb_combine.inputs[2], b_math.outputs[0])

    return rgb_combine.outputs[0]

