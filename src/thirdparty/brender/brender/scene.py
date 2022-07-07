import bpy
import enum
import tempfile
from contextlib import contextmanager

import skimage
import skimage.io

from thirdparty.toolbox.toolbox.io.images import load_hdr


class BackgroundMode(enum.Enum):
    DISABLED = enum.auto()
    ENVMAP = enum.auto()
    COLOR = enum.auto()


class Engine(enum.Enum):
    CYCLES = 'CYCLES'
    BLENDER = 'BLENDER_RENDER'


class Scene:
    _current = None
    _context_stack = []

    @staticmethod
    def current():
        return Scene._current

    def __init__(self, app, shape, device='GPU',
                 engine=Engine.CYCLES,
                 tile_size=(40, 40),
                 aa_samples=1024,
                 diffuse_samples=2,
                 specular_samples=2,
                 num_samples=1024,
                 background_mode=BackgroundMode.ENVMAP,
                 background_color=(0, 0, 0, 1),
                 bscene=None):
        if not app.initialized:
            raise RuntimeError('App must be initialized.')

        self._app = app

        if bscene is None:
            self.bobj = bpy.context.scene
        else:
            self.bobj = bscene

        self.bobj.render.engine = engine.value
        self.bobj.cycles.device = device
        self.bobj.cycles.samples = num_samples
        self.bobj.render.tile_x = tile_size[0]
        self.bobj.render.tile_y = tile_size[1]
        self.bobj.render.resolution_percentage = 100

        # self.bobj.cycles.samples = samples
        # self.bobj.cycles.diffuse_samples = diffuse_samples
        # self.bobj.cycles.glossy_samples = specular_samples
        self.bobj.cycles.aa_samples = aa_samples
        # self.bobj.cycles.progressive = 'BRANCHED_PATH'
        # self.bobj.cycles.caustics_refractive = False
        # self.bobj.cycles.caustics_reflective = False
        # self.bobj.cycles.min_bounces = 2
        # self.bobj.cycles.max_bounces = 6

        self.background_mode = background_mode
        self.background_color = background_color

        self.bobj.use_nodes = True

        # Make envmap invisible to camera.
        if self.background_mode == BackgroundMode.DISABLED:
            self.bobj.world.cycles_visibility.camera = False
        else:
            self.bobj.world.cycles_visibility.camera = True

        if device == 'GPU':
            prefs = bpy.context.user_preferences.addons['cycles'].preferences
            prefs.compute_device_type = 'CUDA'
            prefs.devices[0].use = True


        self.camera = None
        self.shape = shape
        self.meshes = []
        self.bmats = []

        self._envmap_mapping_node = None


    def add_bmat(self, bmat):
        self.bmats.append(bmat)

    def clear_bmats(self):
        while len(self.bmats) > 0:
            bmat = self.bmats.pop()
            bmat.bobj.name = bmat.bobj.name
            del bmat

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape
        self.bobj.render.resolution_y = shape[0]
        self.bobj.render.resolution_x = shape[1]

    def clear(self):
        for obj in self.bobj.objects:
            self.bobj.objects.unlink(obj)

    def set_active_camera(self, camera):
        self.camera = camera
        self.bobj.camera = camera.bobj

    @property
    def active_camera(self):
        return self.camera

    def render(self, path):
        if self.active_camera is None:
            raise RuntimeError('No active camera.')

        if str(path).endswith('.hdr'):
            bpy.context.scene.render.image_settings.file_format = 'HDR'
        elif str(path).endswith('.exr'):
            bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
        else:
            bpy.context.scene.render.image_settings.file_format = 'PNG'

        self.bobj.render.filepath = str(path)
        bpy.ops.render.render(write_still=True, scene=self.bobj.name)

    def render_to_array(self, format='png'):
        """
        Renders the image to an array. If the format is 'hdr' or 'exr' the
        rendering is a linear HDR image. If format is 'png' then the image
        will be processed with Blender's default tonemapping and postprocessing
        effects.

        :param format: one of png, exr, or hdr.
        :return: rendered image
        """
        with tempfile.NamedTemporaryFile(suffix=f'.{format}') as f:
            self.render(f.name)
            if format in {'hdr', 'exr'}:
                return load_hdr(f.name)
            else:
                return skimage.img_as_float(skimage.io.imread(f.name))

    def clear_envmap(self):
        scene = self.bobj
        scene.world.use_nodes = True
        scene.world.node_tree.nodes.clear()
        scene.world.node_tree.links.clear()

    def set_envmap(self, path, scale=1.0, rotation=(0.0, 0.0, 0.0)):
        print(f"Setting envmap to {path}")

        self.clear_envmap()

        scene = self.bobj
        scene.world.use_nodes = True
        nodes = scene.world.node_tree.nodes
        links = scene.world.node_tree.links

        lightpath_node = nodes.new(type='ShaderNodeLightPath')

        texcoord_node = nodes.new(type='ShaderNodeTexCoord')
        mapping_node = nodes.new(type='ShaderNodeMapping')
        mapping_node.rotation = rotation
        self._envmap_mapping_node = mapping_node

        env_image = bpy.data.images.load(filepath=str(path))
        envtex_node = nodes.new(type="ShaderNodeTexEnvironment")
        envtex_node.image = env_image
        env_bg_node = nodes.new(type="ShaderNodeBackground")
        env_bg_node.inputs[1].default_value = scale

        if self.background_mode == BackgroundMode.COLOR:
            white_bg_node = nodes.new(type='ShaderNodeBackground')
            white_bg_node.inputs[0].default_value = self.background_color
            bg_select_node = nodes.new(type='ShaderNodeMixShader')
            links.new(bg_select_node.inputs[0], lightpath_node.outputs[0])
            links.new(bg_select_node.inputs[1], env_bg_node.outputs[0])
            links.new(bg_select_node.inputs[2], white_bg_node.outputs[0])
            bg_output = bg_select_node.outputs[0]
        else:
            bg_output = env_bg_node.outputs[0]

        output_node = nodes.new(type="ShaderNodeOutputWorld")

        links.new(mapping_node.inputs[0], texcoord_node.outputs[0])
        links.new(envtex_node.inputs[0], mapping_node.outputs[0])
        links.new(env_bg_node.inputs[0], envtex_node.outputs[0])
        links.new(output_node.inputs[0], bg_output)

    def set_envmap_rotation(self, rotation):
        if self._envmap_mapping_node is None:
            raise ValueError('Envmap has not been set yet.')
        self._envmap_mapping_node.rotation = rotation

    def clear_materials(self):
        for material in bpy.data.materials:
            if material.name == 'material_floor':
                continue
            print(f'Removing material {material.name!r}')
            # material.user_clear()
            bpy.data.materials.remove(material, do_unlink=True)

    def clear_meshes(self):
        for obj in self.bobj.objects:
            if obj.name == 'floor':
                continue
            if obj.type == 'MESH':
                print('Removing mesh {obj.name!r}')
                bpy.data.objects.remove(obj, do_unlink=True)

    @contextmanager
    def select(self):
        if Scene._current is not None:
            raise RuntimeError('Only one scene may be active.')

        Scene._current = self
        Scene._context_stack.append(bpy.context.screen.scene)
        bpy.context.screen.scene = self.bobj
        try:
            yield self
        finally:
            Scene._current = None
            bpy.context.screen.scene = Scene._context_stack.pop()
