import logging
from typing import Tuple

import numpy as np
from thirdparty.rendkit import rendkit
from thirdparty.rendkit.rendkit import util
from thirdparty.rendkit.rendkit import postprocessing as pp
from thirdparty.rendkit.rendkit.camera import BaseCamera
from thirdparty.rendkit.rendkit.scene import Scene
# from thirdparty.vispy.vispy import app, gloo
from vispy import app, gloo
# from thirdparty.vispy.vispy.gloo import gl
from vispy.gloo import gl

logger = logging.getLogger(__name__)


class _nop():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class BaseRenderer(app.Canvas):
    def __init__(self, size: Tuple[int, int],
                 camera: BaseCamera,
                 render_scale=1.0,
                 *args, **kwargs):
        if size is None:
            size = camera.size
        logger.debug("Render scale is {:.02f}".format(render_scale))
        self.render_scale = render_scale
        size = tuple(int(s * render_scale) for s in size)
        self.camera = camera
        super().__init__(size=size, *args, **kwargs)
        gloo.set_state(depth_test=True)
        gloo.set_viewport(0, 0, *self.size)

        self.size = size

        # Buffer shapes are HxW, not WxH...
        self._rendertex = gloo.Texture2D(
            shape=(size[1], size[0]) + (4,),
            internalformat='rgba32f')
        self._fbo = gloo.FrameBuffer(self._rendertex, gloo.RenderBuffer(
            shape=(size[1], size[0])))

    def set_program(self, vertex_shader, fragment_shader):
        self.active_program = gloo.Program(vertex_shader, fragment_shader)

    def draw(self, camera=None, out_size=None):
        """
        Override and implement drawing logic here. e.g. gloo.clear_color
        """
        raise NotImplementedError

    def render_to_image(self, camera=None, out_size=None, format='rgb') -> np.ndarray:
        """
        Renders to an image.
        :return: image of rendered scene.
        """
        with self._fbo:
            self.draw(camera, out_size)
            pixels: np.ndarray = np.copy(gloo.util.read_pixels(
                out_type=np.float32, format=format))
        return pixels

    def on_resize(self, event):
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        self.camera.size = self.size
        self._fbo.resize(self.size)
        self.update()

    def on_draw(self, event):
        self.draw()

    def on_mouse_move(self, event):
        if event.is_dragging:
            self.camera.handle_mouse(event.last_event.pos, event.pos)
            self.update()

    def on_mouse_wheel(self, event):
        v = self.camera.position - self.camera.lookat
        zoom_size = 1.05
        factor = zoom_size if event.delta[1] < 0 else 1/zoom_size
        self.camera.position = self.camera.lookat + v * factor
        self.update()

    def __enter__(self):
        self._backend._vispy_warmup()
        return self


class SceneRenderer(BaseRenderer):
    def __init__(self, scene: Scene, camera=None, size=None,
                 gamma=None,
                 ssaa=0,
                 tonemap=None,
                 exposure=1.0,
                 reinhard_thres=3.0,
                 conservative_raster=False,
                 resizable=True,
                 *args, **kwargs):
        self.scene = scene
        self.resizable = resizable 
        if camera is None and size is None:
            size = (1024, 1024)
            logger.warning("Neither camera nor size is set. Creating default "
                           "camera with size {}".format(size))
        if camera is None:
            camera = BaseCamera(size, 0.1, 100)
        if size is None:
            size = camera.size
        super().__init__(size, camera, *args, **kwargs)
        gloo.set_state(depth_test=True)
        if conservative_raster:
            from . import nvidia
            self.conservative_raster = nvidia.conservative_raster(True)
        else:
            self.conservative_raster = _nop()
        self.gamma = gamma
        self.ssaa_scale = min(max(1, ssaa),pp.DownsampleProgram.MAX_SCALE)

        # Initialize main camera.
        self.render_size = self.get_render_size(self.size)
        self.rend_target_by_cam = {
            self.camera: rendkit.util.create_rend_target(self.render_size)
        }
        logger.debug("Render size: {} --SSAAx{}--> {}".format(
            self.size, self.ssaa_scale, self.render_size))

        self.pp_pipeline = pp.PostprocessPipeline(self.size)

        if self.ssaa_scale >= 2:
            self.pp_pipeline.add_program(pp.DownsampleProgram(ssaa))

        if tonemap == 'reinhard':
            logger.info("Tonemapping mode REINHARD with threshold={}"
                        .format(reinhard_thres))
            self.pp_pipeline.add_program(pp.ReinhardProgram(reinhard_thres))
        elif tonemap == 'exposure':
            logger.info("Tonemapping mode EXPOSURE with exposure={}"
                        .format(exposure))
            self.pp_pipeline.add_program(pp.ExposureProgram(exposure))

        if gamma is not None:
            logger.info("Gamma correction with gamma={}"
                        .format(gamma))
            self.pp_pipeline.add_program(pp.GammaCorrectionProgram(gamma))
        else:
            self.pp_pipeline.add_program(pp.IdentityProgram())

        self.pp_pipeline.add_program(pp.ClearProgram())

    def get_render_size(self, size):
        return size[0] * self.ssaa_scale, size[1] * self.ssaa_scale

    def draw_scene(self, camera, rend_target):
        rend_fb, rend_colortex, _ = rend_target

        with rend_fb:
            gloo.clear(color=(0, 0, 0, 0))
            gloo.set_state(depth_test=True)
            gloo.set_viewport(0, 0, rend_colortex.shape[1], rend_colortex.shape[0])
            for renderable in self.scene.renderables:
                program = renderable.activate(self.scene, camera)
                with self.conservative_raster:
                    program.draw(gl.GL_TRIANGLES)

    def on_resize(self, event):
        if self.resizable:
            super().on_resize(event)
            for camera, (fb, ct, dt) in self.rend_target_by_cam.items():
                fb.resize(self.get_render_size(camera.size))
            # TODO: Fix resizing pipeline framebuffers.
            # self.pp_pipeline.resize(self.get_render_size(self.size))

    def draw(self, camera=None, out_size=None):
        if camera is None:
            camera = self.camera

        if out_size is None:
            # out_size = (self.physical_size
            #             if camera == self.camera else camera.size)
            out_size = camera.size

        if camera not in self.rend_target_by_cam:
            rend_fb, rend_colortex, rend_depthtex = rendkit.util.create_rend_target(
                tuple(s * self.ssaa_scale for s in out_size))
        else:
            rend_fb, rend_colortex, rend_depthtex = self.rend_target_by_cam[camera]
            # width, height = camera.size
            # rend_colortex.resize((height, width))
            # rend_fb.resize((height, width))
            # rend_depthtex.resize((height, width))

        with rend_fb:
            self.draw_scene(camera, (rend_fb, rend_colortex, rend_depthtex))

        self.pp_pipeline.draw(rend_colortex, rend_depthtex, out_size,
                              clear_color=camera.clear_color)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scene.reset()
        super().__exit__(exc_type, exc_val, exc_tb)
        return self
