import numpy as np

from thirdparty.rendkit.rendkit.core import ContextProvider
from thirdparty.rendkit.rendkit.glsl import GLSLProgram, GLSLTemplate
from thirdparty.vispy.vispy import gloo
from thirdparty.vispy.vispy.gloo import gl


class CubemapToDualParaboloidProgram(GLSLProgram):
    def __init__(self):
        super().__init__(
            GLSLTemplate.fromfile('postprocessing/quad.vert.glsl'),
            GLSLTemplate.fromfile('envmap/cubemap_to_dual_paraboloid.frag.glsl'))

    def upload_uniforms(self, program):
        program['a_position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        program['a_uv'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        return program


def cubemap_to_dual_paraboloid(cube_faces):
    _, height, width, n_channels = cube_faces.shape
    height, width = height * 2, width * 2
    internal_format = 'rgba32f' if n_channels == 4 else 'rgb32f'

    with ContextProvider((height, width)):
        rendtex = gloo.Texture2D(
            (height, width, n_channels), interpolation='linear',
            wrapping='repeat', internalformat=internal_format)

        framebuffer = gloo.FrameBuffer(
            rendtex, gloo.RenderBuffer((width, height, n_channels)))

        program = CubemapToDualParaboloidProgram().compile()
        program['u_cubemap'] = gloo.TextureCubeMap(
            cube_faces, internalformat=internal_format, mipmap_levels=8)

        results = []
        for i in [0, 1]:
            with framebuffer:
                gloo.set_viewport(0, 0, width, height)
                program['u_hemisphere'] = i
                program.draw(gl.GL_TRIANGLE_STRIP)
                results.append(np.flipud(gloo.read_pixels(out_type=np.float32,
                                                          format='rgb')))

    return results[0], results[1]


class PanoramaToCubemapProgram(GLSLProgram):
    def __init__(self):
        super().__init__(
            GLSLTemplate.fromfile('postprocessing/quad.vert.glsl'),
            GLSLTemplate.fromfile('envmap/panorama_to_cubemap.frag.glsl'))

    def upload_uniforms(self, program):
        program['a_position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        program['a_uv'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        return program


def panorama_to_cubemap(panorama, cube_size=256):
    program = PanoramaToCubemapProgram().compile()
    height, width = cube_size, cube_size
    n_channels = 3
    internal_format = 'rgb32f'

    with ContextProvider((height, width)):
        rendtex = gloo.Texture2D(
            (height, width, n_channels), interpolation='linear',
            wrapping='repeat', internalformat=internal_format)
        framebuffer = gloo.FrameBuffer(
            rendtex, gloo.RenderBuffer((width, height, n_channels)))
        gloo.set_viewport(0, 0, width, height)
        program['u_panorama'] = gloo.Texture2D(
            panorama.transpose((1, 0, 2)),
            internalformat=internal_format)
        results = np.zeros((6, height, width, n_channels), dtype=np.float32)
        for i in range(6):
            program['u_cube_face'] = i
            with framebuffer:
                program.draw(gl.GL_TRIANGLE_STRIP)
                results[i] = gloo.read_pixels(out_type=np.float32, format='rgb')
    return results


class CubemapToPanoramaProgram(GLSLProgram):
    def __init__(self):
        super().__init__(
            GLSLTemplate.fromfile('postprocessing/quad.vert.glsl'),
            GLSLTemplate.fromfile('envmap/cubemap_to_panorama.frag.glsl'))

    def upload_uniforms(self, program):
        program['a_position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        program['a_uv'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        return program


def cubemap_to_panorama(cube_faces):
    _, height, width, n_channels = cube_faces.shape
    height, width = height, width * 2
    internal_format = 'rgba32f' if n_channels == 4 else 'rgb32f'

    with ContextProvider((height, width)):
        rendtex = gloo.Texture2D(
            (height, width, n_channels), interpolation='linear',
            wrapping='repeat', internalformat=internal_format)

        framebuffer = gloo.FrameBuffer(
            rendtex, gloo.RenderBuffer((height, width, n_channels)))

        program = CubemapToPanoramaProgram().compile()
        program['u_cubemap'] = gloo.TextureCubeMap(
            cube_faces, internalformat=internal_format, mipmap_levels=8)

        with framebuffer:
            gloo.set_viewport(0, 0, width, height)
            program.draw(gl.GL_TRIANGLE_STRIP)
            result = gloo.read_pixels(out_type=np.float32, format='rgb')

    return result
