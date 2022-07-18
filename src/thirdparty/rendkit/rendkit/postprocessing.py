from thirdparty.rendkit.rendkit.glsl import GLSLTemplate, GLSLProgram
from thirdparty.rendkit.rendkit.util import create_rend_target
from thirdparty.vispy.vispy import gloo
from thirdparty.vispy.vispy.gloo import gl


class RendtexInputMixin:
    def upload_input(self, program, input_tex):
        program['u_rendtex'] = input_tex
        return program


class IdentityProgram(GLSLProgram, RendtexInputMixin):
    def __init__(self):
        super().__init__(
            GLSLTemplate.fromfile('postprocessing/quad.vert.glsl'),
            GLSLTemplate.fromfile('postprocessing/identity.frag.glsl'))

    def upload_uniforms(self, program):
        program['a_uv'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        program['a_position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        return program


class ClearProgram(GLSLProgram, RendtexInputMixin):
    def __init__(self, clear_color=(0, 0, 0, 0)):
        super().__init__(
            GLSLTemplate.fromfile('postprocessing/quad.vert.glsl'),
            GLSLTemplate.fromfile('postprocessing/clear.frag.glsl'))
        self.clear_color = clear_color

    def upload_uniforms(self, program):
        program['a_uv'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        program['a_position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        program['u_clear_color'] = self.clear_color
        return program


class DownsampleProgram(GLSLProgram, RendtexInputMixin):
    MAX_SCALE = 3
    LANCZOS_KERNELS = [
        [0.44031130485056913, 0.29880437751590694,
         0.04535643028360444, -0.06431646022479595],
        [0.2797564513818748, 0.2310717037833796,
         0.11797652759318597, 0.01107354293249700],
    ]

    def __init__(self, scale: int):
        super().__init__(
            GLSLTemplate.fromfile('postprocessing/quad.vert.glsl'),
            GLSLTemplate.fromfile('postprocessing/ssaa.frag.glsl'))
        assert scale == 2 or scale == 3
        self.scale = scale

    def upload_uniforms(self, program):
        program['u_aa_kernel'] = self.LANCZOS_KERNELS[self.scale - 2]
        program['a_uv'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        program['a_position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        return program

    def upload_input(self, program, input_tex):
        program['u_rendtex'] = input_tex
        program['u_texture_shape'] = input_tex.shape[:2]
        return program


class GammaCorrectionProgram(GLSLProgram, RendtexInputMixin):
    def __init__(self, gamma=2.2):
        super().__init__(
            GLSLTemplate.fromfile('postprocessing/quad.vert.glsl'),
            GLSLTemplate.fromfile('postprocessing/gamma_correction.frag.glsl'))
        self.gamma = gamma

    def upload_uniforms(self, program):
        program['a_uv'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        program['a_position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        program['u_gamma'] = self.gamma
        return program


class ReinhardProgram(GLSLProgram, RendtexInputMixin):
    def __init__(self, thres):
        super().__init__(
            GLSLTemplate.fromfile('postprocessing/quad.vert.glsl'),
            GLSLTemplate.fromfile('postprocessing/reinhard_tonemap.frag.glsl'))
        self.thres = thres

    def upload_uniforms(self, program):
        program['a_uv'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        program['a_position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        program['u_thres'] = self.thres
        return program


class ExposureProgram(GLSLProgram, RendtexInputMixin):
    def __init__(self, exposure):
        super().__init__(
            GLSLTemplate.fromfile('postprocessing/quad.vert.glsl'),
            GLSLTemplate.fromfile('postprocessing/exposure_tonemap.frag.glsl'))
        self.exposure = exposure

    def upload_uniforms(self, program):
        program['a_uv'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        program['a_position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        program['u_exposure'] = self.exposure
        return program


class PostprocessPipeline:
    def __init__(self, size):
        self.programs = []
        self.compiled = []
        self.rend_colortex_list = []
        self.rend_fb_list = []
        self.size = size

    def add_program(self, program):
        self.programs.append(program)
        self.compiled.append(program.compile())
        fb, colortex, _ = create_rend_target(self.size)
        self.rend_fb_list.append(fb)
        self.rend_colortex_list.append(colortex)

    def resize(self, size):
        print('pp resize')
        self.size = size
        for fb in self.rend_fb_list:
            fb.resize(self.size)

    def draw(self, input_colortex, input_depthtex, output_size,
             clear_color=(0, 0, 0, 0)):
        # Make sure clear color has alpha channel.
        if len(clear_color) == 3:
            clear_color = (*clear_color, 1)

        current_tex = input_colortex
        for i, program in enumerate(self.programs):
            is_last = i == (len(self.programs) - 1)
            compiled = self.compiled[i]
            program.upload_input(compiled, current_tex)
            gloo.clear(color=True)
            gloo.set_state(depth_test=False)
            if isinstance(program, ClearProgram):
                compiled['u_clear_color'] = clear_color
            if is_last:
                gloo.set_viewport(0, 0, *output_size)
                compiled.draw(gl.GL_TRIANGLE_STRIP)
            else:
                gloo.set_viewport(0, 0, *self.size)
                with self.rend_fb_list[i]:
                    compiled.draw(gl.GL_TRIANGLE_STRIP)
            current_tex = self.rend_colortex_list[i]
