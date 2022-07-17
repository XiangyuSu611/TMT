from OpenGL.GLES2.NV.conservative_raster import \
    GL_CONSERVATIVE_RASTERIZATION_NV

from vispy.gloo import gl


class conservative_raster():
    def __init__(self, enabled):
        self.enabled = enabled

    def __enter__(self):
        if self.enabled:
            gl.glEnable(GL_CONSERVATIVE_RASTERIZATION_NV)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            gl.glDisable(GL_CONSERVATIVE_RASTERIZATION_NV)
