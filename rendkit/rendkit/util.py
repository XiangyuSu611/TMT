# from thirdparty.vispy.vispy import gloo
from vispy import gloo


def create_rend_target(size):
    shape = (size[1], size[0])
    rendtex = gloo.Texture2D((*shape, 4),
                             interpolation='linear',
                             internalformat='rgba32f')
    depthtex = rendtex                
    # depthtex = gloo.Texture2D((*shape, 1), format='depth_component',
    #                           internalformat='depth_component32_oes')
    framebuffer = gloo.FrameBuffer(rendtex, depthtex)
    return framebuffer, rendtex, depthtex