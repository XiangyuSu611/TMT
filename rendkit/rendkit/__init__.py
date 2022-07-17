from vispy.gloo import gl
import vispy
# from thirdparty.vispy.vispy.gloo import gl
# from thirdparty.vispy.vispy import app
gl.use_gl('glplus')


def init_headless():
    # app.use_app('glfw')
    app.use_app('egl')
