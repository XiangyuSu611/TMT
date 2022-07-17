import argparse
import logging

import numpy as np
from vispy import app

from rendkit import jsd
from rendkit.shapes import make_plane
from rendkit.camera import ArcballCamera
from rendkit.scene import Scene
from rendkit.materials import AittalaMaterial
from rendkit.renderers import SceneRenderer
from svbrdf.aittala import AittalaSVBRDF

LOG_FORMAT= "[%(asctime)s] [%(levelname)8s] %(message)s (%(name)s:%(lineno)s)"
LOG_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = logging.INFO
LOG_FORMATTER = logging.Formatter(LOG_FORMAT, LOG_TIME_FORMAT)


app.use_app('pyglet')

parser = argparse.ArgumentParser()
parser.add_argument('--brdf', dest='brdf', type=str, required=True)
parser.add_argument('--radmap', dest='radmap', type=str, required=True)
parser.add_argument('--ssaa', dest='ssaa', type=int, default=2)
parser.add_argument('--gamma', dest='gamma', type=float, default=2.2)
parser.add_argument('--tonemap', dest='tonemap', type=str, default=None)
parser.add_argument(
    '--reinhard-thres', dest='reinhard_thres', type=float, default=3.0)
parser.add_argument('--exposure', dest='exposure', type=float, default=1.0)

args = parser.parse_args()

np.set_printoptions(suppress=True)


class MyRenderer(SceneRenderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, show=True)
        self.current_mat_idx = 0
        self.current_mat = None

    def on_key_press(self, event):
        if event.key == 'Escape':
            self.app.quit()
            return

        if event.key == '=':
            self.update_uv_scale(2)
            print('(+) UV scale')
        elif event.key == '-':
            self.update_uv_scale(0.5)
            print('(-) UV scale')
        self.update()
        self.draw()

    def recompile_renderables(self):
        for renderable in self.scene.renderables:
            renderable._program = renderable.compile(self.scene)

    def update_uv_scale(self, v):
        for renderable in self.scene.renderables:
            renderable.scale_uv_scale(v)
        self.update()


if __name__ == '__main__':
    console = logging.StreamHandler()
    console.setFormatter(LOG_FORMATTER)
    root_logger = logging.getLogger()
    root_logger.addHandler(console)
    root_logger.addHandler(console)
    root_logger.setLevel(logging.INFO)

    camera = ArcballCamera(
        size=(1024, 1024), fov=75, near=0.1, far=1000.0,
        position=[0, 2.0, 0],
        lookat=(0.0, 0.0, -0.0),
        up=(0.0, 0.0, -1.0))

    plane_size = 101
    plane_mesh = make_plane(plane_size, plane_size, nx=4, ny=4)
    plane_mesh.uv_scale = plane_size

    radmap = jsd.import_radiance_map(dict(path=args.radmap))
    material = AittalaMaterial(AittalaSVBRDF(args.brdf))
    scene = Scene()
    scene.set_radiance_map(radmap)
    scene.add_mesh(plane_mesh)
    scene.put_material('plane', material)

    renderer = MyRenderer(scene, camera, dpi=500,
                         gamma=args.gamma,
                         ssaa=args.ssaa,
                         tonemap=args.tonemap,
                         exposure=args.exposure,
                         reinhard_thres=args.reinhard_thres)
    app.run()
