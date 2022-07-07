import os
import argparse
import json
import logging

import numpy as np

from toolbox.logging import init_logger
from vispy import app

from rendkit import jsd
from rendkit.camera import ArcballCamera
from rendkit.materials import AittalaMaterial

app.use_app('pyglet')

logger = init_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--ssaa', dest='ssaa', type=int, default=2)
parser.add_argument('--gamma', dest='gamma', type=float, default=2.2)
parser.add_argument('--tonemap', dest='tonemap', type=str, default=None)
parser.add_argument(
    '--reinhard-thres', dest='reinhard_thres', type=float, default=3.0)
parser.add_argument('--exposure', dest='exposure', type=float, default=1.0)
parser.add_argument('--radmap', dest='radmap', type=str, default=None)
parser.add_argument(dest='jsd_path', type=str)
args = parser.parse_args()

np.set_printoptions(suppress=True)


class MyJSDRenderer(jsd.JSDRenderer):
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
        elif event.key == '1':
            mat_names, mats = zip(*list(self.scene.materials.items()))
            self.current_mat_idx  = (self.current_mat_idx + 1) % len(mats)
            self.current_mat = mats[self.current_mat_idx]
            print("Selecting material {}: {}".format(
                self.current_mat_idx,
                mat_names[self.current_mat_idx]))
        elif event.key == 'a':
            if self.current_mat is not None:
                if isinstance(self.current_mat, AittalaMaterial):
                    self.current_mat.alpha -= 0.1
                    print('Setting alpha to {}'.format(self.current_mat.alpha))
                    self.recompile_renderables()
        elif event.key == 's':
            if self.current_mat is not None:
                if isinstance(self.current_mat, AittalaMaterial):
                    self.current_mat.alpha += 0.1
                    print('Setting alpha to {}'.format(self.current_mat.alpha))
                    self.recompile_renderables()
        elif event.key == 'z':
            if self.current_mat is not None:
                if isinstance(self.current_mat, AittalaMaterial):
                    self.current_mat.spec_shape_map[:, :, :] /= 1.1
                    print(np.mean(self.current_mat.spec_shape_map, axis=(0,1)))
                    self.recompile_renderables()
        elif event.key == 'x':
            if self.current_mat is not None:
                if isinstance(self.current_mat, AittalaMaterial):
                    self.current_mat.spec_shape_map[:, :, :] *= 1.1
                    print(np.mean(self.current_mat.spec_shape_map, axis=(0,1)))
                    self.recompile_renderables()
        self.update()
        self.draw()

    def recompile_renderables(self):
        for renderable in self.scene.renderables:
            renderable._program = renderable.compile(self.scene)

    def update_uv_scale(self, v):
        for renderable in self.scene.renderables:
            renderable.scale_uv_scale(v)
        self.update()


def main():
    with open(args.jsd_path, 'r') as f:
        jsd = json.load(f)

    if args.radmap is not None:
        radmap_dir = '/projects/grail/kpar/envmaps/'
        radmap_path = os.path.join(radmap_dir, args.radmap)
        logger.info("Overriding radmap {}".format(radmap_path))
        if not os.path.exists(radmap_path):
            logger.error("Given radiance map does not exist.")
            return
        jsd['radiance_map'] = {
            'path': os.path.join(radmap_dir, args.radmap)
        }

    camera = ArcballCamera(
        size=(900, 900), fov=90, near=1, far=1000.0,
        position=[0, 40, -60],
        lookat=(0.0, 0.0, -0.0),
        up=(0.0, 1.0, 0.0),
        clear_color=(1, 1, 1, 0))
    renderer = MyJSDRenderer(jsd, camera, dpi=500,
                            gamma=args.gamma,
                            ssaa=args.ssaa,
                            tonemap=args.tonemap,
                            exposure=args.exposure,
                            reinhard_thres=args.reinhard_thres,
                            show_floor=True)
    app.run()

if __name__ == '__main__':
    main()
