import json
import argparse

import rendkit
from rendkit import video
from rendkit.jsd import import_jsd_scene
from toolbox.logging import init_logger, disable_logging
import matplotlib

# matplotlib.verbose.set_level('debug')
rendkit.init_headless()
logger = init_logger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument(dest='jsd_path', type=str)
parser.add_argument(dest='out_path', type=str)
args = parser.parse_args()


def main():
    disable_logging('svbrdf')

    with open(args.jsd_path, 'r') as f:
        jsd_dict = json.load(f)

    logger.info("Loading scene.")
    scene = import_jsd_scene(jsd_dict, show_floor=True, shadows=True)
    size = video.scene_bounding_size(scene, longer_dim=1000)
    n_frames = 240
    logger.info("Rendering frames.")
    frames = video.render_frames(scene, n_frames=n_frames, size=size)
    logger.info("Saving video.")
    video.save_mp4(args.out_path, frames, size=size)


if __name__ == '__main__':
    main()
