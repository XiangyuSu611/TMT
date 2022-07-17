import argparse
import os
import numpy as np
from matplotlib import pyplot as plt

from rendkit.envmap.conversion import panorama_to_cubemap, cubemap_to_panorama
from rendkit.envmap.io import stack_cross, load_envmap
from toolbox.io.images import save_hdr
from vispy import app
from rendkit.pfm import pfm_read, pfm_write


_package_dir = os.path.dirname(os.path.realpath(__file__))
_resource_dir = os.path.join(_package_dir, '..', 'resources')
_cubemap_dir = os.path.join(_resource_dir, 'cubemaps')

app.use_app('glfw')


parser = argparse.ArgumentParser()
parser.add_argument(dest='in_path', type=str)
parser.add_argument(dest='out_path', type=str)
parser.add_argument('--display', dest='display', action='store_true')
args = parser.parse_args()


def main():
    if not os.path.exists(args.in_path):
        print("Input file does not exist.")
        return
    if os.path.exists(args.out_path):
        overwrite = input("Overwrite? y/n: ")
        if overwrite != 'y':
            print("Aborting.")
            return

    cubemap = load_envmap(args.in_path)
    panorama = cubemap_to_panorama(cubemap)
    cross = stack_cross(cubemap)
    if args.display:
        plt.subplot(121)
        plt.imshow(np.clip(cross, 0, 1))
        plt.subplot(122)
        plt.imshow(np.clip(panorama, 0, 1))
        plt.show()
    print("Saving to {}".format(args.out_path))
    save_hdr(args.out_path, panorama)


if __name__ == '__main__':
    main()
