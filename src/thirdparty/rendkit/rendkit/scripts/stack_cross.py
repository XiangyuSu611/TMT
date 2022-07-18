import argparse
import os

from rendkit.envmap.io import stack_cross, unstack_cross, load_envmap
from vispy import app
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--format', dest='format', type=str, required=True)
args = parser.parse_args()


_package_dir = os.path.dirname(os.path.realpath(__file__))
_resource_dir = os.path.join(_package_dir, '..', 'resources')
_cubemap_dir = os.path.join(_resource_dir, 'cubemaps')

app.use_app('glfw')


def main():
    app.Canvas(show=False)
    cross = stack_cross(load_envmap(os.path.join(_cubemap_dir, 'yokohama')),
                        format=args.format)
    plt.imshow(cross)
    plt.show()
    plt.imshow(stack_cross(unstack_cross(cross), format='horizontal'))
    plt.show()


if __name__ == '__main__':
    main()
