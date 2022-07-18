import os
import numpy as np

from rendkit.envmap.io import unstack_cross, load_envmap, stack_cross
from rendkit.envmap.prefilter import prefilter_irradiance
from vispy import app
from matplotlib import pyplot as plt


_package_dir = os.path.dirname(os.path.realpath(__file__))
_resource_dir = os.path.join(_package_dir, '..', 'resources')
_cubemap_dir = os.path.join(_resource_dir, 'cubemaps')

app.use_app('glfw')


def main():
    app.Canvas(show=False)
    cube_faces = unstack_cross(stack_cross(
        load_envmap(os.path.join(_cubemap_dir, 'yokohama'))))
    irradiance_map = prefilter_irradiance(cube_faces)
    plt.imshow(np.vstack((stack_cross(cube_faces),
                          stack_cross(irradiance_map))))
    plt.show()


if __name__ == '__main__':
    main()
