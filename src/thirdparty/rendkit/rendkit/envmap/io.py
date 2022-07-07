import logging
import os
from functools import partial
from time import time

import numpy as np
from scipy import misc
from scipy.misc import imread

from thirdparty.rendkit.rendkit import pfm
from thirdparty.toolbox.toolbox.images import resize

logger = logging.getLogger(__name__)


_FACE_NAMES = {
    '+x': 0,
    '-x': 1,
    '+y': 2,
    '-y': 3,
    '+z': 4,
    '-z': 5,
}


def _set_grid(grid: np.ndarray, height, width, u, v, value):
    grid[u*height:(u+1)*height, v*width:(v+1)*width] = value


def _get_grid(grid, height, width, u, v):
    return grid[u*height:(u+1)*height, v*width:(v+1)*width]


def stack_cross(cube_faces: np.ndarray, format='vertical'):
    _, height, width = cube_faces.shape[:3]
    n_channels = cube_faces.shape[3] if len(cube_faces.shape) == 4 else 1
    if format == 'vertical':
        result = np.zeros((height * 4, width * 3, n_channels))
        gridf = partial(_set_grid, result, height, width)
        gridf(0, 1, cube_faces[_FACE_NAMES['+y']])
        gridf(1, 0, cube_faces[_FACE_NAMES['-x']])
        gridf(1, 1, cube_faces[_FACE_NAMES['+z']])
        gridf(1, 2, cube_faces[_FACE_NAMES['+x']])
        gridf(2, 1, cube_faces[_FACE_NAMES['-y']])
        gridf(3, 1, np.fliplr(np.flipud(cube_faces[_FACE_NAMES['-z']])))
    elif format == 'horizontal':
        result = np.zeros((height * 3, width * 4, n_channels))
        gridf = partial(_set_grid, result, height, width)
        gridf(1, 2, cube_faces[_FACE_NAMES['+x']])
        gridf(1, 0, cube_faces[_FACE_NAMES['-x']])
        gridf(0, 1, cube_faces[_FACE_NAMES['+y']])
        gridf(2, 1, cube_faces[_FACE_NAMES['-y']])
        gridf(1, 1, cube_faces[_FACE_NAMES['+z']])
        gridf(1, 3, cube_faces[_FACE_NAMES['-z']])
    else:
        raise RuntimeError("Unknown format {}".format(format))
    return result


def unstack_cross(cross):
    if cross.shape[0] % 3 == 0 and cross.shape[1] % 4 == 0:
        format = 'horizontal'
        height, width = cross.shape[0] // 3, cross.shape[1] // 4
    elif cross.shape[0] % 4 == 0 and cross.shape[1] % 3 == 0:
        format = 'vertical'
        height, width = cross.shape[0] // 4, cross.shape[1] // 3
    else:
        raise RuntimeError("Unknown cross format.")

    gridf = partial(_get_grid, cross, height, width)

    if format == 'vertical' and np.all(gridf(1, 0) == 0):
        logger.info("Cubemap cross is flipped, flipping.")
        cross = np.flipud(cross)
        gridf = partial(_get_grid, cross, height, width)

    n_channels = cross.shape[2] if len(cross.shape) == 3 else 1
    faces_shape = ((6, height, width, n_channels)
                   if n_channels > 1 else (6, height, width))
    faces = np.zeros(faces_shape, dtype=np.float32)

    if format == 'vertical':
        faces[0] = gridf(1, 2)
        faces[1] = gridf(1, 0)
        faces[2] = gridf(0, 1)
        faces[3] = gridf(2, 1)
        faces[4] = gridf(1, 1)
        faces[5] = np.flipud(np.fliplr(gridf(3, 1)))
    elif format == 'horizontal':
        faces[0] = gridf(1, 2)
        faces[1] = gridf(1, 0)
        faces[2] = gridf(0, 1)
        faces[3] = gridf(2, 1)
        faces[4] = gridf(1, 1)
        faces[5] = gridf(1, 3)
    return faces


def load_envmap(path, size=(512, 512)):
    if not os.path.exists(path):
        raise FileNotFoundError("{} does not exist.".format(path))
    tic = time()
    ext = os.path.splitext(path)[1]
    shape = (*size, 3)
    cube_faces = np.zeros((6, *shape), dtype=np.float32)
    if os.path.isdir(path):
        for fname in os.listdir(path):
            name = os.path.splitext(fname)[0]
            image = misc.imread(os.path.join(path, fname))
            image = misc.imresize(image, size).astype(np.float32) / 255.0
            cube_faces[_FACE_NAMES[name]] = image
    elif ext == '.pfm':
        array = pfm.pfm_read(path)
        for i, face in enumerate(unstack_cross(array)):
            cube_faces[i] = resize(face, size)[:, :, :3]
    elif ext == '.exr' or ext == '.hdr':
        import cv2
        array = cv2.imread(str(path), -1)
        array = array[:, :, [2, 1, 0]]
        for i, face in enumerate(unstack_cross(array)):
            cube_faces[i] = resize(face, size)[:, :, :3]
    elif ext == '.jpg' or ext == '.png' or ext == '.tiff':
        array = imread(path)
        for i, face in enumerate(unstack_cross(array)):
            cube_faces[i] = resize(face, size)[:, :, :3]
    else:
        raise RuntimeError("Unknown cube map format.")
    logger.info("Loaded envmap from {} ({:.04f}s)".format(path, time() - tic))
    return cube_faces
