import math
import random
import numpy as np
from scipy import linalg
from scipy.stats import multivariate_normal

from rendkit.camera import PerspectiveCamera
from rendkit.vector_utils import normalized


def random_camera(size, cam_x_range, cam_z_range, cam_dist,
                  fov=None, up=(0, 0, -1), right=(1, 0, 0), position=(0, 1, 0)):
    up = normalized(np.array(up))
    right = normalized(np.array(right))
    position = normalized(np.array(position))
    if fov is None:
        fov = random.randrange(50, 60)
    lookat = np.zeros((3,))
    lookat += random.uniform(0, 1) * cam_z_range * up
    lookat += random.uniform(-1, 1) * cam_x_range * right

    camera = PerspectiveCamera(
        size=size, fov=fov, near=0.1, far=5000.0,
        position=cam_dist * np.array(position), clear_color=(0, 0, 0, 0),
        lookat=lookat, up=up)
    return camera


def random_hemisphere():
    value = np.zeros((3,))
    while linalg.norm(value) == 0:
        value = multivariate_normal.rvs(mean=(0, 0, 0))
    value[1] = abs(value[1])
    value = normalized(value)
    return value


def rodrigues(v, axis, rad):
    """
    Rotates the vector around an axis by an angle.
    :param v: vector to rotate.
    :param axis: axis to rotate about.
    :param rad: angle to rotate by.
    :return: rotated vector
    """
    return (v * np.cos(rad) + np.cross(axis, v) * np.sin(rad)
            + axis * np.dot(axis, v) * (1 - np.cos(rad)))


def random_rotation():
    normal = random_hemisphere()
    best_dot = 1.0
    best_axis = None
    for axis in [(1, 0, 0), (0, 0, 1), (0, 1, 0)]:
        dot = np.dot(axis, normal)
        if dot < best_dot:
            best_axis = axis
            best_dot = dot
    right = normalized(np.cross(best_axis, normal))
    right = rodrigues(right, normal, random.uniform(0, 2*np.pi))
    up = np.cross(normal, right)
    rot_mat = np.eye(4)
    rot_mat[0, :3] = right
    rot_mat[1, :3] = normal
    rot_mat[2, :3] = up
    return rot_mat.T
