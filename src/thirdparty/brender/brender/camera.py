import math
import bpy
import mathutils

import numpy as np
from scipy import linalg

from thirdparty.brender.brender.mesh import Empty


def _vec_len(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


class Camera:
    def __init__(self, bobj):
        self.bobj = bobj
        self.position = bobj.location
        self.camera_empty = Empty(position=(0, 0, 0))
        self.camera_empty.set_parent_of(bobj)
        self.dist_constraint = None
        self.base_dist = None

    def track_to(self, bobj=None):
        if bobj is None:
            bobj = self.camera_empty.bobj

        track_constraint = self.bobj.constraints.new('TRACK_TO')
        track_constraint.target = bobj
        track_constraint.track_axis = 'TRACK_NEGATIVE_Z'
        track_constraint.use_target_z = True
        track_constraint.up_axis = 'UP_Y'

        dist_constraint = self.bobj.constraints.new('LIMIT_DISTANCE')
        dist_constraint.target = bobj
        dist_constraint.distance = self.base_dist
        self.dist_constraint = dist_constraint

        return track_constraint

    def set_distance(self, dist):
        self.dist_constraint.distance = dist * self.base_dist


class BasicCamera(Camera):
    def __init__(self, scene, position, rotation):
        with scene.select():
            bpy.ops.object.camera_add()
            super().__init__(bpy.context.object)

        self.position = position

        # Default rotation.
        # TODO: generalize this.
        self.bobj.location = position

        # Upright to Z looking towards negative X
        self.bobj.rotation_mode = 'XYZ'
        self.bobj.rotation_euler = rotation
        self.base_dist = _vec_len(self.position)

    def set_location(self, location):
        self.bobj.location = location

    def set_rotation(self, rotation):
        self.bobj.rotation_euler = rotation


class CalibratedCamera(Camera):
    def __init__(self, scene, cam_to_world, fov):
        with scene.select():
            bpy.ops.object.camera_add()
            super().__init__(bpy.context.object)

        cam = self.bobj.data
        cam.name = 'CalibratedCamera'
        self.set_params(cam_to_world, fov)
        self.base_dist = _vec_len(self.position)

    def set_params(self, cam_to_world, fov):
        bcam_to_world = cam_to_world[[0, 2, 1, 3], :]
        bcam_to_world[1] *= -1
        cam = self.bobj.data
        cam.type = 'PERSP'
        cam.lens_unit = 'FOV'
        cam.angle = fov / 180 * math.pi
        self.bobj.matrix_world = mathutils.Matrix(bcam_to_world.tolist())
