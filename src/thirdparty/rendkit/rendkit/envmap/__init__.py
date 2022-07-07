import time

import numpy as np

from thirdparty.rendkit.rendkit.envmap.conversion import cubemap_to_dual_paraboloid
from thirdparty.rendkit.rendkit.envmap.prefilter import prefilter_irradiance
from thirdparty.rendkit.rendkit.lights import logger
from thirdparty.vispy.vispy import gloo


class EnvironmentMap():
    def __init__(self, cube_faces: np.ndarray, scale=1.0):
        if cube_faces.shape[0] != 6:
            raise RuntimeError('Cubemap must have exactly 6 faces.')
        self._radiance_faces = None
        self._radiance_tex = None
        self._irradiance_faces = None
        self._irradiance_tex = None
        self._radiance_upper_map = None
        self._radiance_lower_map = None
        self._radiance_upper_tex = None
        self._radiance_lower_tex = None
        self.radiance_scale = scale
        self.ra_faces = cube_faces
        print('here!!!')

    @property
    def radiance_faces(self):
        return self._radiance_faces

    @radiance_faces.setter
    def radiance_faces(self, radiance_faces):
        self._radiance_faces = radiance_faces
        tic = time.time()
        self._radiance_upper_map, self._radiance_lower_map = \
            cubemap_to_dual_paraboloid(self.radiance_faces)
        logger.info("Computed dual paraboloid maps ({:.04f}s)."
                    .format(time.time() - tic))
        self._radiance_upper_tex = gloo.Texture2D(
            self._radiance_upper_map,
            interpolation='linear_mipmap_linear',
            internalformat='rgb32f',
            mipmap_levels=8)
        self._radiance_lower_tex = gloo.Texture2D(
            self._radiance_lower_map,
            interpolation='linear_mipmap_linear',
            internalformat='rgb32f',
            mipmap_levels=8)
        tic = time.time()
        self.irradiance_faces = prefilter_irradiance(
            self._radiance_faces,
            self._radiance_upper_map,
            self._radiance_lower_map)
        logger.info("Prefiltered irradiance map ({:.04f}s)."
                    .format(time.time() - tic))

    @property
    def irradiance_faces(self):
        return self._irradiance_faces

    @irradiance_faces.setter
    def irradiance_faces(self, irradiance_faces):
        self._irradiance_faces = irradiance_faces
        self._irradiance_tex = gloo.TextureCubeMap(
            self._irradiance_faces,
            interpolation='linear',
            internalformat='rgb32f')

    @property
    def radiance_upper_tex(self):
        return self._radiance_upper_tex

    @property
    def radiance_lower_tex(self):
        return self._radiance_lower_tex

    @property
    def irradiance_tex(self):
        return self._irradiance_tex

    @property
    def size(self):
        return (self.ra_faces.shape[2], self.ra_faces.shape[1])

    def reset(self):
        self.ra_faces = self._radiance_faces
