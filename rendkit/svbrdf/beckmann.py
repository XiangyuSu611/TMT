import time
from pathlib import Path
import numpy as np

from thirdparty.toolbox.toolbox import images
from thirdparty.toolbox.toolbox.io.images import save_image, save_hdr, load_hdr
from thirdparty.toolbox.toolbox.logging import init_logger

logger = init_logger(__name__)


DIFF_MAP_NAME = 'map_diffuse.exr'
SPEC_MAP_NAME = 'map_specular.exr'
NORMAL_MAP_NAME = 'map_normal.exr'
BLEND_NORMAL_MAP_NAME = 'map_normal_blender.png'
ROUGH_MAP_NAME = 'map_roughness.exr'
ANISO_MAP_NAME = 'map_anisotropy.exr'


class BeckmannSVBRDF:
    """
    The Anisotropic Beckmann SVBRDF class.

    This uses the Blender formulation for the anisotropic Beckmann BRDF.
    The roughness and anisotropy values map to the conventional alpha_x and
    alpha_y as:

      if(aniso < 0.0f) {
        alpha_x = roughness/(1.0f + aniso);
        alpha_y = roughness*(1.0f + aniso);
      }
      else {
        alpha_x = roughness*(1.0f - aniso);
        alpha_y = roughness/(1.0f - aniso);
      }

    The Aittala BRDF has many constant factors baked into the albedos. When
    converting from the Aittala BRDF you must
        1. Multiply the diffuse albedo by PI
        2. Divide the specular albedo by 4.0*PI
    to account for the baked in constants.
    """
    @classmethod
    def from_path(cls, path):
        path = Path(path)
        logger.info("[Beckmann] Loading from {}".format(path))
        return BeckmannSVBRDF(
            diffuse_map=load_hdr(path / DIFF_MAP_NAME),
            specular_map=load_hdr(path / SPEC_MAP_NAME),
            normal_map=load_hdr(path / NORMAL_MAP_NAME),
            roughness_map=load_hdr(path / ROUGH_MAP_NAME),
            anisotropy_map=load_hdr(path / ANISO_MAP_NAME),
            path=path, name=path.name)

    def __init__(self, diffuse_map, specular_map, normal_map, roughness_map,
                 anisotropy_map, path=None, name='unnamed',
                 suppress_outliers=True):
        self.diffuse_map = diffuse_map.astype(np.float32)
        self.specular_map = specular_map.astype(np.float32)
        self.normal_map = normal_map.astype(np.float32)
        self.roughness_map = roughness_map.astype(np.float32)
        self.anisotropy_map = anisotropy_map.astype(np.float32)
        self.path = path
        self.name = name

        if suppress_outliers:
            tic = time.time()
            self.diffuse_map = images.suppress_outliers(self.diffuse_map,
                                                        thres=4.5)
            self.specular_map = images.suppress_outliers(self.specular_map,
                                                         thres=4.5)
            logger.info("Suppressing outliers in diffuse and specular maps. "
                        "({:.04f}s)".format(time.time() - tic))

    def save(self, path):
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        logger.info("Saving Beckmann SVBRDF to {}".format(path))
        normal_map_blender = self.normal_map.copy()
        # Normalize X and Y to [0, 1] to follow blender conventions.
        normal_map_blender[:, :, :2] = 1.0
        normal_map_blender[:, :, :2] /= 2.0
        normal_map_blender = np.round(255.0 * normal_map_blender)\
            .astype(np.uint8)

        save_hdr(path / DIFF_MAP_NAME, self.diffuse_map)
        save_hdr(path / SPEC_MAP_NAME, self.specular_map)
        save_hdr(path / NORMAL_MAP_NAME, self.normal_map)
        save_image(path / BLEND_NORMAL_MAP_NAME, normal_map_blender)
        save_hdr(path / ROUGH_MAP_NAME, self.roughness_map)
        save_hdr(path / ANISO_MAP_NAME, self.anisotropy_map)

    def to_jsd(self, inline=False):
        if inline:
            return {
                'type': 'beckmann_inline',
                'params': {
                    'diffuse_map': self.diffuse_map,
                    'specular_map': self.specular_map,
                    'normal_map': self.normal_map,
                    'roughness_map': self.roughness_map,
                    'anisotropy_map': self.anisotropy_map,
                }
            }
        return {
            'type': 'beckmann',
            'path': str(self.path),
        }
