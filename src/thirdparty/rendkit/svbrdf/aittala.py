import os
from time import time

import numpy as np
from numpy import linalg
from scipy.special._ufuncs import gammaincinv, gamma
from thirdparty.toolbox.toolbox.images import resize

from thirdparty.rendkit.rendkit.pfm import pfm_read, pfm_write
from thirdparty.toolbox.toolbox import images
from thirdparty.toolbox.toolbox.logging import init_logger

logger = init_logger(__name__)


MAP_DIFF_FNAME = 'map_diff.pfm'
MAP_SPEC_FNAME = 'map_spec.pfm'
MAP_NORMAL_FNAME = 'map_normal.pfm'
MAP_SPEC_SHAPE_FNAME = 'map_spec_shape.pfm'
MAP_PARAMS_FNAME = 'map_params.dat'
IS_SIGMA_RANGE_FNAME = 'is_sigma_range.dat'
IS_CDF_FNAME = 'is_cdf_sampler.pfm'
IS_PDF_FNAME = 'is_pdf_sampler.pfm'


class AittalaSVBRDF:

    @classmethod
    def from_path(cls, path):
        return cls(path)

    def __init__(self, path=None,
                 diffuse_map=None,
                 specular_map=None,
                 spec_shape_map=None,
                 normal_map=None,
                 alpha=None,
                 cdf_sampler=None,
                 pdf_sampler=None,
                 sigma_min=None,
                 sigma_max=None,
                 suppress_outliers=True,
                 transposed=False):
        if path is not None:
            if not os.path.exists(str(path)):
                raise FileNotFoundError('The path {} does not exist'
                                        .format(path))

            self.path = str(path)

            data_path = os.path.join(self.path, 'out/reverse')

            with open(os.path.join(data_path, MAP_PARAMS_FNAME), 'r') as f:
                line = f.readline()
                self.alpha, _ = [float(i) for i in line.split(' ')]

            tic = time()
            self.diffuse_map = pfm_read(
                os.path.join(data_path, MAP_DIFF_FNAME), transposed=transposed)
            self.specular_map = pfm_read(
                os.path.join(data_path, MAP_SPEC_FNAME), transposed=transposed)
            self.normal_map = pfm_read(
                os.path.join(data_path, MAP_NORMAL_FNAME), transposed=transposed)
            self.spec_shape_map = pfm_read(
                os.path.join(data_path, MAP_SPEC_SHAPE_FNAME), transposed=transposed)

            logger.info("Loaded \'{}\', shape={}, alpha={} ({:.04f}s)"
                .format(self.name, self.diffuse_map.shape, self.alpha,
                        time() - tic))

            tic = time()
            is_cdf_path = os.path.join(data_path, IS_CDF_FNAME)
            is_pdf_path = os.path.join(data_path, IS_PDF_FNAME)
            is_sigma_range_path = os.path.join(data_path, IS_SIGMA_RANGE_FNAME)
            if os.path.exists(is_cdf_path):
                self.cdf_sampler = pfm_read(is_cdf_path)
                self.pdf_sampler = pfm_read(is_pdf_path)
                with open(is_sigma_range_path, 'r') as f:
                    line = f.readline()
                    self.sigma_min, self.sigma_max = [
                        float(i) for i in line.split(' ')]
                logger.info("Loaded existing importance sampling params. "
                            "({:.04f}s)".format(time() - tic))
            else:
                sigma, self.pdf_sampler, self.cdf_sampler = \
                    self.compute_is_params()
                self.sigma_min = sigma.min()
                self.sigma_max = sigma.max()
                logger.info("Saving CDF sampler to {}".format(is_cdf_path))
                pfm_write(is_cdf_path, self.cdf_sampler)
                logger.info("Saving PDF sampler to {}".format(is_pdf_path))
                pfm_write(is_pdf_path, self.pdf_sampler)
                with open(is_sigma_range_path, 'w') as f:
                    f.write("{} {}".format(self.sigma_min, self.sigma_max))
                logger.info("Computed importance sampling params. "
                            "({:.04f}s)".format(time() - tic))
        else:
            self.diffuse_map = diffuse_map
            self.specular_map = specular_map
            self.spec_shape_map = spec_shape_map
            self.normal_map = normal_map
            self.alpha = alpha
            if pdf_sampler is not None and cdf_sampler is not None:
                self.cdf_sampler = cdf_sampler
                self.pdf_sampler = pdf_sampler
                self.sigma_min, self.sigma_max = sigma_min, sigma_max
            else:
                sigma, self.pdf_sampler, self.cdf_sampler = \
                    self.compute_is_params()
                self.sigma_min, self.sigma_max = sigma.min(), sigma.max()

        if suppress_outliers:
            tic = time()
            self.diffuse_map = images.suppress_outliers(self.diffuse_map,
                                                        thres=4.5)
            self.specular_map = images.suppress_outliers(self.specular_map,
                                                         thres=4.5)
            logger.info("Suppressing outliers in diffuse and specular maps. "
                        "({:.04f}s)".format(time() - tic))

    def compute_is_params(self):
        S = self.spec_shape_map.reshape((-1, 3))
        S = S[:, [0, 2, 2, 1]].reshape((-1, 2, 2))

        # Approximate isotropic roughness with smallest eigenvalue of S.
        trace = S[:, 0, 0] + S[:, 1, 1]
        root = np.sqrt(np.clip(trace*trace - 4 * linalg.det(S), 0, None))
        beta = (trace + root) / 2
        sigma: np.ndarray = 1.0 / np.sqrt(beta)

        # Create 2D sample texture for sampling the CDF since we need different
        # CDFs for difference roughness values.
        xi_samps = np.linspace(0.0, 1, 256, endpoint=True)
        sigma_samps = np.linspace(sigma.min(), sigma.max(), 256)

        p = self.alpha / 2
        gamma_inv_xi_theta = gammaincinv(1 / p, xi_samps) ** p
        cdf_sampler = np.apply_along_axis(
            compute_cdf, 1, sigma_samps[:, None],
            gamma_inv_xi_theta=gamma_inv_xi_theta)
        pdf_sampler = np.apply_along_axis(
            compute_pdf, 1, sigma_samps[:, None],
            alpha=self.alpha, gamma_inv_xi_theta=gamma_inv_xi_theta)

        return sigma, pdf_sampler, cdf_sampler

    @property
    def name(self):
        return os.path.split(os.path.realpath(self.path))[1]

    def save(self, path):
        reverse_path = os.path.join(path, 'out', 'reverse')
        if not os.path.exists(reverse_path):
            os.makedirs(reverse_path)
        pfm_write(os.path.join(reverse_path, MAP_DIFF_FNAME),
                      self.diffuse_map)
        pfm_write(os.path.join(reverse_path, MAP_SPEC_FNAME),
                      self.specular_map)
        pfm_write(os.path.join(reverse_path, MAP_SPEC_SHAPE_FNAME),
                      self.spec_shape_map)
        pfm_write(os.path.join(reverse_path, MAP_NORMAL_FNAME),
                      self.normal_map)
        with open(os.path.join(reverse_path, MAP_PARAMS_FNAME), 'w') as f:
            f.write("{} {}".format(self.alpha, 0.0))

    def to_jsd(self, inline=True):
        if inline:
            return {
                'type': 'svbrdf_inline',
                'params': {
                    'diffuse_map': self.diffuse_map,
                    'specular_map': self.specular_map,
                    'spec_shape_map': self.spec_shape_map,
                    'normal_map': self.normal_map,
                    'alpha': self.alpha,
                    'pdf_sampler': self.pdf_sampler,
                    'cdf_sampler': self.cdf_sampler,
                    'sigma_min': self.sigma_min,
                    'sigma_max': self.sigma_max,
                }
            }
        return {
            'type': 'svbrdf',
            'path': self.path,
            'id': self.name,
            'name': self.name,
        }

    def resize(self, shape):
        self.diffuse_map = resize(self.diffuse_map, shape)
        self.specular_map = resize(self.specular_map, shape)
        self.spec_shape_map = resize(self.spec_shape_map, shape)
        self.normal_map = resize(self.normal_map, shape)

        return self


def compute_cdf(sigma, gamma_inv_xi_theta: np.ndarray):
    return np.arctan(sigma * gamma_inv_xi_theta)


def compute_pdf(sigma, alpha, gamma_inv_xi_theta):
    p = alpha / 2
    theta = compute_cdf(sigma, gamma_inv_xi_theta)
    norm = p / ((sigma ** 2) * np.pi * gamma(1 / p))
    return norm * np.exp(-((np.tan(theta) ** 2) / (sigma ** 2)) ** p)
