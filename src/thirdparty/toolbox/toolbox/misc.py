import os
import logging
import numpy as np
from numpy.random import ranf

from toolbox.stats import gaussian2d


logger = logging.getLogger(__name__)


def random_radiance_map(size=50, n_gaussians=16):
    radmap = np.zeros((size, size))
    for i in range(n_gaussians):
        mean = ranf(2)*1.5 - 1.5/2
        sigma_min, sigma_max = 0.05, 0.4
        sigma = np.random.uniform(0.05, 0.4, 2)
        # Make smaller lights brighter.
        a = np.random.normal(-np.log(sigma.min())/(-np.log(sigma_min)), 1.0, 1)
        a = np.clip(a, 0, None)
        radmap += a * gaussian2d(mean=mean, sigma=sigma, size=radmap.shape)
    return radmap


def maybe_mkdirs(path):
    if not os.path.exists(path):
        logger.info("Creating {}".format(path))
        os.makedirs(path)
    return path
