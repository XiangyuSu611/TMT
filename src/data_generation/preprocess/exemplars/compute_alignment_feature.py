"""
compute  image HOG features. 
"""
import sys
sys.path.append('/home/code/TMT/src/')

import config
import numpy as np
import os
from pathlib import Path
from skimage.io import imread
from skimage.filters import gaussian
from skimage.transform import resize
from tqdm import tqdm
from thirdparty.pyhog import pyhog


def compute_features(image, bin_size, im_shape):
    # compute HOG features.
    image = resize(image, im_shape, anti_aliasing=True, mode='constant')
    image = gaussian(image, sigma=0.1, multichannel=True)
    image = image.astype(dtype=np.float32)
    for c in range(3):
        image[:, :, c] -= image.mean()
        image[:, :, c] /= image.std()

    padded = np.dstack([np.pad(image[:, :, d], bin_size,
                               mode='constant', constant_values=image.mean())
                        for d in range(image.shape[-1])])
    feat = pyhog.compute_pedro(padded.astype(dtype=np.float64), bin_size)
    feat = feat[:, :, -8:]
    feat = feat.reshape((1, -1))
    return feat.astype(np.float32)


def main(directory):
    pbar = tqdm(sorted(directory.iterdir()))
    for exemplar in pbar:
        pbar.set_description(f'{exemplar}')
        image = imread(Path(exemplar, 'cropped.jpg'))
        feature = compute_features(
            image,
            bin_size=config.ALIGN_BIN_SIZE,
            im_shape=config.ALIGN_IM_SHAPE)
        save_path = Path(exemplar, 'numpy/align_hog_8.npz')
        if not os.path.exists(save_path.parent):
            os.makedirs(save_path.parent)
        with open(save_path, 'wb') as f:
            np.savez(f, feature)


if __name__ == '__main__':
    directory = Path(config.DATA_ROOT, 'exemplars')
    main(directory)