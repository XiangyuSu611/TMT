"""
Computes aligning features for all shapes.
"""
import sys
sys.path.append('/home/code/TMT/src/')
import os
import config
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from skimage.io import imread
from skimage.filters import gaussian
from skimage.transform import resize
from tqdm import tqdm
from thirdparty.pyhog import pyhog

EXEMPLAR_SIZE = (100, 100)


def compute_HOG_features(image, bin_size, im_shape):
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


def parse_rend_filename(fname):
    fname, _ = os.path.splitext(fname)
    s = [s.split('=') for s in fname.split(',')]
    return {k: v for k, v in s}


def compute_features(path):
    image = imread(path)
    return compute_HOG_features(
        image, bin_size=config.ALIGN_BIN_SIZE, im_shape=config.ALIGN_IM_SHAPE)


def main(directory):
    shapes = sorted(directory.iterdir())
    pbar = tqdm(shapes)
    for shape in pbar:
        rend_dir = Path(shape, 'images/alignment/renderings')
        rend_paths = sorted(rend_dir.iterdir())
        pbar.set_description(f'Computing features for shape {shape}')
        with ProcessPoolExecutor(max_workers=8) as executor:
            feats = executor.map(compute_features, rend_paths)
        feats = np.vstack(feats).astype(dtype=np.float32)

        parsed_filenames = [parse_rend_filename(f.name) for f in rend_paths]
        fovs = [float(d['fov']) for d in parsed_filenames]
        thetas = [float(d['theta']) for d in parsed_filenames]
        phis = [float(d['phi']) for d in parsed_filenames]

        data = {
            'fovs': np.array(fovs, dtype=np.float16),
            'thetas': np.array(thetas, dtype=np.float16),
            'phis': np.array(phis, dtype=np.float16),
            'feats': feats
        }
        save_path = Path(shape, 'numpy', config.SHAPE_ALIGN_DATA_NAME)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            np.savez(f, data)

if __name__ == '__main__':
    directory = Path(config.DATA_ROOT, 'shapes')
    main(directory)