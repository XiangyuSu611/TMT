"""
predict pixel-wised substance category.
"""

import sys
sys.path.append('/home/code/TMT/src/')
import config
import numpy as np
import skimage
import thirdparty.toolbox.toolbox.images as images

from pathlib import Path
from tqdm import tqdm
from skimage.io import imread, imsave
from thirdparty.kitnn.kitnn.models import minc

IMAGE_PAD_SIZE = 25
IMAGE_SHAPE = (500, 500)


def main(directory):
    # load MINC.
    print('Loading VGG...')
    mincnet = minc.MincVGG()
    mincnet.load_npy(config.MINC_VGG16_WEIGHTS_PATH)
    mincnet = mincnet.cuda()
    # load exemplars.
    directory = Path(directory).resolve()
    pbar = tqdm(directory.iterdir())
    for exemplar in pbar:
        pbar.set_description(f'{exemplar}: loading')
        image = imread(Path(exemplar, 'cropped.jpg'))
        iamge = skimage.img_as_float32(image)
        image = images.pad(image, IMAGE_PAD_SIZE, mode='constant')
        fg_mask = images.bright_pixel_mask(image, percentile=99)
        # predict substances.
        subst_map, _ = compute_substance_map(pbar, mincnet, image, fg_mask)
        subst_map = images.apply_mask(
            subst_map, fg_mask, fill=minc.REMAPPED_SUBSTANCES.index('background'))
        subst_map = images.unpad(subst_map, IMAGE_PAD_SIZE)
        subst_map_vis = images.visualize_map(
            subst_map,
            bg_value=minc.REMAPPED_SUBSTANCES.index('background'),
            values=list(range(0, len(minc.REMAPPED_SUBSTANCES))))
        # save data.
        pbar.set_description(f'{exemplar}: saving data')
        substance_map_path = Path(exemplar, 'image', config.EXEMPLAR_SUBST_MAP_NAME)
        substance_map_path.parent.mkdir(parents=True, exist_ok=True)
        imsave(substance_map_path, subst_map.astype(np.uint8))
        substance_vis_path = Path(exemplar, 'image', config.EXEMPLAR_SUBST_VIS_NAME)
        imsave(substance_vis_path, (subst_map_vis * 255).astype(np.uint8))        
       

def compute_substance_map(pbar, mincnet, image, fg_mask):
    processed_image = minc.preprocess_image(image)
    s = 1
    prob_maps, feat_dicts = minc.compute_probs_multiscale(
        processed_image, mincnet, use_cuda=True,
        scales=[1.0*s, 1.414*s, 1/1.414*s])
    prob_map_avg = minc.combine_probs(prob_maps, processed_image,
                                      remap=True, fg_mask=fg_mask)
    pbar.set_description("Running dense CRF")
    prob_map_crf = minc.compute_probs_crf(image, prob_map_avg)
    pbar.set_description("Resizing {} => {}"
          .format(prob_map_crf.shape[:2], processed_image.shape[:2]))
    prob_map_crf = images.resize(
        prob_map_crf, processed_image.shape[:2], order=2)
    subst_id_map = np.argmax(prob_map_crf, axis=-1)
    return subst_id_map, prob_map_crf

if __name__ == '__main__':
    directory = Path(config.DATA_ROOT, 'exemplars')
    main(directory)