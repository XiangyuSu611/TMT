"""
Crop and center initial photographs.
"""

import sys
sys.path.append('/home/code/TMT/src/')
import config
import numpy as np
from pathlib import Path
from skimage.io import imread, imsave
from skimage.color import rgb2gray, gray2rgb
from skimage.transform import resize
from thirdparty.toolbox.toolbox.masks import mask_to_bbox, crop_bbox


def bright_pixel_mask(image, percentile=80):
    # get forward mask.
    image = rgb2gray(image)
    perc = np.percentile(np.unique(image), percentile)
    mask = image < perc
    return mask


def bright_pixel_bbox(image, percentile=80):
    # change mask to bounding box.
    bbox = mask_to_bbox(bright_pixel_mask(image, percentile))
    return bbox


def make_square_image(image, max_size):
    # re-save corpped image.
    fg_bbox = bright_pixel_bbox(image, percentile=80)
    cropped_im = crop_bbox(image, fg_bbox)
    output_im = np.full((max_size, max_size, 3), dtype=cropped_im.dtype,
                        fill_value=255)
    height, width = cropped_im.shape[:2]
    if height >= width:
        new_width = int(width * max_size / height)
        padding = (max_size - new_width) // 2
        cropped_im = resize(cropped_im, (max_size, new_width),
                            mode='constant', cval=255.0, anti_aliasing=True) * 255
        output_im[:, padding:padding + new_width] = cropped_im[:, :, :3]
    else:
        new_height = int(height * max_size / width)
        padding = (max_size - new_height) // 2
        cropped_im = resize(cropped_im, (new_height, max_size),
                            mode='constant', cval=255.0, anti_aliasing=True) * 255
        output_im[padding:padding + new_height, :] = cropped_im[:, :, :3]
    return output_im


def main(directory):
    directory = Path(directory).resolve()
    for path in sorted(directory.iterdir()):
        image = imread(str(Path(path, 'original.jpg')))
        if len(image.shape) == 2:
            image = gray2rgb(img)
        cropped_image = make_square_image(image, max_size=1000)
        imsave(str(Path(path, 'cropped.jpg')), cropped_image)


if __name__ == '__main__':
    directory = Path(config.DATA_ROOT, 'exemplars')
    main(directory)