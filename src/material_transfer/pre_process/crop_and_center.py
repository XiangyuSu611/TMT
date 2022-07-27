'''
crop images get croped files, centering and resize.
'''
import sys
sys.path.append('/home/code/TMT/src')
import numpy as np
import os
from pathlib import Path
from skimage.io import imread, imsave
from skimage.color import rgb2gray, gray2rgb
from skimage.transform import resize
from thirdparty.toolbox.toolbox.masks import mask_to_bbox, crop_bbox


def bright_pixel_mask(image, percentile=80):
    image = rgb2gray(image)
    perc = np.percentile(np.unique(image), percentile)
    mask = image < perc
    return mask


def bright_pixel_bbox(image, percentile=80):
    bbox = mask_to_bbox(bright_pixel_mask(image, percentile))
    return bbox


def make_square_image(image, max_size):
    fg_bbox = bright_pixel_bbox(image, percentile=80)
    bg = (image == (255,255,255,255)) * 1
    bg = (np.sum(bg,axis=2) != 4) * 255
    cropped_im = crop_bbox(image, fg_bbox)
    cropped_bg = crop_bbox(bg, fg_bbox)

    output_im = np.full((max_size, max_size, 3), dtype=cropped_im.dtype,
                        fill_value=255)
    output_bg = np.full((max_size, max_size, 1), dtype=cropped_im.dtype,
                        fill_value=0)

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


def main(oriDir):
    cropDir = oriDir.replace('removebg','cropped')
    tarDir = oriDir.replace('removebg','target')
    if not os.path.exists(cropDir): os.makedirs(cropDir)
    if not os.path.exists(tarDir): os.makedirs(tarDir)

    imgs = os.listdir(oriDir)
    imgs.sort()
    
    for index, img in enumerate(imgs):
        # center
        oriImg = imread(oriDir + img)
        cropImg = make_square_image(oriImg, max_size=1000)
        cropImgFile = cropDir + img.replace('.','_cropped.')
        imsave(cropImgFile, cropImg)
        # padding
        exemplar = imread(cropImgFile)
        exemplar_resize = resize(exemplar, (500,500))
        up = np.full((50,500,3), 1, dtype=np.uint8)
        exemplar_resize = np.append(up, exemplar_resize, axis=0)
        down = np.full((50,500,3), 1, dtype=np.uint8)
        exemplar_resize = np.append(exemplar_resize, down, axis=0)
        left = np.full((600,50,3), 1, dtype=np.uint8)
        exemplar_resize = np.append(left, exemplar_resize, axis=1)
        exemplar_resize = np.append(exemplar_resize, left, axis=1)
        exemplar_resize = resize(exemplar_resize, (500,500))
        exemplar_resize = exemplar_resize * 255
        exemplar_resize = exemplar_resize.astype(np.uint8)
        imsave(tarDir + 'render_val_'  + str(index + 1).zfill(8) + '.jpg', exemplar_resize)        


if __name__ == '__main__':
    main('/home/code/TMT/src/material_transfer/exemplar/wild_photo/removebg/')