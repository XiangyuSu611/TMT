import argparse
import os

import numpy as np
from matplotlib import pyplot as plt
from scipy import misc

from rendkit import pfm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='path', type=str)
    parser.add_argument('-o', '--out', dest='out_path', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print("File does not exist.")
        return

    im = pfm.pfm_read(args.path)
    print("min={}, max={}, median={}"
          .format(im.min(axis=(0,1)), im.max(axis=(0,1)), np.median(im, axis=(0,1))))

    plt.imshow(np.clip(im, 0, 1))
    plt.show()

    if args.out_path:
        print("Saving image to {}".format(args.out_path))
        misc.imsave(args.out_path, im)


if __name__ == '__main__':
    main()
