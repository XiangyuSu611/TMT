import argparse
import os

import numpy as np
from scipy import misc

from rendkit import pfm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='path', type=str)
    parser.add_argument('-o', '--out', dest='out_path', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print("File does not exist.")
        return

    im = misc.imread(args.path)
    print(im.min(), im.max())

    if im.dtype == np.uint8:
        im = im.astype(np.float32) / 255.0

    print("Saving PFM to {}".format(args.out_path))
    pfm.pfm_write(args.out_path, im)


if __name__ == '__main__':
    main()
