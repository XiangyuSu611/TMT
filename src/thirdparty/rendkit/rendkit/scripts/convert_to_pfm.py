import os
import argparse
import cv2
from rendkit.pfm import pfm_write


parser = argparse.ArgumentParser()
parser.add_argument(dest='in_path', type=str)
parser.add_argument(dest='out_path', type=str)
args = parser.parse_args()


def main():
    if not os.path.exists(args.in_path):
        print("Input file does not exist.")
        return
    if os.path.exists(args.out_path):
        overwrite = input("Overwrite? y/n: ")
        if overwrite != 'y':
            print("Aborting.")
            return

    im = cv2.imread(args.in_path, -1)
    # OpenCV uses BGR.
    im = im[:, :, [2, 1, 0]]
    print("Saving to {}".format(args.out_path))
    pfm_write(args.out_path, im)


if __name__ == '__main__':
    main()
