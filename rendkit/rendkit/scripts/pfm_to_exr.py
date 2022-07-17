import os
import argparse
import cv2
from rendkit.pfm import pfm_write, pfm_read

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

    pfm = pfm_read(args.in_path)
    # OpenCV uses BGR.
    pfm = pfm[:, :, [2, 1, 0]]
    print("Saving to {}".format(args.out_path))
    im = cv2.imwrite(args.out_path, pfm)


if __name__ == '__main__':
    main()
