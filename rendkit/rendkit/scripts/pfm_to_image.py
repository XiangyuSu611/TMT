import argparse
import os

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

    im = pfm.pfm_read(args.path)

    if args.out_path:
        print("Saving image to {}".format(args.out_path))
        misc.imsave(args.out_path, im)


if __name__ == '__main__':
    main()
