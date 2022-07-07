import argparse

from . import parse_mdl


parser = argparse.ArgumentParser()
parser.add_argument(dest='path', type=str)
args = parser.parse_args()


def main():
    print(f'== {args.path}')
    print(parse_mdl(args.path))
    print()


if __name__ == '__main__':
    main()