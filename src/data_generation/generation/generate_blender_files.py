"""
generate final blender according to json files.
"""
import sys
sys.path.append('./')
import argparse
import json
import multiprocessing
import os
import subprocess
from functools import partial
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--inference_dir', default = f'./src/material_transfer/exemplar/final_prediction', type=Path)
parser.add_argument('--out_dir', default = f'./src/material_transfer/exemplar/blender', type=Path)
parser.add_argument('--num-workers', default=10, type=int)
parser.add_argument('--frontal', action='store_true')
parser.add_argument('--diagonal', action='store_true')
parser.add_argument('--no-floor', action='store_true')
args = parser.parse_args()

scene_types = {'inferred'}

def worker(gen_file, renderings_dir):
    inference_path = args.inference_dir / gen_file
    if not inference_path.exists():
        tqdm.write(f'{inference_path!s} does not exist')
        return
    for scene_type in scene_types:
        out_path = ( renderings_dir / f'{inference_path.stem}.{scene_type}.blend')
        command = [
            'python', '-m', 'finalResults.newdata_blender',
            str(inference_path),
            str(out_path),
            '--pack-assets',
        ]
        if args.frontal:
            command.append('--frontal')
        elif args.diagonal:
            command.append('--diagonal')
        if args.no_floor:
            command.append('--no-floor')
        print(f' * Launching command {command}')
        # command = ['ls']
        subprocess.call(command)
    return gen_file


def main():
    renderings_dir = args.out_dir
    renderings_dir.mkdir(parents=True, exist_ok=True)
    pool = multiprocessing.Pool(processes=args.num_workers)    
    gen_list = []
    all_files = os.listdir(args.inference_dir)
    for files in all_files:
        if '.json' in files:
            gen_list.append(files)
    pbar = tqdm(total=len(gen_list))
    for i in pool.imap_unordered(partial(worker, renderings_dir=renderings_dir), gen_list):
        pbar.set_description(str(i))
        pbar.update(1)


if __name__ == '__main__':
    main()