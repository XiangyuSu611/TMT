"""
generate final json files.

python -m realimage.generate_blender_file \
    /mnt/new/zhuoyue/code/photoshape/photoshape-master/newdata/json_test/ \
    /mnt/new/zhuoyue/code/photoshape/photoshape-master/newdata/blender_fail/
"""
import os
import json
import visdom
import warnings
import argparse
import subprocess
import multiprocessing

from tqdm import tqdm
from pathlib import Path
from random import choice
from src.terial import models
from functools import partial
from src.terial import controllers
from src.terial.database import session_scope
from src.terial.models import ExemplarShapePair

parser = argparse.ArgumentParser()
parser.add_argument(dest='inference_dir', dafault = '/mnt/new/zhuoyue/code/photoshape/photoshape-master/newdata/json_test/', type=Path)
parser.add_argument(dest='out_name', default = '/mnt/new/zhuoyue/code/photoshape/photoshape-master/newdata/blender_fail/', type=Path)
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--output-mtl', action='store_true')
parser.add_argument('--mtl-only', action='store_true')
parser.add_argument('--use-minc-substances', action='store_true')
parser.add_argument('--frontal', action='store_true')
parser.add_argument('--diagonal', action='store_true')
parser.add_argument('--no-floor', action='store_true')
args = parser.parse_args()


scene_types = {'inferred'}
if args.output_mtl:
    scene_types.add('mtl')
if args.mtl_only:
    scene_types = {'mtl'}

def worker(gen_file, renderings_dir):
    inference_path = args.inference_dir / gen_file
    if not inference_path.exists():
        tqdm.write(f'{inference_path!s} does not exist')
        return
    for scene_type in scene_types:
        out_path = (
                renderings_dir / f'{inference_path.stem}.{scene_type}.blend')
        # if out_path.exists():
        #     continue
        command = [
            'python', '-m', 'realimage.newdata_blender',
            str(inference_path),
            str(out_path),
            '--type', scene_type,
            '--pack-assets',
        ]
        if args.frontal:
            command.append('--frontal')
        elif args.diagonal:
            command.append('--diagonal')
        if args.use_minc_substances:
            command.append('--use-minc-substances')
        if args.no_floor:
            command.append('--no-floor')
        print(f' * Launching command {command}')
        subprocess.call(command)
    return gen_file

    
def main():
    warnings.filterwarnings('ignore', '.*output shape of zoom.*')
    renderings_dir = args.inference_dir / args.out_name
    renderings_dir.mkdir(parents=True, exist_ok=True)

    pool = multiprocessing.Pool(processes=args.num_workers)
    
    gen_list = []
    all_files = os.listdir(args.inference_dir)
    for files in all_files:
        if '.json' in files:
            gen_list.append(files)
    gen_list.sort()
    gen_list = gen_list[0:]
    pbar = tqdm(total=len(gen_list))
    for i in pool.imap_unordered(partial(worker, renderings_dir=renderings_dir),
                                 gen_list):
        pbar.set_description(str(i))
        pbar.update(1)

if __name__ == '__main__':
    main()
