import os
import argparse
import subprocess
import time
import tqdm
from pathlib import Path

import tempfile
import random
import multiprocessing


parser = argparse.ArgumentParser()
parser.add_argument(dest='in_dir', type=Path)
parser.add_argument(dest='out_dir', type=Path)
parser.add_argument('--blender-command', type=str, default='blender')
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=None)
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--reversed', action='store_true')
parser.add_argument('--num-workers', type=int, default=4)
args = parser.parse_args()


init_script = """
import bpy

scene = bpy.context.scene
scene.render.tile_x = 640
scene.render.tile_y = 640
scene.render.use_overwrite = False
scene.cycles.samples = 256
scene.cycles.device = 'GPU'
prefs = bpy.context.user_preferences.addons['cycles'].preferences
prefs.compute_device_type = 'CUDA'
prefs.devices[0].use = True
"""

convert_bl_script = """
bpy.ops.wm.addon_enable(module="materials_utils")
bpy.ops.ml.refresh()
"""


mtl_script = init_script


def worker(blend_path):
    out_path = (args.out_dir / f'{blend_path.stem}.0000.jpg')
    if not args.overwrite and out_path.exists():
        return

    with tempfile.NamedTemporaryFile(suffix='.py', mode='w+') as f:
        f.write(init_script)
        if blend_path.name.endswith('.mtl.blend'):
            f.write(convert_bl_script)
        f.flush()

        command = [
            args.blender_command,
            '-b', str(blend_path),
            '-o', str(args.out_dir / f'{blend_path.stem}.####'),
            '-F', 'JPEG',
            '-P', f.name,
            '-x', '1',
            '-f', '0',
        ]
        print(f' * Launching command {" ".join(command)}')
        subprocess.call(command)
    return blend_path.name


def main():
    args.out_dir.mkdir(exist_ok=True, parents=True)

    processed_set = set()

    pool = multiprocessing.Pool(processes=args.num_workers)

    while True:
        paths = list(args.in_dir.glob('*.blend'))
        paths = [
            p for p in paths
            if not (args.out_dir / f'{p.stem}.0000.png').exists()]
        print(len(paths))

        if args.shuffle:
            random.shuffle(paths)
        else:
            paths.sort(key=lambda x: os.path.getmtime(str(x)))
            paths = paths[args.start:args.end]
            if args.reversed:
                paths = paths[::-1]
        pbar = tqdm.tqdm(total=len(paths))
        try:
            for i in pool.imap_unordered(worker, paths):
                pbar.set_description(str(i))
                pbar.update(1)
            print(f"All done! Sleeping 20 seconds before checking for more..")
            time.sleep(20)
        except KeyboardInterrupt:
            break



if __name__ == '__main__':
    main()
