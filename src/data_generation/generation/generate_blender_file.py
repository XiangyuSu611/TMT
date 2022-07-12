import sys
sys.path.append('/home/code/TMT/src')
import argparse
import bpy
import config
import json
import os
from pathlib import Path
from data_generation.generation import blender
from thirdparty.brender import brender as brender

parser = argparse.ArgumentParser()
parser.add_argument(dest='inference_path', type=Path)
parser.add_argument(dest='out_path', type=Path)
parser.add_argument('--pack-assets', action='store_true')
parser.add_argument('--type', default='inferred', choices=['inferred', 'mtl'])
parser.add_argument('--animate', action='store_true')
parser.add_argument('--use-weighted-scores', action='store_true')
parser.add_argument('--use-minc-substances', action='store_true')
parser.add_argument('--frontal', action='store_true')
parser.add_argument('--diagonal', action='store_true')
parser.add_argument('--no-floor', action='store_true')
args = parser.parse_args()

if args.animate:
    _REND_SHAPE = (1080, 1920)
else:
    _REND_SHAPE = (1000, 1000)

def main():
    app = brender.Brender()
    # if os.path.exists(str(args.out_path)):
    #     return
    if not args.inference_path.exists():
        print(f' * {args.inference_path!s} does not exist.')
        return

    with args.inference_path.open('r') as f:
        inference_dict = json.load(f)

    with Path(config.MATERIAL_ROOT, 'materials.json').open('r') as f:
        materials_by_index = json.load(f)

    app.init()
    scene = blender.construct_realimage_scene_chair(
        app, inference_dict, materials_by_index, scene_type=args.type,
        rend_shape=_REND_SHAPE,
        frontal_camera=args.frontal,
        diagonal_camera=args.diagonal,
        add_floor=not args.no_floor)
    
    if args.animate:
        blender.animate_scene(scene)
    print(f' * Saving blend file to {args.out_path!s}')
    if args.pack_assets:
        bpy.ops.file.pack_all()
    else:
        bpy.ops.file.make_paths_absolute()
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out_path))
    scene.clear_bmats()
    
    print(f' * Saved!!!')

if __name__ == '__main__':
    main()
