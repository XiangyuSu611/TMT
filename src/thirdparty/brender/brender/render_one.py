import argparse
import subprocess
from pathlib import Path

import tempfile


parser = argparse.ArgumentParser()
parser.add_argument(dest='in_path', type=Path)
parser.add_argument(dest='out_path', type=Path)
parser.add_argument('--blender-command', type=str, default='blender')
parser.add_argument('--resolution', type=str)
parser.add_argument('--num-samples', type=int, default=256)
args = parser.parse_args()


init_script = f"""
import bpy

scene = bpy.context.scene
scene.render.tile_x = 640
scene.render.tile_y = 640
scene.render.use_overwrite = False
scene.cycles.samples = {args.num_samples}
scene.cycles.device = 'GPU'
prefs = bpy.context.user_preferences.addons['cycles'].preferences
prefs.compute_device_type = 'CUDA'
prefs.devices[0].use = True
"""

if args.resolution:
    width, height = tuple(int(i) for i in args.resolution.split(','))
    init_script += \
f'''
scene.render.resolution_x = {width}
scene.render.resolution_y = {height}
''' \



def main():
    blend_path = args.in_path

    with tempfile.NamedTemporaryFile(suffix='.py', mode='w+') as f:
        f.write(init_script)
        f.flush()

        command = [
            args.blender_command,
            '-b', str(blend_path),
            '-o', str(args.out_path),
            '-F', 'JPEG',
            '-P', f.name,
            '-x', '1',
            '-f', '0',
        ]
        print(f' * Launching command {" ".join(command)}')
        subprocess.call(command)


if __name__ == '__main__':
    main()
