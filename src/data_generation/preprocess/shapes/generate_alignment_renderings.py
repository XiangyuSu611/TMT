"""
Render every shape in different camera angles
"""
import sys
sys.path.append('/home/code/TMT/src/')
import config
import itertools
import math
import numpy as np
import os
from pathlib import Path


def spherical_to_cartesian(radius, azimuth, elevation):
    x = radius * math.cos(azimuth) * math.sin(elevation)
    y = radius * math.cos(elevation)
    z = radius * math.sin(azimuth) * math.sin(elevation)
    return (x, y, z)


def save_cam_pose():
    if not os.path.exists(config.ALIGN_POSE_PATH):    
        dist = 200
        max_azimuth_samps = 36
        fovs = [50]
        # sample camera pose.
        elevations = np.linspace(math.pi/4, math.pi/2 + math.pi/16, 10)
        azimuth_by_elev = {}
        for phi in elevations:
            n_azimuths = int(round(max_azimuth_samps * math.sin(phi)))
            azimuth_by_elev[phi] = np.linspace(0, 2*math.pi, n_azimuths)
        iterables = []
        for fov, phi in itertools.product(fovs, elevations):
            for theta in azimuth_by_elev[phi]:
                iterables.append((fov, theta, phi))
        # save camera pose.
        np.save(config.ALIGN_POSE_PATH, np.array(iterables))
        

def main(directory):
    save_cam_pose()
    shapes = sorted(directory.iterdir())
    # render.
    for i, shape in enumerate(shapes):
        os.system(f'blender -b -P ./src/data_generation/preprocess/shapes/generate_alignment_rendering.py -- \
            {str(shape)}/models/uvmapped_v2.obj \
            {str(config.ALIGN_POSE_PATH)}')

if __name__ == '__main__':
    directory = Path(config.DATA_ROOT, 'shapes')
    main(directory)