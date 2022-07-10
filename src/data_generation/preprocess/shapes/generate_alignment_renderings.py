"""
Render every shape in different camera angles
"""
import sys
sys.path.append('/home/code/TMT/src/')
import config
import math
import itertools
import numpy as np
from tqdm import tqdm


from pathlib import Path
from skimage.io import imsave
from thirdparty.toolbox.toolbox.images import crop_tight_fg, mask_bbox
from thirdparty.vispy.vispy import app
from thirdparty.rendkit.meshkit import wavefront
from thirdparty.rendkit.rendkit import jsd
from thirdparty.rendkit.rendkit.materials import BlinnPhongMaterial
from thirdparty.rendkit.rendkit.renderers import SceneRenderer
from thirdparty.rendkit.rendkit.camera import PerspectiveCamera
from thirdparty.rendkit.rendkit.scene import Scene

app.use_app('egl')


def spherical_to_cartesian(radius, azimuth, elevation):
    x = radius * math.cos(azimuth) * math.sin(elevation)
    y = radius * math.cos(elevation)
    z = radius * math.sin(azimuth) * math.sin(elevation)
    return (x, y, z)


def main(directory):
    print("Loading models.")
    shapes = sorted(directory.iterdir())
    # rendering scene generation.
    radmap_path = config.RADMAP_PATH
    radmap = jsd.import_radiance_map(dict(path=radmap_path))
    scene = Scene()
    scene.set_radiance_map(radmap)
    dist = 200
    camera = PerspectiveCamera(
        size=(200, 200), fov=0, near=0.1, far=5000.0,
        position=(0, 0, -dist), clear_color=(1, 1, 1, 0),
        lookat=(0, 0, 0), up=(0, 1, 0))
    renderer = SceneRenderer(scene, camera=camera,
                             gamma=2.2, ssaa=3,
                             tonemap='reinhard',
                             reinhard_thres=3.0,
                             show=False)
    print('renderer enter ok')
    renderer.__enter__()
    fovs = [50]
    elevations = np.linspace(math.pi/4, math.pi/2 + math.pi/16, 10)
    max_azimuth_samps = 36
    azimuth_by_elev = {}
    n_viewpoints = 0
    for phi in elevations:
        n_azimuths = int(round(max_azimuth_samps * math.sin(phi)))
        azimuth_by_elev[phi] = np.linspace(0, 2*math.pi, n_azimuths)
        n_viewpoints += n_azimuths
    # rendering.
    for i, shape in enumerate(shapes):
        last_name = "alignment/renderings/fov=50,theta=6.28318531,phi=1.76714587.png"
        mesh = wavefront.read_obj_file(shape.obj_path)
        mesh.resize(100)
        materials = wavefront.read_mtl_file(shape.mtl_path, mesh)
        scene.add_mesh(mesh)
        for mat_name, mat in materials.items():
            roughness = math.sqrt(2/(mat.specular_exponent + 2))
            scene.put_material(mat_name,
                               BlinnPhongMaterial(mat.diffuse_color,
                                                  mat.specular_color,
                                                  roughness))

        iterables = []
        for fov, phi in itertools.product(fovs, elevations):
            for theta in azimuth_by_elev[phi]:
                iterables.append((fov, theta, phi))

        pbar = tqdm(iterables)
        for fov, theta, phi in pbar:
            rend_fname = f"fov={fov},theta={theta:.08f},phi={phi:.08f}.png"
            data_name = f'alignment/renderings/{rend_fname}'
            if shape.data_exists(data_name):
                continue
            pbar.set_description(rend_fname)
            camera.position = spherical_to_cartesian(dist, theta, phi)
            camera.fov = fov
        
            # Set format to RGBA so we can crop foreground using alpha.
            image = renderer.render_to_image(format='rgba')
            image = np.clip(image, 0, 1)
            fg_bbox = mask_bbox(image[:, :, 3] > 0)
            image = crop_tight_fg(
                image[:, :, :3], (100, 100), bbox=fg_bbox, use_pil=True)
            imsave(data_name, (image * 255).astype(np.uint8))
        scene.clear()


if __name__ == '__main__':
    directory = Path(config.DATA_ROOT, 'shapes')
    main(directory)