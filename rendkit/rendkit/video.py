import math
import numpy as np
from matplotlib import pyplot as plt, animation
from matplotlib.pyplot import tight_layout
from tqdm import trange, tqdm

from rendkit.camera import PerspectiveCamera
from rendkit.renderers import SceneRenderer
from toolbox.logging import init_logger

logger = init_logger(__name__)


def _scene_max_dims(scene):
    max_width = 10
    max_height = 10
    for mesh in scene.meshes:
        if mesh.name == 'floor':
            continue
        max_dims = mesh.max_dims()
        max_width = max(max_width, max_dims[0], max_dims[2])
        max_height = max(max_height, max_dims[1])
    return max_width, max_height


def scene_bounding_size(scene, longer_dim):
    max_width, max_height = _scene_max_dims(scene)
    ratio = max(max_width, max_height) / min(max_width, max_height)
    ratio = min(ratio, 1.5)
    if max_width > max_height:
        size = (longer_dim, longer_dim / ratio)
    else:
        size = (longer_dim / ratio, longer_dim)
    return size


def render_frames(scene, size, n_frames=240):
    frames = []
    pbar = tqdm(total=n_frames)
    for frame in frame_generator(scene, size, n_frames=n_frames):
        frames.append(frame)
        pbar.update(1)
    return frames


def frame_generator(scene, size, n_frames=240):
    max_width, max_height = _scene_max_dims(scene)
    camera = PerspectiveCamera(
        size=size, fov=75, near=0.1, far=1000.0,
        position=[0, 0, 0], lookat=(0.0, 0.0, -0.0),
        up=(0.0, 1.0, 0.0),
        clear_color=(1, 1, 1))
    renderer = SceneRenderer(scene, camera, tonemap='reinhard',
                             reinhard_thres=3.0,
                             gamma=2.2, ssaa=2)
    renderer.__enter__()
    for i in range(n_frames):
        angle = i * 2 * math.pi / (n_frames / 2)
        phi = angle / 2
        dist = max_width - max_width / 6 * math.cos(phi * 1.3)
        camera.position[:] = (
            dist * math.cos(angle),
            max_height / 2 + max_height / 8 + max_height / 2 * (
            math.log(n_frames) - math.log(i + 1)) / math.log(
                n_frames) * math.cos(phi),
            dist * math.sin(angle))
        camera.lookat[:] = (0, max_height / 3 * -math.sin(phi), 0)
        frame = np.clip(renderer.render_to_image(camera), 0, 1)
        yield frame
    renderer.close()


def save_mp4(path, frames, size, fps=24):
    fig = plt.figure(frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect('auto')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis('off')

    im = ax.imshow(frames[0], interpolation='bilinear')
    im.set_clim([0, 1])
    fig.set_size_inches([size[0]/100, size[1]/100])

    with tqdm(total=len(frames)) as pbar:
        def update_im(frame_no):
            pbar.update(1)
            im.set_data(frames[frame_no])
            return im
        ani = animation.FuncAnimation(fig, update_im, len(frames), interval=1)
        writer = animation.writers['ffmpeg'](
            fps=fps, extra_args=[
                # '-filter:v', 'minterpolate',
                '-crf', '18',
            ])
        ani.save(str(path), writer=writer, dpi=100)

    plt.close(fig)
