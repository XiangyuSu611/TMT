import numpy as np
import warnings

HEADER_MAGIC = 'PF'


def pfm_read(filename, transposed=False):
    with open(filename, 'rb') as f:
        header_magic = f.readline().decode().strip()
        header_dims = f.readline().decode().strip()
        _ = f.readline().decode()
        width, height = [int(i) for i in header_dims.split(' ')]
        tex = np.fromfile(f, dtype=np.float32)
        dims = int(len(tex) / width / height)
        tex = np.squeeze(tex.reshape((height, width, dims)))
    if transposed:
        tex = np.swapaxes(tex, 0, 1)

    return tex


def pfm_write(filename, tex: np.ndarray):
    if tex.dtype != np.float32:
        warnings.warn('Input is not 32 bit precision: converting to 32 bits.')
        tex = tex.astype(np.float32)
    height, width = tex.shape[0], tex.shape[1]
    with open(filename, 'wb+') as f:
        f.write('{}\n'.format(HEADER_MAGIC).encode())
        f.write('{} {}\n'.format(width, height).encode())
        f.write('-1.0\n'.encode())
        f.write(tex.tobytes())
