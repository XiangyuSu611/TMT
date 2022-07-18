import random

import numpy as np
from numpy import linalg
from scipy.spatial import Delaunay

from thirdparty.rendkit.rendkit import jsd
from thirdparty.rendkit.rendkit.vector_utils import normalized


def make_plane(height, width, material_name='plane',
               nx=1, ny=1):
    vertices = []
    uvs = []
    faces = []
    dx, dy = width / nx, height / ny
    for i in range(nx):
        for j in range(ny):
            base_idx = (nx * j + i) * 4
            offset_x = dx * i
            offset_y = dy * j
            vertices.extend([
                [offset_x + dx, 0.0, offset_y],
                [offset_x + dx, 0.0, offset_y + dy],
                [offset_x, 0.0, offset_y + dy],
                [offset_x, 0.0, offset_y]
            ])
            uvs.extend([
                [(i+1)/nx, j/ny],
                [(i+1)/nx, (j+1)/ny],
                [i/nx, (j+1)/ny],
                [i/nx, j/ny]
            ])
            faces.extend([
            {
                "vertices": [base_idx, base_idx+1, base_idx+2],
                "uvs": [base_idx, base_idx+1, base_idx+2],
                "normals": [0, 0, 0],
                "material": 0
            },
            {
                "vertices": [base_idx, base_idx+2, base_idx+3],
                "uvs": [base_idx, base_idx+2, base_idx+3],
                "normals": [0, 0, 0],
                "material": 0
            }
            ])

    return jsd.import_jsd_mesh({
        "type": "inline",
        "vertices": vertices,
        "uvs": uvs,
        "normals": [
            [0.0, 1.0, 0.0]
        ],
        "materials": [material_name],
        "faces": faces
    })


def make_random(size=100, material_name='plane'):
    t = np.linspace(0, np.pi, size)
    sig1 = (np.exp(-t * random.uniform(-1, 1))
            * np.cos(t * np.clip(random.gauss(0, 2), -2, 2)))
    sig2 = (np.exp(-t * random.uniform(-1, 0.5))
            * np.cos(t * np.clip(random.gauss(0, 2), -2, 2)))
    y = np.outer(sig1, sig2)
    y -= y.min()
    y = y / y.max() * 0.5
    xx = np.linspace(-1, 1, 100)
    x, z = np.meshgrid(xx, xx)
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    vertices = np.vstack((x, y, z)).T
    tri = Delaunay(vertices[:, [0, 2]])
    faces = []
    normals = np.zeros(vertices.shape)
    uvs = np.vstack((x, z)).T / 2.0 + 1.0
    for inds in tri.simplices:
        p1 = vertices[inds[0]]
        p2 = vertices[inds[1]]
        p3 = vertices[inds[2]]
        u = normalized(p1 - p2)
        v = normalized(p3 - p2)
        for i in inds:
            n = np.cross(u, v)
            # if np.dot(n, [0, 1, 0]) < 0:
            #     n *= -1
            normals[i] += n
        faces.append({
            "vertices": inds,
            "normals": inds,
            "uvs": inds,
            "material": 0,
        })
    normals = normals / linalg.norm(normals, axis=1)[:, None]
    return jsd.import_jsd_mesh({
        "scale": 1,
        "type": "inline",
        "vertices": vertices,
        "uvs": uvs,
        "normals": normals,
        "materials": [material_name],
        "faces": faces})
