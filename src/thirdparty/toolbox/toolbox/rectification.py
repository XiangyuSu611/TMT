import logging

import numpy as np
from numpy import linalg
from skimage import transform

from toolbox.stats import reject_outliers

logger = logging.getLogger(__name__)


def _find_3d_extrema(hull_verts_3d, centroid, u, v):
    """Searches for extrema in the u and v "axis" directions."""
    mags_u = np.sort(np.dot(hull_verts_3d - centroid, u))
    mags_v = np.sort(np.dot(hull_verts_3d - centroid, v))
    min_u = u * mags_u[0]
    max_u = u * mags_u[-1]
    min_v = v * mags_v[0]
    max_v = v * mags_v[-1]
    return np.array((centroid + min_u + min_v,
                     centroid + min_u + max_v,
                     centroid + max_u + max_v,
                     centroid + max_u + min_v))


def find_3d_bbox(coords_im, tangents_im, bitangents_im, region_mask) -> np.ndarray:
    """
    Finds the 3D bounding box of a planar region given a mask of a planar
    region. The bounding box is given as a 4-tuple which defines each corner
    of the bounding box in 3D.

    The algorithm is as follows:
        1. Use tangent/bitangent directions as u, v directions.
        2. Find extrema in the u, v directions which defines the corners.
    """
    hull_verts_3d = coords_im[region_mask]

    tangents = reject_outliers(tangents_im[region_mask], thres=1)
    bitangents = reject_outliers(bitangents_im[region_mask], thres=1)

    u = np.mean(tangents, axis=0)
    if linalg.norm(u) == 0:
        raise RuntimeError("Tangent is zero!")
    u /= linalg.norm(u)
    v = np.mean(bitangents, axis=0)
    if linalg.norm(v) == 0:
        raise RuntimeError("Tangent is zero!")
    v /= linalg.norm(v)

    centroid = np.mean(coords_im[region_mask], axis=0)
    return _find_3d_extrema(hull_verts_3d, centroid, u, v)


def compute_rectify_tform(corners, corners_3d, height=None, width=None,
                          scale=None):
    if width is None or height is None:
        height = linalg.norm(corners_3d[0] - corners_3d[1])
        width = linalg.norm(corners_3d[1] - corners_3d[2])
        if scale is None:
            max_len = max(linalg.norm(corners[0] - corners[1]),
                          linalg.norm(corners[2] - corners[3]),
                          linalg.norm(corners[1] - corners[2]),
                          linalg.norm(corners[0] - corners[3]))
            max_len_3d = max(height, width)
            scale = max_len / max_len_3d
            height, width = height * scale, width * scale
        else:
            height = linalg.norm(corners_3d[0] - corners_3d[1]) * scale
            width = linalg.norm(corners_3d[1] - corners_3d[2]) * scale
    reference_corners = np.array(
        ((0, 0), (height, 0), (height, width), (0, width)))
    tform = transform.ProjectiveTransform()
    tform.estimate(np.fliplr(reference_corners), np.fliplr(corners))
    return tform, height, width


def apply_rectify_tform(image, tform, height, width):
    """
    Rectifies a region of the image defined by the bounding box. The bounding
    box is a list of corners.

    Note that we need all 4 coordinates since the corners define a
    projectively transformed rectangle.
    """
    dtype = image.dtype
    im_min, im_max = image.min(), image.max()
    if im_min < -1 or im_max > 1:
        image = image.astype(dtype=float)
        image = (image - im_min) / (im_max - im_min)
    rectified_image = transform.warp(
        image, inverse_map=tform, output_shape=(int(height), int(width)),
        order=3)
    if im_min < -1 or im_max > 1:
        rectified_image = rectified_image * (im_max - im_min) + im_min
    return rectified_image.astype(dtype=dtype)


def compute_uv_texture_extents(uv_coords, shape=(1000, 1000)):
    uv_coords_max, uv_coords_min = uv_coords.max(axis=0), uv_coords.min(axis=0)
    logger.info("uv_coords_max={}, uv_coords_min={}"
                .format(uv_coords_max, uv_coords_min))
    u_len, v_len = uv_coords_max - uv_coords_min
    print(u_len, v_len)
    v_scale, u_scale = shape[0] * v_len, shape[1] * u_len
    ymin, xmin = uv_coords_min[1] * v_scale, uv_coords_min[0] * u_scale
    ymax, xmax = uv_coords_max[1] * v_scale, uv_coords_max[0] * u_scale
    return int(ymin), int(ymax), int(xmin), int(xmax)
