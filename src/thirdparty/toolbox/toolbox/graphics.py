import numpy as np
from scipy import linalg


def euclidean_to_homogeneous(points):
    """
    Converts euclideans coordinates to homogeneous coordinates by appending a
    column of ones.
    :param points: points to convert
    :return: points homogeneous coordinates
    """
    ones = np.ones((points.shape[0], 1))
    return np.concatenate((points, ones), 1)


def homogeneous_to_euclidean(points):
    """
    Convertes homogeneous coordinates to euclidean coordinates by dividing by
    the last column
    and truncating it.
    :param points: points to convert
    :return: points in euclidean coordinates divided by the projective factor
    """
    ndims = points.shape[1]
    euclidean_points = np.array(points[:, 0:ndims - 1]) / points[:, -1, None]
    return euclidean_points


def extrinsic_to_lookat(extrinsic_mat):
    rotation_mat = extrinsic_mat[:3, :3]
    position = -extrinsic_mat[:3, 3]
    up = rotation_mat[1, :]
    forward = -rotation_mat[2, :]
    lookat = forward + position
    return position, lookat, up


def extrinsic_to_opengl_modelview(extrinsic_mat):
    """
    Converts extrinsic matrix to OpenGL format.
    :param extrinsic_mat: Extrinsic matrix in row-major order.
    :return: OpenGL view matrix in column-major order.
    """
    return np.vstack((extrinsic_mat, [0, 0, 0, 1]))


def intrinsic_to_opengl_projection(intrinsic_mat, left, right, top, bottom,
                                   near, far):
    """
    Converts intrinsic matrix to OpenGL format.
    :param intrinsic_mat: Intrinsic matrix in row-major order.
    :return: OpenGL perspective mat (including NDC matrix) in column-major
             format.
    """
    perspective_mat = np.vstack((
        np.hstack((intrinsic_mat[0, :], 0)),
        np.hstack((intrinsic_mat[1, :], 0)),
        [0, 0, near + far, near * far],
        np.hstack((intrinsic_mat[2, :], 0))
    ))
    ndc_mat = ortho(left, right, bottom, top, near, far).T

    return ndc_mat.dot(perspective_mat)


def unproject(width, height, projection_mat, modelview_mat,
              pixel_x: np.ndarray,
              pixel_y: np.ndarray,
              pixel_depth: np.ndarray):
    ndc_x = pixel_x / width * 2.0 - 1.0
    ndc_y = -(pixel_y / height * 2.0 - 1.0)
    ndc_z = pixel_depth * 2.0 - 1.0
    matrix = projection_mat.dot(modelview_mat)
    points = np.vstack((
        ndc_x.flatten(),
        ndc_y.flatten(),
        ndc_z.flatten(),
        np.ones(ndc_x.shape).flatten()
    ))
    unprojected = linalg.inv(matrix).dot(points).T
    unprojected = homogeneous_to_euclidean(unprojected)

    return unprojected


def compute_vertex_tight_clipping_planes(vertices, padding=0.1):
    near = np.abs(vertices[:, 2].max()) - padding
    far = np.abs(vertices[:, 2].min()) + padding
    return near, far


def compute_tight_clipping_planes(model, extrinsic, padding=0.1):
    augmented_vertices = euclidean_to_homogeneous(model.vertices)
    transformed_vertices = extrinsic.dot(augmented_vertices.T).T

    # Near and far are flipped so near is the max and far is the min.
    return compute_vertex_tight_clipping_planes(transformed_vertices, padding)


def ortho(left, right, bottom, top, znear, zfar):
    """
    Create an orthographic projection matrix.

    Copied from the Vispy project.

    :param left: left coordinate of the field of view
    :param right: right coordinate of the field of view
    :param bottom: bottom coordinate of the field of view
    :param top: top coordinate of the field of view
    :param znear: near coordinate of the field of view
    :param zfar: far coordinate of the field of view
    :return: an orthographic projection matrix (4x4)
    """
    assert(right != left)
    assert(bottom != top)
    assert(znear != zfar)

    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 / (right - left)
    M[3, 0] = -(right + left) / float(right - left)
    M[1, 1] = +2.0 / (top - bottom)
    M[3, 1] = -(top + bottom) / float(top - bottom)
    M[2, 2] = -2.0 / (zfar - znear)
    M[3, 2] = -(zfar + znear) / float(zfar - znear)
    M[3, 3] = 1.0
    return M
