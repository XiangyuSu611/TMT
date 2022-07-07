"""
Some parts adapted from Magic UV:
    https://github.com/nutti/Magic-UV/blob/master/uv_magic_uv/common.py
"""
import math

from thirdparty.brender.brender.utils import check_version

import bpy
import bmesh


def calc_polygon_2d_area(points):
    area = 0.0
    for i, p1 in enumerate(points):
        p2 = points[(i + 1) % len(points)]
        v1 = p1 - points[0]
        v2 = p2 - points[0]
        a = v1.x * v2.y - v1.y * v2.x
        area = area + a

    return math.fabs(0.5 * area)


def calc_polygon_3d_area(points):
    area = 0.0
    for i, p1 in enumerate(points):
        p2 = points[(i + 1) % len(points)]
        v1 = p1 - points[0]
        v2 = p2 - points[0]
        cx = v1.y * v2.z - v1.z * v2.y
        cy = v1.z * v2.x - v1.x * v2.z
        cz = v1.x * v2.y - v1.y * v2.x
        a = math.sqrt(cx * cx + cy * cy + cz * cz)
        area = area + a

    return 0.5 * area


def measure_mesh_area(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    if check_version(2, 73, 0) >= 0:
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

    sel_faces = [f for f in bm.faces if f.select]

    # measure
    mesh_area = 0.0
    for f in sel_faces:
        verts = [l.vert.co for l in f.loops]
        f_mesh_area = calc_polygon_3d_area(verts)
        mesh_area = mesh_area + f_mesh_area

    return mesh_area


def measure_uv_area(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    if check_version(2, 73, 0) >= 0:
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

    if not bm.loops.layers.uv:
        return None
    uv_layer = bm.loops.layers.uv.verify()

    if not bm.faces.layers.tex:
        return None
    tex_layer = bm.faces.layers.tex.verify()

    sel_faces = [f for f in bm.faces if f.select]

    # measure
    uv_area = 0.0
    for f in sel_faces:
        uvs = [l[uv_layer].uv for l in f.loops]
        f_uv_area = calc_polygon_2d_area(uvs)

        # if not tex_layer:
        #     return None
        # img = f[tex_layer].image
        # # not found, try to search from node
        # if not img:
        #     for mat in obj.material_slots:
        #         for node in mat.material.node_tree.nodes:
        #             tex_node_types = [
        #                 'TEX_ENVIRONMENT',
        #                 'TEX_IMAGE',
        #             ]
        #             if (node.type in tex_node_types) and node.image:
        #                 img = node.image
        # if not img:
        #     return None
        # uv_area = uv_area + f_uv_area * img.size[0] * img.size[1]
        uv_area = uv_area + f_uv_area

    return uv_area


def measure_uv_density(obj):
    mesh_area = measure_mesh_area(obj)
    uv_area = measure_uv_area(obj)

    if not uv_area:
        return None, None, None

    if mesh_area == 0.0:
        density = 0.0
    else:
        density = math.sqrt(uv_area) / math.sqrt(mesh_area)

    return uv_area, mesh_area, density
