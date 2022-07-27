import os
import logging
import re
from collections import OrderedDict

import numpy as np

from  src.thirdparty.rendkit.meshkit import Mesh

OBJ_COMMENT_MARKER = '#'
OBJ_VERTEX_MARKER = 'v'
OBJ_NORMAL_MARKER = 'vn'
OBJ_UV_MARKER = 'vt'
OBJ_FACE_MARKER = 'f'
OBJ_MTL_LIB_MARKER = 'mtllib'
OBJ_MTL_USE_MARKER = 'usemtl'
OBJ_GROUP_NAME_MARKER = 'g'
OBJ_OBJECT_NAME_MARKER = 'o'

MTL_COMMENT_MARKER = '#'
MTL_NEWMTL_MARKER = 'newmtl'
MTL_SPECULAR_EXPONENT_MARKER = 'Ns'
MTL_SPECULAR_COLOR_MARKER = 'Ks'
MTL_DIFFUSE_COLOR_MARKER = 'Kd'
MTL_AMBIENT_COLOR_MARKER = 'Ka'
MTL_EMMISSIVE_COLOR_MARKER = 'Ke'

logger = logging.getLogger(__name__)


def __parse_face(parts, material_id, group_id, object_id):
    face_vertices = []
    face_normals = []
    face_uvs = []
    for i in [1, 2, 3]:
        face_vertex_def = parts[i]
        split = face_vertex_def.split('/')

        vertex_idx = int(split[0])
        uv_idx = int(split[1]) if (len(split) > 1 and
                                   len(split[1]) > 0) else 1
        normal_idx = int(split[2]) if (len(split) > 2 and
                                       len(split[2]) > 0) else 1

        face_vertices.append(vertex_idx - 1)
        face_normals.append(normal_idx - 1)
        face_uvs.append(uv_idx - 1)

    return {
        'vertices': face_vertices,
        'normals': face_normals,
        'uvs': face_uvs,
        'material': material_id,
        'group': group_id,
        'object': object_id,
    }


def save_obj_file(f, mesh: Mesh):
    close_file = False
    if isinstance(f, str):
        f = open(f, 'w')
        close_file = True
    for vertex in mesh.vertices:
        vert_str = ' '.join([str(p) for p in vertex])
        f.write('{} {}\n'.format(OBJ_VERTEX_MARKER, vert_str))
    for normal in mesh.normals:
        normal_str = ' '.join([str(p) for p in normal])
        f.write('{} {}\n'.format(OBJ_NORMAL_MARKER, normal_str))
    for uv in mesh.uvs:
        uv_str = ' '.join([str(p) for p in uv])
        f.write('{} {}\n'.format(OBJ_UV_MARKER, uv_str))

    cur_mat_id = None
    for face in mesh.faces:
        mat_id = face['material']
        if mat_id >= len(mesh.materials):
            logger.warning("Material {} is out of range.".format(mat_id))
            mat_id = None
        mat_name = mesh.materials[mat_id]
        if cur_mat_id != mat_id and mat_id != None:
            cur_mat_id = mat_id
            f.write("{} {}\n".format(OBJ_MTL_USE_MARKER, mat_name))
        face_str = ''
        for i in range(3):
            face_str += ' {}/{}/{}'.format(face['vertices'][i] + 1,
                                           face['uvs'][i] + 1,
                                           face['normals'][i] + 1)
        f.write("{}{}\n".format(OBJ_FACE_MARKER, face_str))

    f.flush()
    if close_file:
        f.close()


def read_obj_file(path):
    vertices = []
    faces = []
    normals = []
    uvs = []

    material_ids = OrderedDict([])
    material_counter = -1
    current_material_id = -1

    group_ids = OrderedDict({})
    group_counter = -1
    current_group_id = -1

    object_ids = OrderedDict({})
    object_counter = -1
    current_object_id = -1

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            # Ignore comments and whitespace.
            if len(line) < 3 or line[0] == OBJ_COMMENT_MARKER:
                continue

            parts = re.split(r'\s+', line)
            if parts[0] == OBJ_VERTEX_MARKER:
                vertex = [float(v) for v in parts[1:]]
                vertices.append(vertex)
            elif parts[0] == OBJ_NORMAL_MARKER:
                normal = [float(n) for n in parts[1:]]
                normals.append(normal)
            elif parts[0] == OBJ_UV_MARKER:
                uv = [float(u) for u in parts[1:]]
                uvs.append(uv)
            elif parts[0] == OBJ_FACE_MARKER:
                faces.append(__parse_face(parts,
                                          current_material_id,
                                          current_group_id,
                                          current_object_id))
            elif parts[0] == OBJ_MTL_USE_MARKER:
                material_name = parts[1]
                if material_name not in material_ids:
                    material_counter += 1
                    material_ids[material_name] = material_counter
                current_material_id = material_ids[material_name]
            elif parts[0] == OBJ_GROUP_NAME_MARKER:
                group_name = parts[1]
                if group_name not in group_ids:
                    group_counter += 1
                    group_ids[group_name] = group_counter
                current_group_id = group_ids[group_name]
            elif parts[0] == OBJ_OBJECT_NAME_MARKER:
                object_name = parts[1]
                if object_name not in object_ids:
                    object_counter += 1
                    object_ids[object_name] = object_counter
                current_object_id = object_ids[object_name]

    name = os.path.split(path)[-1]
    return Mesh(np.array(vertices, dtype=np.float32),
                np.array(normals, dtype=np.float32),
                np.array(uvs, dtype=np.float32),
                faces,
                list(material_ids.keys()),
                list(group_ids.keys()),
                list(object_ids.keys()),
                name=name)


def read_mtl_file(path, model):
    materials = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            # Ignore comments and whitespace.
            if len(line) < 3 or line[0] == OBJ_COMMENT_MARKER:
                continue
            parts = re.split(r'\s+', line)

            if parts[0] == MTL_NEWMTL_MARKER:
                material_name = parts[1]
                if material_name not in model.materials:
                    logger.warning('Material name {} not present in model'
                                   .format(material_name))
                materials[material_name] = WavefrontMaterial(material_name,
                                                             len(materials))
                current_material = materials[material_name]
            elif parts[0] == MTL_SPECULAR_EXPONENT_MARKER:
                current_material.specular_exponent = float(parts[1])
            elif parts[0] == MTL_SPECULAR_COLOR_MARKER:
                components = [float(c) for c in parts[1:]]
                current_material.specular_color = components
            elif parts[0] == MTL_DIFFUSE_COLOR_MARKER:
                components = [float(c) for c in parts[1:]]
                current_material.diffuse_color = components
            elif parts[0] == MTL_AMBIENT_COLOR_MARKER:
                components = [float(c) for c in parts[1:]]
                current_material.ambient_color = components
            elif parts[0] == MTL_EMMISSIVE_COLOR_MARKER:
                components = [float(c) for c in parts[1:]]
                current_material.emmissive_color = components

    return materials


class WavefrontMaterial:
    def __init__(self, name, index):
        self.name = name
        self.index = index
        self.specular_exponent = 2
        self.specular_color = (0.0, 0.0, 0.0)
        self.diffuse_color = (1.0, 1.0, 1.0)
        self.ambient_color = (0.0, 0.0, 0.0)
        self.emmissive_color = (0.0, 0.0, 0.0)

    def build_mtl(self):
        str = """
newmtl {}
Ns {}
Ks {} {} {}
Ka {} {} {}
Kd {} {} {}
Ke {} {} {}
illum 2
        """.format(self.name,
                   self.specular_exponent,
                   self.specular_color[0], self.specular_color[1],
                   self.specular_color[2],
                   self.ambient_color[0], self.ambient_color[1],
                   self.ambient_color[2],
                   self.diffuse_color[0], self.diffuse_color[1],
                   self.diffuse_color[2],
                   self.emmissive_color[0], self.emmissive_color[1],
                   self.emmissive_color[2])
        return str
