from contextlib import contextmanager

import bpy
import mathutils

from thirdparty.brender.brender.world_scale_uvs import measure_uv_density
from thirdparty.toolbox.toolbox import cameras


_context_stack = []


PASS_INDEX_UV_DENSITY_MULT = 10000.0


@contextmanager
def select_object(bobj):
    _context_stack.append(bpy.context.scene.objects.active)
    bpy.context.scene.objects.active = bobj
    bobj.select = True
    try:
        yield bobj
    finally:
        bobj.select = False
        bpy.context.scene.objects.active = _context_stack.pop()



@contextmanager
def stash_selection():
    was_selected = bpy.context.selected_objects
    bpy.ops.object.select_all(action='DESELECT')
    try:
        yield
    finally:
        bpy.ops.object.select_all(action='DESELECT')
        for bobj in was_selected:
            try:
                bobj.select = True
            except ReferenceError:
                pass


@contextmanager
def select_objects(bobjs):
    with stash_selection():
        for bobj in bobjs:
            bobj.select = True
        yield bobjs


@contextmanager
def edit_mode():
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        yield
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')


def unwrap_uv(bobj, mode):
    with select_object(bobj):
        with edit_mode():
            bpy.ops.mesh.select_all(action='DESELECT')
            bobj.select = True
            bpy.ops.mesh.select_all(action='SELECT')
            if mode == 'smart_uv':
                bpy.ops.uv.smart_project()
            elif mode == 'sphere':
                bpy.ops.uv.sphere_project()
            else:
                raise ValueError(f'Invalid unwrap mode {mode}')
            bobj.select = False


class Mesh:
    _next_id = 0
    _current = None
    _context_stack = []

    @classmethod
    def from_obj(cls, scene, path, size=None, **kwargs) -> 'Mesh':
        print(f'Importing mesh from {path}')

        with scene.select():
            bpy.ops.import_scene.obj(
                filepath=str(path),
                axis_forward='-Z', axis_up='Y',
                use_split_objects=False, use_split_groups=False,
                use_groups_as_vgroups=False, use_image_search=True)
            objects = bpy.context.selected_objects
            if size is not None:
                _resize_objects(objects, size, axis=None)
        return cls(objects, **kwargs)

    @classmethod
    def from_3ds(cls, scene, path) -> 'Mesh':
        print(f'Importing mesh from {path}')

        with scene.select():
            bpy.ops.import_scene.autodesk_3ds(filepath=str(path))
            objects = bpy.context.selected_objects
        return cls(objects)

    @classmethod
    def _load_objects_to_center(cls, bobjs, recenter):
        with select_objects(bobjs):
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
            center = mathutils.Vector((0, 0, 0))
            for bobj in bobjs:
                center += bobj.location / len(bobjs)
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=center)
        parent_bobj = bpy.context.selected_objects[0]
        bpy.ops.object.select_all(action='DESELECT')
        with select_objects(bobjs):
            parent_bobj.select = True
            bpy.ops.object.parent_set(type='OBJECT', keep_transform=False)
        if recenter:
            parent_bobj.location = (0, 0, 0)
        return parent_bobj

    def __init__(self, objects, name=None, recenter=False):
        self.do_not_delete = True
        if isinstance(objects, list):
            to_keep = set(
                o for o in objects if o.type == 'MESH' and o.name != 'floor')
            to_delete = [o for o in objects if o not in to_keep]

            with select_objects(to_delete):
                bpy.ops.object.delete()

            self.bobj = self._load_objects_to_center(to_keep, recenter)
            self.bobjs = to_keep
            self.do_not_delete = False
        else:
            self.bobj = objects
            self.bobjs = [self.bobj]

        if name is None:
            name = f'brender_mesh_{self._next_id}'
            Mesh._next_id += 1

        self.bobj.name = name

    def __del__(self):
        if not self.do_not_delete:
            bpy.data.objects.remove(self.bobj, do_unlink=True)

    @property
    def name(self):
        return self.bobj.name

    @property
    def mesh_bobjs(self):
        return [bobj for bobj in self.bobjs
                if bobj.type == 'MESH']

    def set_material(self, material: object) -> object:
        set_material(self.bobj, material)

    def set_rotation(self, rotation):
        self.bobj.rotation_euler = rotation

    def set_location(self, location):
        self.bobj.location = location

    def rotate(self, angle, axis):
        with select_object(self.bobj):
            bpy.ops.transform.rotate(value=angle, axis=axis)

    def resize(self, size, axis=None):
        _resize_objects([self.bobj], size, axis)

    def bounding_size(self):
        return _compute_bounding_size(self.bobj)

    def translate(self, amount):
        with select_object(self.bobj):
            bpy.ops.transform.translate(value=amount)

    def recenter(self):
        self.bobj.location = (0, 0, 0)

    def unwrap_uv(self, mode='smart_uv'):
        for bobj in self.mesh_bobjs:
            unwrap_uv(bobj, mode)

    def remove_doubles(self):
        for bobj in self.mesh_bobjs:
            with select_object(bobj):
                with edit_mode():
                    bpy.ops.mesh.remove_doubles()

    def make_normals_consistent(self):
        for bobj in self.mesh_bobjs:
            with select_object(bobj):
                with edit_mode():
                    bpy.ops.mesh.normals_make_consistent()

    def enable_smooth_shading(self):
        for bobj in self.mesh_bobjs:
            with select_object(bobj):
                with edit_mode():
                    bpy.ops.mesh.mark_sharp(clear=True, use_verts=True)
                    bpy.ops.mesh.faces_shade_smooth()

                bpy.ops.object.modifier_add(type='EDGE_SPLIT')
                bpy.context.object.modifiers["EdgeSplit"].split_angle = 1.32645
                # bpy.context.object.data.show_double_sided = True

    def set_uv_density(self, uv_density):
        for bobj in self.bobjs:
            with select_object(bobj):
                with edit_mode():
                    bobj.pass_index = PASS_INDEX_UV_DENSITY_MULT * uv_density

    def compute_uv_density(self, base_size=1.0):
        for bobj in self.bobjs:
            with select_object(bobj):
                with edit_mode():
                    _, _, uv_density = measure_uv_density(bobj)
                    uv_density *= base_size
                    bobj.pass_index = PASS_INDEX_UV_DENSITY_MULT * uv_density

    def compute_min_pos(self):
        return self.bobj.location[2] - _compute_bounding_size(self.bobj)[1] / 2
        # # return _compute_min_pos(self.bobj)
        # floor_pos = 0
        # for bpy_mesh in chain([self.bobj], self.bobj.children):
        #     if bpy_mesh.name == 'floor':
        #         continue
        #     if bpy_mesh.type == 'MESH' and bpy_mesh.data:
        #         floor_pos = min(
        #             floor_pos, min(p.co.y for p in bpy_mesh.data.vertices))
        # return 0, 0, floor_pos

    def get_materials(self):
       return get_material_list(self.bobj)


def _resize_objects(bobjs, size, axis=None):
    dimensions = _compute_bounding_size(bobjs)
    with select_objects(bobjs):
        # dimensions = self.bobj.children[0].dimensions
        if axis is None:
            scale = size / max(dimensions)
        else:
            scale = size / dimensions[axis]
        # for bobj in bobjs:
        #     _multiply_vertices(bobj, scale)
        bpy.ops.transform.resize(value=(scale, scale, scale))


def _multiply_vertices(bobj, scale):
    for child in bobj.children:
        _multiply_vertices(child, scale)

    if bobj.type == 'MESH' and bobj.data:
        for vertex in bobj.data.vertices:
            vertex.co *= scale


def _compute_bounding_size(bobjs):
    size = [0, 0, 0]
    if not isinstance(bobjs, list):
        bobjs = [bobjs]

    for bobj in bobjs:
        for i in range(3):
            size[i] = max(size[i], bobj.dimensions[i])

        for child in bobj.children:
            child_size = _compute_bounding_size(child)
            for i in range(3):
                size[i] = max(size[i], child_size[i])

    return size


def _compute_min_pos(bobj):
    if bobj.name == 'floor':
        return
    pos = float('inf')
    print(bobj.dimensions)
    for child in bobj.children:
        pos = min(pos, _compute_min_pos(child))

    if bobj.type == 'MESH' and bobj.data:
        pos = min(pos, min(p.co.y for p in bobj.data.vertices))

    return pos


class Empty(object):

    def __init__(self, position):
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=position)
        self.bobj = bpy.context.selected_objects[0]

    def set_parent_of(self, bobj):
        with select_object(self.bobj):
            bobj.select = True
            bpy.ops.object.parent_set(type='OBJECT', keep_transform=False)

    def rotate(self, angle, axis):
        with select_object(self.bobj):
            bpy.ops.transform.rotate(value=angle, axis=axis)

    def set_rotation(self, rotation):
        self.bobj.rotation_euler = rotation


class Sphere(Mesh):

    def __init__(self, position, radius,
                 segments=128, rings=64, **kwargs):
        bpy.ops.mesh.primitive_uv_sphere_add(
            size=radius, location=position, segments=128, ring_count=rings)
        bobj = bpy.context.selected_objects[0]
        bpy.ops.object.mode_set(mode='EDIT')
        bobj.select = True
        bpy.ops.mesh.faces_shade_smooth()
        bobj.select = False
        super().__init__([bobj], **kwargs)


class Monkey(Mesh):

    def __init__(self, position, **kwargs):
        bpy.ops.mesh.primitive_monkey_add(location=position, calc_uvs=True)
        bobj = bpy.context.selected_objects[0]
        with select_object(bobj):
            with edit_mode():
                bpy.ops.object.modifier_add(type='SUBSURF')
        super().__init__([bobj], **kwargs)


class Plane(Mesh):

    def __init__(self, position, radius=100, **kwargs):
        bpy.ops.mesh.primitive_plane_add(radius=radius, location=position)
        bobj = bpy.context.selected_objects[0]
        super().__init__([bobj], **kwargs)



def set_material(bobj, material):
    if hasattr(bobj, 'data') and bobj.data is not None:
        if hasattr(bobj.data, 'materials'):
            # assign to 1st material slot
            for k, v in enumerate(bobj.data.materials):
                bobj.data.materials[k] = material.bobj
        else:
            # no slots
            bobj.data.materials.append(material.bobj)

    bobj.active_material = material.bobj

    for child in bobj.children:
        set_material(child, material)


def get_material_list(bobj):
    materials = []
    if hasattr(bobj, 'data') and bobj.data is not None:
        if hasattr(bobj.data, 'materials'):
            # assign to 1st material slot
            for k, v in enumerate(bobj.data.materials):
                materials.append(v)

    for child in bobj.children:
        materials.extend(get_material_list(child))

    return materials


def align_mesh_to_direction(mesh, back_azimuth, back_elevation):
    """
    Changes the mesh matrix such that its front is facing the camera.
    :param mesh: the Brender mesh to align to the camera matrix.
    :param back_azimuth: the azimuth of the back facing direction.
    :param back_elevation: the elevation of the back facing direction.
    """
    cam_to_world = cameras.spherical_coord_to_cam(
        60.0, azimuth=back_azimuth, elevation=back_elevation).cam_to_world()

    # Ignore translation.
    cam_to_world[:, 3] = 0

    # Invert X and Z coordinates (Y being UP).
    cam_to_world[0, :] *= -1
    cam_to_world[2, :] *= -1

    bcam_to_world = cam_to_world[[0, 2, 1, 3], :]
    bcam_to_world[1] *= -1

    mesh.bobj.matrix_world = mathutils.Matrix(bcam_to_world.tolist())
