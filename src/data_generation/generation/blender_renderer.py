"""
find ./blender_fail -name '*000.inferred.blend'  -exec blender -b -P blender2.83_img_mask.py  -- {} \;
"""
import argparse
import bpy
import copy
import math
import numpy as np
import os as os 
import sys

from mathutils import Matrix, Vector

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--views', type=int, default=30, help='number of views to be rendered')
parser.add_argument('obj', type=str, help='Path to the obj file to be rendere.')
argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

# handle args.
print('args.obj:' + args.obj)

# set GPU render.
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'GPU'
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.preferences.addons["cycles"].preferences.get_devices()
for d in bpy.context.preferences.addons["cycles"].preferences.devices:
    # only use GPU.
    if d["name"] == 'Intel Xeon CPU E5-2630 v4 @ 2.20GHz':
        d["use"] = 0 
    else:
        d["use"] = 1
    print(d["name"], d["use"])
num_samples = 256

# load blender files.
bpy.ops.wm.open_mainfile(filepath=args.obj)
args.obj = args.obj[:-15] 

def get_material_names():
    mats = bpy.data.materials
    mat_list = []
    for m in mats:
        if m.name.startswith('material_'):
            mat_list.append(m.name)
    return mat_list

# set nodes.
bpy.context.scene.view_layers['RenderLayer'].use_pass_material_index = True
mat_list = get_material_names()
for mat in mat_list:
    bpy.data.materials[mat].pass_index = int(mat[9:]) + 1
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links
for n in tree.nodes:
    tree.nodes.remove(n)
render_layers = tree.nodes.new('CompositorNodeRLayers')
math_node = tree.nodes.new(type='CompositorNodeMath')
math_node.operation = 'DIVIDE'
math_node.inputs[1].default_value = 255.0
output_node_img = tree.nodes.new(type='CompositorNodeOutputFile')
output_node_img.format.color_mode = 'RGB'
output_node_img.format.file_format = 'JPEG'
output_node_seg = tree.nodes.new(type='CompositorNodeOutputFile')
output_node_seg.format.color_mode = 'BW'
links.new(render_layers.outputs['Image'], output_node_img.inputs[0])
links.new(render_layers.outputs['IndexMA'], math_node.inputs[0])
links.new(math_node.outputs['Value'], output_node_seg.inputs[0])


# set output path.
output_node_img.base_path = '/home/code/TMT/data/training_data/original'
output_node_seg.base_path = '/home/code/TMT/data/training_data/original'
if not os.path.exists(output_node_img.base_path):
    os.makedirs(output_node_img.base_path)
    

def vec_cos_theta(vec_a, vec_b):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)    
    len_a = np.linalg.norm(vec_a)
    len_b = np.linalg.norm(vec_b)
    return (np.dot(vec_a, vec_b) / ( len_a * len_b ))


def vec_to_euler(vec):
    y_cam_dir = copy.deepcopy(vec)
    y_cam_dir[1] = 0
    x_cam_dir = copy.deepcopy(vec)
    x_cam_dir[0] = 0
    z_cam_dir = copy.deepcopy(vec)
    z_cam_dir[2] = 0
    x_axis = np.array([1,0,0])
    y_axis = np.array([0,1,0])
    z_axis = np.array([0,0,1])
    x_rot_degree = math.acos(vec_cos_theta( vec, -z_axis ))
    z_rot_degree = math.acos(vec_cos_theta( z_cam_dir, y_axis ))
    if vec[0] > 0:
        z_rot_degree = -z_rot_degree
    if np.isnan(z_rot_degree):
        z_rot_degree = 0
    return [ x_rot_degree, 0, z_rot_degree ]


def euler_to_matrix(euler) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(euler[0]), -math.sin(euler[0]) ],
                    [0,         math.sin(euler[0]), math.cos(euler[0])  ]
                    ])

    R_y = np.array([[math.cos(euler[1]),    0,      math.sin(euler[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(euler[1]),   0,      math.cos(euler[1])  ]
                    ])

    R_z = np.array([[math.cos(euler[2]),    -math.sin(euler[2]),    0],
                    [math.sin(euler[2]),    math.cos(euler[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

def set_camera(cam_pos, look_at_pos, len_type='PERSP'):
    look_at_pos = np.array(look_at_pos)
    cam_pos = np.array(cam_pos)
    cam_dir = look_at_pos - cam_pos
    scene = bpy.data.scenes['Scene']
    camera = scene.camera
    if camera is None:
        camera = new_camera()
    camera.data.type = len_type
    if len_type == 'ORTHO':
        camera.data.ortho_scale = 5
    camera.rotation_mode = 'XYZ'
    camera.rotation_euler = vec_to_euler( cam_dir )
    camera.location = cam_pos
    cam_rot = np.array(camera.rotation_euler)
    return cam_rot, cam_dir / np.linalg.norm(cam_dir)

camera = bpy.data.objects['Camera']
# previous 1.15
camera.location = camera.location * 1.0
obj = bpy.data.objects['uvmapped_v2']

bpy.context.scene.render.resolution_x = 500
bpy.context.scene.render.resolution_y = 500

# get camera look at position at this time.
model = bpy.context.scene.objects['uvmapped_v2']
lowest_z = min([(model.matrix_world @ v.co).z for v in model.data.vertices])
highest_z = max([(model.matrix_world @ v.co).z for v in model.data.vertices])
lowest_y = min([(model.matrix_world @ v.co).y for v in model.data.vertices])
highest_y = max([(model.matrix_world @ v.co).y for v in model.data.vertices])
lowest_x = min([(model.matrix_world @ v.co).x for v in model.data.vertices])
highest_x = max([(model.matrix_world @ v.co).x for v in model.data.vertices])
look_at_pose = np.array([(highest_x + lowest_x) / 2, (highest_y + lowest_y) / 2, (highest_z + lowest_z) / 2])
# look_at_pose = np.array([0., 0., 0.])
camera = bpy.data.scenes['Scene'].camera
cam_dir = look_at_pose - np.array(camera.location)
camera.rotation_mode = 'XYZ'
camera.rotation_euler = vec_to_euler( cam_dir )


focus_point = [0.0, 0.0, 0.0]
light_distance = 5
light_names = ['Light_front', 'Light_back', 'Light_left', 'Light_right', 'Light_top', 'Light_bottom']
light_locations = []
for i in range(3):
    light_location = focus_point[:]
    light_location[i] -= light_distance
    light_locations.append(light_location)
    light_location = focus_point[:]
    light_location[i] += light_distance
    light_locations.append(light_location)
for i in range(len(light_names)):
    light_data = bpy.data.lights.new(name=light_names[i], type='POINT')
    light_data.energy = 666
    light_object = bpy.data.objects.new(name=light_names[i], object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    bpy.context.view_layer.objects.active = light_object
    light_object.location = light_locations[i]

for obj in bpy.data.objects:
    if obj.name == 'Plane' or obj.name == 'Cube':
        obj.select_set(True)
        bpy.ops.object.delete()

output_node_img.file_slots[0].path = args.obj + '_final_results'
output_node_seg.file_slots[0].path = args.obj + '_final_masks'
print(output_node_img.base_path + args.obj[1:] + '_final_results0001.jpg')
if not os.path.exists(output_node_img.base_path + args.obj[1:] + '_final_results0001.jpg'):
    bpy.ops.render.render(write_still=True)
