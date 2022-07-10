# 需要修改
"""
Align corresponding shapes of ShapeNet with shapes of PartNet
"""

import json
import open3d as o3d
import numpy as np
import os
import shutil
from PIL import Image
from tqdm import tqdm

partNet_path = './data/3D_Dataset/partNet/data_v0/'
shapenet_path = './data/3D_Dataset/ShapeNetCore.v1/'
matrics_json = open('./preprocess/chair_table_storagefurniture_bed_shapenetv1_to_partnet_alignment.json')
matric = json.load(matrics_json)
category = 'chair'


def get_corr_dict():
    corr_dict = {}
    obj_dir = os.listdir(partNet_path)
    pbar = tqdm(range(0, len(obj_dir)))
    pbar.set_description("Getting corr dict")
    for i in pbar:
        target_dir = partNet_path + obj_dir[i]
        jsonopen = open(target_dir + '/meta.json')
        jsonfile = json.load(jsonopen)
        if jsonfile['model_cat'] == category.capitalize():
            if obj_dir[i] in matric:
                corr_dict[obj_dir[i]] = jsonfile['model_id']
    return corr_dict


def read_corr_dict(filename):
    with open(filename, 'r') as f:
        corr_dict = json.load(f)
    return corr_dict


def save_corr_dict(corr_list, filename, mode='w'):
    with open(filename, mode) as f:
        json.dump(corr_list, f)


def main():
    # corr_dict = read_corr_dict(category + '_corr_dict.txt')
    corr_dict = get_corr_dict()
    save_corr_dict(corr_dict, category + '_corr_dict.txt')

    corr_list = list(corr_dict.keys())
    corr_list.sort()
    pbar = tqdm(range(len(corr_list)))
    if not os.path.exists(f'./data/3D_Dataset/shapes/shapenet_align_{category}/'):
        os.makedirs(f'./data/3D_Dataset/shapes/shapenet_align_{category}/')
    for i in pbar:
        shapenet_source = corr_dict[corr_list[i]]
        shapenet_target_path = shapenet_path + '/03001627(chair)/03001627/' + shapenet_source + '/model.obj'
        # 由于某些 obj 文件的图片无法正常读取，会导致程序终止，因此终止后重新运行时之前遍历过的 obj 文件可以直接跳过处理
        if os.path.exists('./data/3D_Dataset/shapes/shapenet_align_'+ category + '/' + corr_list[i] + '.obj'):
            continue

        # 由于某些 obj 文件的图片会因为不是 jpg 格式而无法正常读取，会导致程序终止，
        # 为了防止程序终止，我们直接将 images 文件夹中的图片转化成 jpg 格式
        if os.path.exists(shapenet_target_path.replace('model.obj', 'images')):
            img_list = os.listdir(shapenet_target_path.replace('model.obj', 'images/'))
            for img in img_list:
                try:
                    texture_img = Image.open(shapenet_target_path.replace('model.obj', 'images/') + img)
                    texture_img.convert('RGB')
                    texture_img.save(shapenet_target_path.replace('model.obj', 'images/') + img)
                except:
                    shutil.rmtree(shapenet_target_path.replace('model.obj', 'images'))
                    break   
        
        ori_mesh = o3d.io.read_triangle_mesh(shapenet_target_path)
        transform_mat = np.array(matric[corr_list[i]]['transmat']).reshape(4, 4)
        mesh_tr = ori_mesh.transform(transform_mat)
        try:
            o3d.io.write_triangle_mesh('./data/3D_Dataset/shapes/shapenet_align_'+ category + '/' + corr_list[i] + '.obj', mesh_tr, write_triangle_uvs=False)            
        except:
            shutil.rmtree(shapenet_target_path.replace('model.obj', 'images'))
            i = i - 1
            continue


if __name__ == '__main__':
    main()