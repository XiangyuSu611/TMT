import os
import pandas
from pathlib import Path
import numpy as np
import skimage
import skimage.io
from collections import defaultdict
import shutil
import time

import torch
from tqdm import tqdm


print("This is l2-lab computing program!!!")

def mean_labdistance(source_labs, target_labs):
    distance = 0
    width = source_labs.shape[0]
    height = source_labs.shape[1]

    distance = ((source_labs - target_labs)**2).sum(dim=2).sqrt().sum()
    distance = distance / (width * height)

    return distance


def main():
    photo_path = './data/materials/material_harven_0610/'
    preview_path = './data/materials/preview_all_0610/'
    # similarity_matrix saving path
    save_path = './data/materials/material_harven_0610/similarity_matrix/'
    # the path to store the renderings of the topN similar materials retrieved from the similarity matrix
    vis_path = './data/materials/material_harven_0610/visualization/'
    # the path to store the scene preview images of the topN similar materials retrieved from the similarity matrix
    vis_preview_path = '/mnt/d/xiangyu/code/photoshape/data/blobs/materials/material_harven_0610/visualization_previewall/'
   
    # csv file name saving similarity matrix
    total_simi_matrix_name = 'total_similarity_matrix_sqrt.csv'
    subst_name_list = ['fabric', 'leather', 'metal', 'plastic', 'wood']

    photo_dir = os.listdir(photo_path)
   
    # a dictionary that stores the material id number and the corresponding image name
    pho_dict = {}
    # a dictionary that stores the material category name and its corresponding id numbers
    subst_id_dict = defaultdict(list)
    
    for photo in photo_dir:
        if photo.strip().split('.')[-1] != 'png':
            continue
        prefix = int(photo[photo.rfind('_') + 1:photo.rfind('.')])
        pho_dict[prefix] = photo_path + '/' + photo
        for subst in subst_name_list:
            if subst in photo:
                subst_id_dict[subst].append(prefix)
    list1 = list(pho_dict.keys())
    list1.sort()
    pho_dict = { k:pho_dict[k] for k in list1 }
    
    for subst in subst_name_list:
        subst_id_dict[subst].sort()

    preview_dir = os.listdir(preview_path)

    preview_dict = {}
    for preview in preview_dir:
        if preview.strip().split('.')[-1] != 'jpg':
            continue
        prefix = preview[preview.find('_') + 1:preview.rfind('final')-1]
        prefix = int(prefix[prefix.rfind('_') + 1:])
        preview_dict[prefix] = preview_path + '/' + preview

    list2 = list(preview_dict.keys())
    list2.sort()
    preview_dict = { k:preview_dict[k] for k in list2 }

    photo_number = len(preview_dict)

    # pre-calculate the lab format of the material image and store in labs_batch
    print('start precomputing the labs batch!')
    precompute_start_time = time.time()
    labs_batch = torch.zeros(photo_number, 960, 1280, 3)
    pbar = tqdm(total=len(pho_dict.keys()), desc='Trans',
                dynamic_ncols=True)
    for i, prefix_i in enumerate(pho_dict.keys()):
        image = skimage.io.imread(pho_dict[prefix_i])
        labs = skimage.color.rgb2lab(image)
        labs = torch.from_numpy(labs)
        labs_batch[i] = labs
        pbar.update()
    precompute_end_time = time.time()
    print('pre-compute: ' + str(precompute_end_time - precompute_start_time) + 's.')

    similarity_matrix = torch.zeros((photo_number, photo_number), dtype=torch.float32)
    print('相似性矩阵的长度为：' + str(photo_number))

    for i, prefix_i in enumerate(pho_dict.keys()):
        start_time = time.time()
        for j, prefix_j in enumerate(pho_dict.keys()):
            # only calculate the upper triangular matrix
            if j > i:
                source_labs = labs_batch[i].cuda()
                target_labs = labs_batch[j].cuda()
                mean_labdist = mean_labdistance(source_labs, target_labs)
                similarity_matrix[i][j] = mean_labdist
                similarity_matrix[j][i] = mean_labdist
        end_time = time.time()
        print('the time consuming of row ' + str(i+1) + ': ' + str(end_time - start_time) + 's.')

    similarity_matrix = similarity_matrix.numpy()
    total_data = pandas.DataFrame(similarity_matrix)
    total_data.columns = pho_dict.keys()
    total_data.index = pho_dict.keys()

    fabric_matrix = np.zeros((len(subst_id_dict['fabric']), len(subst_id_dict['fabric'])), dtype=float)
    leather_matrix = np.zeros((len(subst_id_dict['leather']), len(subst_id_dict['leather'])), dtype=float)
    metal_matrix = np.zeros((len(subst_id_dict['metal']), len(subst_id_dict['metal'])), dtype=float)
    plastic_matrix = np.zeros((len(subst_id_dict['plastic']), len(subst_id_dict['plastic'])), dtype=float)
    wood_matrix = np.zeros((len(subst_id_dict['wood']), len(subst_id_dict['wood'])), dtype=float)

    fabric2leather = np.zeros((len(subst_id_dict['fabric']), len(subst_id_dict['leather'])), dtype=float)
    fabric2metal = np.zeros((len(subst_id_dict['fabric']), len(subst_id_dict['metal'])), dtype=float)
    fabric2plastic = np.zeros((len(subst_id_dict['fabric']), len(subst_id_dict['plastic'])), dtype=float)
    fabric2wood = np.zeros((len(subst_id_dict['fabric']), len(subst_id_dict['wood'])), dtype=float)

    leather2metal = np.zeros((len(subst_id_dict['leather']), len(subst_id_dict['metal'])), dtype=float)
    leather2plastic = np.zeros((len(subst_id_dict['leather']), len(subst_id_dict['plastic'])), dtype=float)
    leather2wood = np.zeros((len(subst_id_dict['leather']), len(subst_id_dict['wood'])), dtype=float)

    metal2plastic = np.zeros((len(subst_id_dict['metal']), len(subst_id_dict['plastic'])), dtype=float)
    metal2wood = np.zeros((len(subst_id_dict['metal']), len(subst_id_dict['wood'])), dtype=float)

    plastic2wood = np.zeros((len(subst_id_dict['plastic']), len(subst_id_dict['wood'])), dtype=float)
    
    fabric_data = pandas.DataFrame(fabric_matrix)
    leather_data = pandas.DataFrame(leather_matrix)
    metal_data = pandas.DataFrame(metal_matrix)
    plastic_data = pandas.DataFrame(plastic_matrix)
    wood_data = pandas.DataFrame(wood_matrix)

    fabric2leather_data = pandas.DataFrame(fabric2leather)
    fabric2metal_data = pandas.DataFrame(fabric2metal)
    fabric2plastic_data = pandas.DataFrame(fabric2plastic)
    fabric2wood_data = pandas.DataFrame(fabric2wood)

    leather2metal_data = pandas.DataFrame(leather2metal)
    leather2plastic_data = pandas.DataFrame(leather2plastic)
    leather2wood_data = pandas.DataFrame(leather2wood)

    metal2plastic_data = pandas.DataFrame(metal2plastic)
    metal2wood_data = pandas.DataFrame(metal2wood)

    plastic2wood_data = pandas.DataFrame(plastic2wood)

    print('!!!!!')
    fabric_data.columns = subst_id_dict['fabric']
    fabric_data.index = subst_id_dict['fabric']

    leather_data.columns = subst_id_dict['leather']
    leather_data.index = subst_id_dict['leather']

    metal_data.columns = subst_id_dict['metal']
    metal_data.index = subst_id_dict['metal']

    plastic_data.columns = subst_id_dict['plastic']
    plastic_data.index = subst_id_dict['plastic']
    
    wood_data.columns = subst_id_dict['wood']
    wood_data.index = subst_id_dict['wood']

    fabric2leather_data.columns = subst_id_dict['leather']
    fabric2leather_data.index = subst_id_dict['fabric']
    fabric2metal_data.columns = subst_id_dict['metal']
    fabric2metal_data.index = subst_id_dict['fabric']
    fabric2plastic_data.columns = subst_id_dict['plastic']
    fabric2plastic_data.index = subst_id_dict['fabric']
    fabric2wood_data.columns = subst_id_dict['wood']
    fabric2wood_data.index = subst_id_dict['fabric']

    leather2metal_data.columns = subst_id_dict['metal']
    leather2metal_data.index = subst_id_dict['leather']
    leather2plastic_data.columns = subst_id_dict['plastic']
    leather2plastic_data.index = subst_id_dict['leather']
    leather2wood_data.columns = subst_id_dict['wood']
    leather2wood_data.index = subst_id_dict['leather']

    metal2plastic_data.columns = subst_id_dict['plastic']
    metal2plastic_data.index = subst_id_dict['metal']
    metal2wood_data.columns = subst_id_dict['wood']
    metal2wood_data.index = subst_id_dict['metal']

    plastic2wood_data.columns = subst_id_dict['wood']
    plastic2wood_data.index = subst_id_dict['plastic']

    for i in subst_id_dict['fabric']:
        for j in subst_id_dict['fabric']:
            if i > j:
                fabric_data[i][j] = total_data[i][j]
                fabric_data[j][i] = fabric_data[i][j]

    for i in subst_id_dict['leather']:
        for j in subst_id_dict['leather']:
            if i > j:
                leather_data[i][j] = total_data[i][j]
                leather_data[j][i] = leather_data[i][j]
    
    for i in subst_id_dict['metal']:
        for j in subst_id_dict['metal']:
            if i > j:
                metal_data[i][j] = total_data[i][j]
                metal_data[j][i] = metal_data[i][j]
    
    for i in subst_id_dict['plastic']:
        for j in subst_id_dict['plastic']:
            if i > j:
                plastic_data[i][j] = total_data[i][j]
                plastic_data[j][i] = plastic_data[i][j]
    
    for i in subst_id_dict['wood']:
        for j in subst_id_dict['wood']:
            if i > j:
                wood_data[i][j] = total_data[i][j]
                wood_data[j][i] = wood_data[i][j]

    for i in subst_id_dict['fabric']:
        for j in subst_id_dict['leather']:
            fabric2leather_data[j][i] = total_data[j][i]
    
        for j in subst_id_dict['metal']:
            fabric2metal_data[j][i] = total_data[j][i]
    
        for j in subst_id_dict['plastic']:
            fabric2plastic_data[j][i] = total_data[j][i]
    
        for j in subst_id_dict['wood']:
            fabric2wood_data[j][i] = total_data[j][i]

    for i in subst_id_dict['leather']:
        for j in subst_id_dict['metal']:
            leather2metal_data[j][i] = total_data[j][i]

        for j in subst_id_dict['plastic']:
            leather2plastic_data[j][i] = total_data[j][i]

        for j in subst_id_dict['wood']:
            leather2wood_data[j][i] = total_data[j][i]

    for i in subst_id_dict['metal']:
        for j in subst_id_dict['plastic']:
            metal2plastic_data[j][i] = total_data[j][i]

        for j in subst_id_dict['wood']:
            metal2wood_data[j][i] = total_data[j][i]
    
    for i in subst_id_dict['plastic']:
        for j in subst_id_dict['wood']:
            plastic2wood_data[j][i] = total_data[j][i]


    # save as .csv file
    total_data.to_csv(save_path + total_simi_matrix_name)
    fabric_data.to_csv(save_path + 'fabric_sqrt.csv')
    leather_data.to_csv(save_path + 'leather_sqrt.csv')
    metal_data.to_csv(save_path + 'metal_sqrt.csv')
    plastic_data.to_csv(save_path + 'plastic_sqrt.csv')
    wood_data.to_csv(save_path + 'wood_sqrt.csv')

    fabric2leather_data.to_csv(save_path + 'fabric2leather_sqrt.csv')
    fabric2metal_data.to_csv(save_path + 'fabric2metal_sqrt.csv')
    fabric2plastic_data.to_csv(save_path + 'fabric2plastic_sqrt.csv')
    fabric2wood_data.to_csv(save_path + 'fabric2wood_sqrt.csv')

    leather2metal_data.to_csv(save_path + 'leather2metal_sqrt.csv')
    leather2plastic_data.to_csv(save_path + 'leather2plastic_sqrt.csv')
    leather2wood_data.to_csv(save_path + 'leather2wood_sqrt.csv')

    metal2plastic_data.to_csv(save_path + 'metal2plastic_sqrt.csv')
    metal2wood_data.to_csv(save_path + 'metal2wood_sqrt.csv')

    plastic2wood_data.to_csv(save_path + 'plastic2wood_sqrt.csv')

    # simi_path = os.path.join(save_path, total_simi_matrix_name)
    # simi_matrix = np.array(pandas.read_csv(simi_path), dtype=float)
    # simi_matrix = simi_matrix[:, 1:simi_matrix.shape[1]]
    # total_data = pandas.DataFrame(simi_matrix)

    # total_data.columns = pho_dict.keys()
    # total_data.index = pho_dict.keys()

    # total_data.columns = subst_id_dict['wood']
    # total_data.index = subst_id_dict['wood']
    
    # find the top10 images with the smallest l2-lab distance of each material rendered image
    for ref_id in preview_dict.keys():
        # sort by distance at row ref_id
        sorted_total_data = total_data.sort_values(by=ref_id, axis=1, ascending=True)
        # Returns a list of id numbers sorted according to the l2-lab distance
        sorted_ids = list(sorted_total_data)
        for i, sorted_id in enumerate(sorted_ids[0:11]):
            if os.path.isfile(preview_dict[sorted_id]):
                shutil.copy(pho_dict[sorted_id], vis_path + '/ref_' + str(ref_id) + '_no_'+ str(i) 
                            + '_id_' + str(sorted_id)  + '_dis_' + str(sorted_total_data[sorted_id][ref_id]) + '.png')
                shutil.copy(preview_dict[sorted_id], vis_preview_path + '/ref_' + str(ref_id) + '_no_'+ str(i) 
                            + '_id_' + str(sorted_id)  + '_dis_' + str(sorted_total_data[sorted_id][ref_id]) + '.png')



if __name__ == '__main__':
    main()
