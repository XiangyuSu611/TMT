import os
import pandas as pd
import numpy as np
import torch
import json
import random
from collections import defaultdict


def json_readfile(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def json_savefile(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)


def generate_triplet_for_each_subst(substance_id_list, id2matDis, mat_id_to_label):
    '''
    for all materials within each substance(materials category),
    randomly select material in the same substance as positive examples, 
    and select material with different substance and the similarity distance is slightly 
    greater than that of positive examples as negative examples.
    Use this rule to pre-sample a bunch of triples.
    Finally, triplets with the material label as the index is generated and saved in the txt file

    Args:
        substance_id_list: a list that stores material ids in each substance
        id2matDis: a dict that stores the similarity distance between each id material and other materials
        mat_id_to_label: a dict that stores the material id and corresponding label
    '''
    triplet_list = []
    for ref_id in substance_id_list:
        ref_label = mat_id_to_label[ref_id]
        for positive_id in substance_id_list:
            positive_label = mat_id_to_label[positive_id]
            if ref_id == positive_id:
                continue
            positive_dis = id2matDis[ref_id][mat_id_to_label[positive_id]].item()
            # find nagetive label
            temp_id2matDis = id2matDis[ref_id].copy()
            for idx in substance_id_list:
                temp_id2matDis[mat_id_to_label[idx]] = 0
            # if positive_dis is greater than the distance to other substance, skip this triple case
            if (temp_id2matDis > positive_dis).any() == False:
                continue
            min_negative_dis = np.min(id2matDis[ref_id][temp_id2matDis > positive_dis])
            negative_label = (id2matDis[ref_id] == min_negative_dis).nonzero()[0][0]
            negative_id = list(mat_id_to_label.keys())[list(mat_id_to_label.values()).index(negative_label)]
            # to facilitate post-processing, triples are stored in label format here
            triplet_list.append([int(ref_label), int(positive_label), int(negative_label)])

    return triplet_list


if __name__ == '__main__':
    # path that stores material renderings
    photo_path = './data/material/material_preview/material_harven_600/'
    # path that stores similarity matrix
    similarity_matrix_path = './data/materials/material_harven_600/similarity_matrix/total_similarity_matrix_sqrt.csv'
    # path that stores training data
    snapshot_dir = './data/training_data/material_prediction/'
    # filename of training triplet
    train_triplet_file = 'triplet_label_train.json'
    # filename of validation triplet
    test_triplet_file = 'triplet_label_test.json'
    meta_file = 'meta.json'

    subst_name_list = ['fabric', 'leather', 'metal', 'plastic', 'wood']

    meta_dict = json_readfile(snapshot_dir + meta_file)
    mat_id_to_label = {int(k): v
                       for k, v in meta_dict['mat_id_to_label'].items()}

    mat_dis = np.array(pd.read_csv(similarity_matrix_path), dtype=float)

    mat_id_list = list(mat_dis[:, 0].astype(int))
    mat_dis = mat_dis[:, 1:mat_dis.shape[1]]

    photo_dir = os.listdir(photo_path)
   
    subst_id_dict = defaultdict(list)
    
    for photo in photo_dir:
        if photo.strip().split('.')[-1] != 'png':
            continue
        prefix = int(photo[photo.rfind('_') + 1:photo.rfind('.')])
        for subst in subst_name_list:
            if subst in photo:
                subst_id_dict[subst].append(prefix)
    
    for subst in subst_name_list:
        subst_id_dict[subst].sort()

    id2matDis = {}
    for i, mat_id in enumerate(mat_id_list):
        list_i = mat_dis[i]
        id2matDis[mat_id] = np.array(list_i)

    total_triplet = []
    fabric_triplet = generate_triplet_for_each_subst(subst_id_dict['fabric'], id2matDis, mat_id_to_label)
    leather_triplet = generate_triplet_for_each_subst(subst_id_dict['leather'], id2matDis, mat_id_to_label)
    metal_triplet = generate_triplet_for_each_subst(subst_id_dict['metal'], id2matDis, mat_id_to_label)
    plastic_triplet = generate_triplet_for_each_subst(subst_id_dict['plastic'], id2matDis, mat_id_to_label)
    wood_triplet = generate_triplet_for_each_subst(subst_id_dict['wood'], id2matDis, mat_id_to_label)
    total_triplet.extend(fabric_triplet)
    total_triplet.extend(leather_triplet)
    total_triplet.extend(metal_triplet)
    total_triplet.extend(plastic_triplet)
    total_triplet.extend(wood_triplet)

    random.shuffle(total_triplet)

    train_length = int(len(total_triplet) *4/5)
    train_triplet = total_triplet[0:train_length]
    test_triplet = total_triplet[train_length:]

    train_triplet_data = {'train_answer_diff': train_triplet}
    test_triplet_data = {'test_answer_diff': test_triplet}

    json_savefile(snapshot_dir + train_triplet_file, train_triplet_data)
    json_savefile(snapshot_dir + test_triplet_file, test_triplet_data)