"""
use pre-computed material groups.
"""
import sys
sys.path.append('/home/code/TMT/src')
import config
import os
import json
import skimage
import skimage.io
import numpy as np
from collections import Counter
from pathlib import Path
from random import choice

class UnionFindSet(object):
    def __init__(self, data_list):
        self.father_dict = {}
        self.size_dict = {}
        for node in data_list:
            self.father_dict[node] = node
            self.size_dict[node] = 1
    def find_head(self, node):
        father = self.father_dict[node]
        if(node != father):
            father = self.find_head(father)
        self.father_dict[node] = father
        return father
    def union(self, node_a, node_b):
        if node_a is None or node_b is None:
            return
        a_head = self.find_head(node_a)
        b_head = self.find_head(node_b)
        if(a_head != b_head):
            a_set_size = self.size_dict[a_head]
            b_set_size = self.size_dict[b_head]
            if(a_set_size >= b_set_size):
                self.father_dict[b_head] = a_head
                self.size_dict[a_head] = a_set_size + b_set_size
            else:
                self.father_dict[a_head] = b_head
                self.size_dict[b_head] = a_set_size + b_set_size

ori_jsons = sorted(config.BLENDER_JSON_PATH.iterdir())
merge_list = np.loadtxt('./src/data_generation/generation/reasonable_merge.txt').astype(np.uint8)

with open(Path(config.MATERIAL_ROOT, 'materials.json'), 'r') as f:
    materials = json.load(f)

mat_by_sub = {}
for mat in materials:
    if mat["substance"] not in mat_by_sub:
        mat_by_sub[mat["substance"]] = []
        mat_by_sub[mat["substance"]].append(mat["id"])
    else:
        mat_by_sub[mat["substance"]].append(mat["id"])

for ori in ori_jsons:
    with open(ori, 'r') as f1:
        ori_json = json.load(f1)
        ori_segment = ori_json['segment']
    segment_list = []
    for mat in ori_segment['segment_ids']:
        segment_list.append(int(mat[9:]) + 1)
    segment_list.sort()
    # segment_list = segment_list[1:]
    father_list = np.full(len(segment_list), 0, dtype=np.uint8)
    union_find_set = UnionFindSet(segment_list)
    for merge_id in range(len(merge_list)):
        if (merge_list[merge_id][0] + 1)  in segment_list and (merge_list[merge_id][1] + 1) in segment_list:
            union_find_set.union(merge_list[merge_id][0] + 1, merge_list[merge_id][1] + 1)
    for idx, segment in enumerate(segment_list):
        father_list[idx] = union_find_set.find_head(segment)
    # group tar json.
    tar_json = {}
    for item in ori_json:
        tar_json[item] = ori_json[item]
    tar_seg = {}
    tar_seg['segment_ids'] = ori_segment['segment_ids']
    tar_seg['segment_group_head_ids'] = {}
    for idx, segment in enumerate(segment_list):
        tar_seg['segment_group_head_ids']['material_' + str(segment - 1)] = 'material_' + str(father_list[idx] - 1)
    tar_seg['substances'] = {}
    for father in np.unique(father_list):
        decide_substance = []
        for idx in range(len(segment_list)):
            if father_list[idx] == father:
                decide_substance.append(ori_segment['substances']['material_' + str(segment_list[idx] - 1)])
        sorted_substance = list(Counter(decide_substance))
        for idx in range(len(segment_list)):
            if father_list[idx] == father:
                tar_seg['substances']['material_' + str(segment_list[idx] - 1)] = sorted_substance[0]
    
    # random assign mateirals.
    heads = {}
    for seg in tar_seg['segment_group_head_ids']:
        head = tar_seg['segment_group_head_ids'][seg]
        if head not in heads:
            heads[head] = {}
            substance = tar_seg['substances'][head]
            if substance == 'null':
                substance = choice(['wood', 'metal', 'plastic', 'leather', 'fabric'])
            heads[head]['substance'] = substance
            heads[head]['material_ids'] = choice(mat_by_sub[substance])
    selected_mats = {}
    for seg in tar_seg['segment_group_head_ids']:
        selected_mats[seg] = heads[tar_seg['segment_group_head_ids'][seg]]['material_ids']
    tar_seg['material_ids'] = selected_mats
    tar_json['tar_segment'] = tar_seg
    with open(ori, 'w') as f1:
        json.dump(tar_json, f1, indent=2)