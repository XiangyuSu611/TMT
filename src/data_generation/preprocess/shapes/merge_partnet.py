"""
Merge different part of obj files, and preserve original part information at the same time -- chair
"""

import json
import os
from tqdm import tqdm


name2id = {}
# back.
name2id['chair'] = 0
name2id['chair_back'] = 1
name2id['back_support'] = 2
name2id['back_connector'] = 3
name2id['back_surface'] = 4
name2id['back_single_surface'] = 5
name2id['back_surface_horizontal_bar'] = 6
name2id['back_surface_vertical_bar'] = 7
name2id['back_frame'] = 8
name2id['back_holistic_frame'] = 9
name2id['back_frame_horizontal_bar'] = 10
name2id['back_frame_vertical_bar'] = 11

# seat.
# name2id['chair_seat'] = 12
# name2id['seat_surface'] = 13
name2id['seat_single_surface'] = 14
name2id['seat_surface_bar'] = 15
name2id['seat_frame'] = 16
name2id['seat_frame_bar'] = 17
name2id['seat_holistic_frame'] = 18
# name2id['seat_support'] = 19

# chair head.
name2id['chair_head'] = 20
name2id['head_connector'] = 21
name2id['headrest'] = 22

# chair arm.
name2id['chair_arm'] = 23
name2id['arm_connector'] = 24
name2id['arm_holistic_frame'] = 25
name2id['arm_writing_table'] = 26
name2id['arm_near_vertical_bar'] = 27
name2id['arm_horizontal_bar'] = 28
name2id['arm_sofa_style'] = 29

# footrest.
name2id['footrest'] = 30
# name2id['chair_base'] = 31
# name2id['chair_seat'] = 32
# name2id['seat_surface'] = 33
# name2id['seat_support'] = 34

# base.
# name2id['chair_base'] = 35
name2id['regular_leg_base'] = 36
# name2id['leg'] = 37
name2id['rocker'] = 38
name2id['bar_stretcher'] = 39
name2id['runner'] = 40
# name2id['foot'] = 41
name2id['star_leg_base'] = 42
name2id['mechanical_control'] = 43
name2id['lever'] = 44
name2id['knob'] = 45
# name2id['central_support'] = 46
name2id['star_leg_set'] = 47
# name2id['leg'] = 48
name2id['caster'] = 49
name2id['caster_stem'] = 50
name2id['wheel'] = 51
name2id['foot_base'] = 52
# name2id['foot'] = 53
name2id['pedestal_base'] = 54
# name2id['central_support'] = 55
name2id['pedestal'] = 56


partNet_path = './data/3D_Dataset/partNet/data_v0'


def deal_with_dict(dict, target_dict):
    if 'objs' in dict:
        prefex = dict['text']
        objs = dict['objs']
        for i in range(len(objs)):
            if objs[i] in target_dict:
                target_dict[objs[i]].append(prefex)
            else:
                target_dict[objs[i]] = []
                target_dict[objs[i]].append(prefex)
    if 'children' in dict:
        for i in range(len(dict['children'])):
            dict_children = dict['children'][i]
            deal_with_dict(dict_children, target_dict)
    else:
        return target_dict


def change_v(a):
    a = a[2:]
    v1 = int(a[0:a.find(' ')])
    a = a[a.find(' ') + 1:]
    v2 = int(a[0:a.find(' ')])
    a = a[a.find(' ') + 1:]
    v3 = int(a[0:a.find('\n')])
    return v1, v2, v3


def generate_final_obj(tar_dict, obj_list, obj_name):
    verticals = {}
    faces = []
    obj_list.sort()
    print(len(obj_list))
    for i in range(len(obj_list)):
        need_lan = 1
        prefix = tar_dict[obj_list[i][:-4]][-1]
        # print(prefix)
        if prefix == 'leg':
            if tar_dict[obj_list[i][:-4]][-2] == 'regular_leg_base':
                part_id = 37
            elif tar_dict[obj_list[i][:-4]][-2] == 'star_leg_set':
                part_id = 48
        elif prefix == 'foot':
            if tar_dict[obj_list[i][:-4]][-2] == 'regular_leg_base':
                part_id = 41
            elif tar_dict[obj_list[i][:-4]][-2] == 'foot_base':
                part_id = 53
        elif prefix == 'central_support':
            if tar_dict[obj_list[i][:-4]][-2] == 'star_leg_base':
                part_id = 46
            elif tar_dict[obj_list[i][:-4]][-2] == 'pedestal_base':
                part_id = 55
        elif prefix == 'chair_seat':
            if tar_dict[obj_list[i][:-4]][-2] == 'chair':
                part_id = 12
            elif tar_dict[obj_list[i][:-4]][-2] == 'footrest':
                part_id = 32
        elif prefix == 'seat_surface':
            if tar_dict[obj_list[i][:-4]][-3] == 'chair':
                part_id = 13
            elif tar_dict[obj_list[i][:-4]][-3] == 'footrest':
                part_id = 33
        elif prefix == 'seat_support':
            if tar_dict[obj_list[i][:-4]][-3] == 'chair':
                part_id = 19
            elif tar_dict[obj_list[i][:-4]][-3] == 'footrest':
                part_id = 34
        elif prefix == 'chair_base':
            if tar_dict[obj_list[i][:-4]][-2] == 'footrest':
                part_id = 31
            elif tar_dict[obj_list[i][:-4]][-2] == 'chair':
                part_id = 35
        else:
            part_id = name2id[prefix]
        with open(partNet_path + obj_name + '/objs/' + obj_list[i], 'r') as f:
            part_file = f.readlines()
            for j in range(len(part_file)):
                if 'v' in part_file[j]:
                    if part_file[j] not in verticals:
                        verticals[part_file[j]] = len(verticals) + 1
                elif 'f' in part_file[j]:
                    v1, v2, v3 = change_v(part_file[j])
                    v1_new = verticals[(part_file[v1 - 1])]
                    v2_new = verticals[(part_file[v2 - 1])]
                    v3_new = verticals[(part_file[v3 - 1])]
                    tar_face = 'f ' + str(v1_new) + ' ' + str(v2_new) + ' ' + str(v3_new) + '\n'
                    if need_lan == 1:
                        faces.append('g ' + prefix + '\nusemtl material_' + str(part_id) + '\n' + tar_face)
                        need_lan = 0
                    else:
                        faces.append(tar_face)

    with open('./data/3D_Dataset/shapes/partnet_merge_chair/' + obj_name + '.obj', 'w') as f1:
        f1.writelines(list(verticals.keys()) + faces)


def get_corr_list():
    corr_list = []
    obj_dir = os.listdir(partNet_path)
    matrics_json = open('./preprocess/chair_table_storagefurniture_bed_shapenetv1_to_partnet_alignment.json')
    matric = json.load(matrics_json)
    pbar = tqdm(range(0, len(obj_dir)))
    for i in pbar:
        pbar.set_description("Getting corr list")
        target_dir = partNet_path + '/' + obj_dir[i]
        jsonopen = open(target_dir + '/meta.json')
        jsonfile = json.load(jsonopen)
        if jsonfile['model_cat'] == 'Chair':
            if obj_dir[i] in matric:
                corr_list.append(obj_dir[i])
    return corr_list


def save_corr_list(corr_list, filename, mode='w'):
    with open(filename, mode) as f:
        json.dump(corr_list, f)


def read_corr_list(filename):
    with open(filename, 'r') as f:
        corr_list = json.load(f)
    return corr_list


def main():
    corr_list = get_corr_list()
    corr_list.sort()
    save_corr_list(corr_list, 'bed_corr_list.txt')
    print(len(corr_list))

    print('get corr_list ready.')
    pbar = tqdm(range(len(corr_list)))
    for i in pbar:
        pbar.set_description("Processing")
        dict_file = open(partNet_path + '/' + corr_list[i] + '/result_after_merging.json')
        source_dict = json.load(dict_file)[0]
        target_dict = {}
        deal_with_dict(source_dict, target_dict)
        print(target_dict)
        objs_path = partNet_path + corr_list[i] + '/objs'
        obj_dirs = os.listdir(objs_path)
        generate_final_obj(target_dict, obj_dirs, corr_list[i])


if __name__ == '__main__':
    main()