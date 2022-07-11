"""
get camera priori.
"""

import config
import json
import sys

with open(config.PAIRS_JSON_PATH, 'w') as f1:
    all_pairs = json.load(f1)
print(len(all_pairs))

def get_blocks(azimuth_num, elevation_num):
       azi_start = 0.0
       azi_end = 6.11546875
       azi_pre = (azi_end - azi_start) / azimuth_num
       ele_start = 0.78515625
       ele_end = 1.777581
       ele_pre = (ele_end - ele_start) / elevation_num
       blocks_dict = {}
       total_blocks = azimuth_num * elevation_num
       for i in range(elevation_num):
              for j in range(azimuth_num):
                     blocks_dict[i * azimuth_num + j] = {}
                     blocks_dict[i * azimuth_num + j]['sta_azi'] = azi_start + j * azi_pre
                     blocks_dict[i * azimuth_num + j]['sta_ele'] = ele_start + i * ele_pre
                     blocks_dict[i * azimuth_num + j]['end_azi'] = blocks_dict[i * azimuth_num + j]['sta_azi'] + azi_pre
                     blocks_dict[i * azimuth_num + j]['end_ele'] = blocks_dict[i * azimuth_num + j]['sta_ele'] + ele_pre
                     blocks_dict[i * azimuth_num + j]['total_num'] = 0
                     blocks_dict[i * azimuth_num + j]['sta_num'] = 0
                     blocks_dict[i * azimuth_num + j]['end_num'] = 0
       return blocks_dict
       
blocks_dict = get_blocks(36,10)

num = 0
for pair in all_pairs:
    azi = pair['azimuth']
    ele = pair['elevation']
    for block in blocks_dict:
            if (azi >= blocks_dict[block]['sta_azi'] and azi < blocks_dict[block]['end_azi']) and \
                    ele >= blocks_dict[block]['sta_ele'] and ele < blocks_dict[block]['end_ele']:
                    blocks_dict[block]['total_num'] += 1
                    break
    num += 1

total_num = 0
for block in blocks_dict:
       if block == 0:
              blocks_dict[block]['sta_num'] = 0
              blocks_dict[block]['end_num'] = blocks_dict[block]['sta_num'] + blocks_dict[block]['total_num']
       else:
              blocks_dict[block]['sta_num'] = blocks_dict[block - 1]['end_num']
              blocks_dict[block]['end_num'] = blocks_dict[block]['sta_num'] + blocks_dict[block]['total_num']
              # print(blocks_dict[block]['end_num'])
       total_num += blocks_dict[block]['total_num']
       print('start_num')
       print(blocks_dict[block]['sta_num'])
       print('end_num')
       print(blocks_dict[block]['end_num'])
with open(config.CAMEA_POSE_PIROR_PATH, 'w') as f1:
       json.dump(blocks_dict,f1,indent=2)

print(total_num)