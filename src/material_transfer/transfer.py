"""
generate all results - image, (bin & conti) mask.
"""
import sys
sys.path.append('/home/code/TMT/src/image_translation')
import cv2
import data
import json
import numpy as np
import os
import pandas as pd
import random
import skimage.io
import skimage.transform
import torch
import torch.nn.functional as F
import torchvision.utils as vutils

from collections import OrderedDict
from collections import Counter
from models.pix2pix_model import Pix2PixModel
from options.test_options import TestOptions
from PIL import Image
from pathlib import Path, WindowsPath
from pho_predictor import transforms as pre_transforms
from pho_predictor.network.network_cls import FLModel
from skimage.io import imread
from skimage.color import rgb2gray
from scipy import ndimage as ndi
from torchvision import transforms, models
from torchvision.transforms import functional as T
from util.util import masktorgb

mat_dis_path = './src/image_translation/pho_predictor/mat_dis/total_similarity_matrix_sqrt_600.csv'
mat_dis = pd.read_csv(mat_dis_path, header=None)
# load id2matDis.
id2matDis = {}
for i in range(1,mat_dis.values.shape[0]):
    list_i = mat_dis.values[i]
    id2matDis[int(list_i[0])] = (list_i[1:])

snapshot_path = './src/image_translation/pho_predictor/checkpoint/meta_600.json'
with open(snapshot_path, 'r') as f:
    mat_id_to_label = json.load(f)['mat_id_to_label']
    label_to_mat_id = {v: k for k, v in mat_id_to_label.items()}

MATDIS_MAX = 999
SUBSTANCES = ['fabric','leather','wood','metal','plastic']
COLOR_PLATTER = [(255, 255, 255), (255, 25, 0), (255, 50, 0), (255, 76, 0), (255, 102, 0), (255, 127, 0), (255, 153, 0), (255, 178, 0), (255, 204, 0), (255, 229, 0), 
                (255, 255, 0), (229, 255, 0), (203, 255, 0), (178, 255, 0), (153, 255, 0), (127, 255, 0), (101, 255, 0), (76, 255, 0), (51, 255, 0), (25, 255, 0), 
                (0, 255, 0), (0, 255, 25), (0, 255, 50), (0, 255, 76), (0, 255, 102), (0, 255, 127), (0, 255, 153), (0, 255, 178), (0, 255, 203), (0, 255, 229), 
                (0, 255, 255), (0, 229, 255), (0, 203, 255), (0, 178, 255), (0, 153, 255), (0, 127, 255), (0, 102, 255), (0, 76, 255), (0, 51, 255), (0, 25, 255), 
                (0, 0, 255), (25, 0, 255), (50, 0, 255), (76, 0, 255), (101, 0, 255), (127, 0, 255), (152, 0, 255), (178, 0, 255), (204, 0, 255), (229, 0, 255), 
                (255, 0, 255), (255, 0, 229), (255, 0, 203), (255, 0, 178), (255, 0, 152), (255, 0, 127), (255, 0, 102), (255, 0, 76), (255, 0, 51), (255, 0, 25)]

irange = range
opt = TestOptions().parse()
torch.manual_seed(6666)

dataloader = data.create_dataloader(opt)
model = Pix2PixModel(opt)
model.eval()
save_root = os.path.join('/home/code/TMT/src/material_transfer/exemplar')

# predictor 1.
check_point_path = Path('./src/image_translation/pho_predictor/checkpoint/model-path1.best_pth.tar')
checkpoint = torch.load(check_point_path)
predictor1 = FLModel(
    models.resnet34(pretrained=True),
    layers_to_remove=1,
    num_features=128,
    num_materials=600,
    num_substances=5,
    input_size=4,
)
predictor1.load_state_dict(checkpoint['state_dict'], strict=True)
predictor1.train(False)
predictor1 = predictor1.cuda()

# predictor 2.
check_point_path = Path('./src/image_translation/pho_predictor/checkpoint/model-path2.best_pth.tar')
checkpoint = torch.load(check_point_path)
predictor2 = FLModel(
    models.resnet34(pretrained=True),
    layers_to_remove=1,
    num_features=128,
    num_materials=600,
    num_substances=5,
    input_size=4,
)
predictor2.load_state_dict(checkpoint['state_dict'], strict=True)
predictor2.train(False)
predictor2 = predictor2.cuda()


def bright_pixel_mask(image, percentile=80):
    image = rgb2gray(image)
    perc = np.percentile(np.unique(image), percentile)
    mask = image < perc
    return mask

def compute_top1(label_to_mat_id, pmodel:FLModel, image, seg_mask):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    seg_mask = skimage.transform.resize(seg_mask, (256, 256), order=0, anti_aliasing=False, mode='reflect')
    seg_mask = seg_mask[:, :, np.newaxis].astype(dtype=np.uint8) * 255
    image_tensor = pre_transforms.inference_image_transform(input_size=224, output_size=224, pad=0, to_pil=True)(image)
    mask_tensor = pre_transforms.inference_mask_transform(input_size=224, output_size=224, pad=0)(seg_mask)
    input_tensor = torch.cat((image_tensor, mask_tensor), dim=0).unsqueeze(0)
    output = pmodel.forward(input_tensor.cuda())
    topk_mat_scores, topk_mat_labels = torch.topk(F.softmax(output[0], dim=1), k=10)
    topk_dict = {
        'material': [
            {
                'score': score,
                'id': int(label_to_mat_id[int(label)]),
            } for score, label in zip(topk_mat_scores.squeeze().tolist(),
                                        topk_mat_labels.squeeze().tolist())
        ]
    }
    topk_subst_scores, topk_subst_labels = torch.topk(F.softmax(output[1].cpu(), dim=1),k=output[1].size(1))
    topk_dict['substance'] = \
        [
            {
                'score': score,
                'id': label,
                'name': SUBSTANCES[int(label)],
            } for score, label in zip(topk_subst_scores.squeeze().tolist(),
                                        topk_subst_labels.squeeze().tolist())
        ]
    return topk_dict


for o, data_i in enumerate(dataloader):
    print('{} / {}'.format(o, len(dataloader)))
    if o * opt.batchSize >= opt.how_many:
        break
    out = model(data_i, mode='inference')
    masks_num = data_i['label'].shape[0]
    imgs_num = data_i['label'].shape[0]
    
    if opt.save_per_img:
        warped_mask_root = save_root + '/transferred_mask/'
        img_root = save_root + '/transferred_img/'
        pre_root = save_root + '/final_prediction/'
        if not os.path.exists(warped_mask_root):
            os.makedirs(warped_mask_root)
        if not os.path.exists(img_root):
            os.makedirs(img_root)
        if not os.path.exists(pre_root):
            os.makedirs(pre_root)
        
        # get cocosnet predicted imgs and masks.
        ref_labels = data_i['label_ref'].cuda()
        gt_labels = data_i['label'].cuda()
        ref_pathes = data_i['ref_path']
        pathes = data_i['path']
        warped_masks_ori = out['warp_mask_to_ref']
        warped_masks = F.interpolate(warped_masks_ori, size=[500,500], mode='bilinear')
        warped_imgs = out['warp_out'] 
        fake_imgs = out['fake_image']
        fake_imgs = (fake_imgs + 1) / 2
        
        for j in range(masks_num):
            # set saving-filename.
            ori_path = pathes[j]
            ref_path = ref_pathes[j]         
            name_prefix = os.path.basename(data_i['path'][j])
            prefix = name_prefix[13:21] if 'train' in name_prefix else name_prefix[11:19]
            mask_name = os.path.basename(data_i['ref_path'][j])
            mask_name = mask_name.replace('.jpg','.png')
            mask_name = prefix + '_' + mask_name
            img_name = os.path.basename(data_i['path'][j])
            warp_name = os.path.basename(data_i['path'][j].replace('.jpg','_warped.jpg'))
            warped_mask_name = name_prefix.replace('.jpg','.png')

            # Get cocosnet predicted masks, saved as one pic, and vis this pic.
            ref_label = skimage.io.imread(ref_path.replace('jpg', 'png'))
            ori_warped_mask = warped_masks[j]    
            ori_warped_mask_0 = ori_warped_mask[1:]
            _, ori_warped_max = torch.max(ori_warped_mask_0, dim=0)
            ori_warped_max = ori_warped_max + 1
            fore_tensor = torch.from_numpy((ref_label > 0) * 1.0).cuda()
            zero = torch.zeros_like(ori_warped_max) 
            ori_warped_max_final = torch.where(fore_tensor > 0, ori_warped_max, zero)
            final_mask_np = ori_warped_max_final.squeeze().cpu().numpy().astype(np.uint8)
            final_mask_vis = np.full((500,500,3),255)
            for sem in range(1,58):
                final_mask_vis[final_mask_np == sem] = (COLOR_PLATTER[sem][2], COLOR_PLATTER[sem][1], COLOR_PLATTER[sem][0])
            cv2.imwrite(warped_mask_root + '/' + mask_name.replace('.png','_vis.png'), final_mask_vis)
            cv2.imwrite(warped_mask_root  + '/' + mask_name, final_mask_np)

            # Get cocosnet predicted img and warped img.
            vutils.save_image(fake_imgs[j:j+1], img_root  + '/' + img_name, nrow=1, padding=0, normalize=False)
            vutils.save_image(warped_imgs[j:j+1], img_root  + '/' + warp_name, nrow=1, padding=0, normalize=False)
            
            # Predict path1.
            result_dict_path1 = {}
            transferred_texture = imread(img_root + '/' + img_name)
            if len(transferred_texture.shape) > 2 and transferred_texture.shape[2] > 3:
                transferred_texture = transferred_texture[:,:,:3]
            transferred_texture = skimage.img_as_ubyte(transferred_texture)
            gt_mask = imread(data_i['path'][j].replace('jpg', 'png'))
            gt_mask = gt_mask.astype(np.int) - 1
            for seg_id in [s for s in np.unique(gt_mask) if s >= 0]:
                seg_mask = (gt_mask == seg_id)
                top1_dict = compute_top1(label_to_mat_id, predictor1, transferred_texture, seg_mask)
                result_dict_path1[str(seg_id)] = top1_dict
            
            result_dict_path1_res = {}
            for seg in list(result_dict_path1.keys()):
                result_dict_path1_res['material_' + seg] = int(result_dict_path1[seg]['material'][0]['id'])


            # Predict path2.
            # w/o superpix.
            result_dict_path2 = {}
            gt_texture = imread(ref_path)
            if len(gt_texture.shape) > 2 and gt_texture.shape[2] > 3:
                    gt_texture = gt_texture[:, :, :3]
            gt_texture = skimage.transform.resize(gt_texture, (256,256), anti_aliasing=True, order=3,mode='constant', cval=1)
            gt_texture = skimage.img_as_ubyte(gt_texture)
            transferred_segmentation = imread(warped_mask_root + '/' + mask_name)
            transferred_segmentation = transferred_segmentation.astype(np.int) - 1
            for seg_id in [s for s in np.unique(transferred_segmentation) if s >= 0]:
                seg_mask = (transferred_segmentation == seg_id)
                top1_dict = compute_top1(label_to_mat_id, predictor2, gt_texture, seg_mask)
                result_dict_path2[str(seg_id)] = top1_dict
            # w superpix.
            result_dict_path2_superpix = {} 
            superpix = imread(ref_path.replace('jpg','png'))
            for seg_id in [s for s in np.unique(transferred_segmentation) if s >= 0]:
                result_dict_path2_superpix[str(seg_id)] = {}
                superpixs = Counter(superpix[transferred_segmentation == seg_id])
                for sp in dict(superpixs):
                    if sp > 0:
                        seg_mask = (transferred_segmentation == seg_id) * 1 + (superpix == sp) * 1
                        seg_mask = (seg_mask == 2)
                        top1_dict = compute_top1(label_to_mat_id, predictor2, gt_texture, seg_mask)
                        result_dict_path2_superpix[str(seg_id)][str(sp)] = top1_dict
            result_dict_path2_res = {}
            result_dict_path2_sup_res = {}
            for seg in list(result_dict_path2):
                result_dict_path2_res['material_' + seg] = int(result_dict_path2[seg]['material'][0]['id'])
                result_dict_path2_sup_res['material_' + seg] = {}
                for sp in result_dict_path2_superpix[seg]:
                    result_dict_path2_sup_res['material_' + seg][sp] = int(result_dict_path2_superpix[seg][sp]['material'][0]['id'])

            # generate final results.
            # w/o superpix results.
            final_result = {}
            for seg_id in result_dict_path1:
                if seg_id in result_dict_path2:
                    final_result[seg_id] = result_dict_path2[seg_id]
                else:
                    final_result[seg_id] = result_dict_path1[seg_id]
            

            # w superpix results.
            final_result_superpix = {}
            for seg_id in result_dict_path1:
                if seg_id not in result_dict_path2:
                    final_result_superpix[seg_id] = result_dict_path1[seg_id]
                else:
                    path1_distance = id2matDis[int(result_dict_path1[seg_id]['material'][0]['id'])]
                    tem_dis = 999
                    for sp in result_dict_path2_superpix[seg_id]:
                        path2_mat = result_dict_path2_superpix[seg_id][sp]['material'][0]['id']
                        if path1_distance[path2_mat - 1] < tem_dis:
                            tem_sp = sp
                            tem_dis = path1_distance[path2_mat - 1]
                    final_result_superpix[seg_id] = result_dict_path2_superpix[seg_id][sp]

            final_result_res = {}
            final_result_superpix_res = {}
            for key in final_result:
                final_result_res['material_' + key] = int(final_result[key]['material'][0]['id'])
                final_result_superpix_res['material_' + key] = int(final_result_superpix[key]['material'][0]['id'])
            
            with open(ori_path.replace('jpg', 'json'), 'r') as f2:
                base_info = json.load(f2)
            
            with open(pre_root + img_name.replace('jpg', 'json'), 'w') as f1:
                json.dump(
                    {   
                        'base_info': base_info,
                        'path1_result': result_dict_path1_res,
                        'path2_result': result_dict_path2_res,
                        'final_result': final_result_res,
                        'final_result_sup': final_result_superpix_res,
                    }, f1, indent=2
                )
    else:
        if not os.path.exists(save_root + '/test/'):
            os.makedirs(save_root + '/test/' )

        if opt.dataset_mode == 'deepfashion':
            label = data_i['label'][:,:3,:,:]
        elif opt.dataset_mode == 'celebahqedge':
            label = data_i['label'].expand(-1, 3, -1, -1).float()
        else:
            label = masktorgb(data_i['label'].cpu().numpy())
            label = torch.from_numpy(label).float() / 128 - 1

        imgs = torch.cat((label.cpu(), data_i['ref'].cpu(), out['fake_image'].data.cpu()), 0)
        try:
            imgs = (imgs + 1) / 2
            vutils.save_image(imgs, save_root + '/test/' + '/' + str(o) + '.png',  
                    nrow=imgs_num, padding=0, normalize=False)
        except OSError as err:
            print(err)