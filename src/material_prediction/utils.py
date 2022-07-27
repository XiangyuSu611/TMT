import operator
import random
from functools import reduce

import torch
from jinja2 import Template
from torch.nn import functional as F

import numpy as np

from src.material_prediction.config import SUBSTANCES
from src.material_prediction.data.transforms import denormalize_transform
from src.thirdparty.toolbox.toolbox.colors import visualize_lab_color_hist, lab_rgb_gamut_bin_mask


class MeterTable(object):
    _tmpl = Template('''
    <h5>{{title}}</h5>
    <table class="table table-small table-bordered" 
           style="font-family: monaco; font-size: 10px !important;">
        <thead>
            <tr>
                <th>Name</th>a
                <th>Current</th>
                <th>Mean</th>
                <th>Std</th>
            </tr>
        </thead>
        <tbody>
            {% for name, meter in meters %}
                <tr>
                    <td>{{name}}</td>
                    <td>{{meter.val}}</td>
                    <td>{{meter.mean}}</td>
                    <td>{{meter.std}}</td>
                </tr>
            {% endfor %}
        </tbody>
    </table>
    ''')
    def __init__(self, title='', meters=None):
        self.title = title
        if meters is None:
            self.meters = []
        else:
            self.meters = meters

    def add(self, meter, name):
        self.meters.append((name, meter))

    def render(self):
        return self._tmpl.render(title=self.title,
                                 meters=self.meters)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_precision(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    if len(res) > 1:
        return res
    else:
        return res[0]

def compute_mat_precision_finetune(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
    res = []
    for k in topk:
        rig = 0
        if k == 1:
            for j in range(batch_size):
                mat_gt = target[j][:,1]
                mat_pre = pred[j][0]
                if mat_pre in mat_gt:
                    rig += 1
            res.append(rig / batch_size * 100)
        elif k > 1:
            for j in range(batch_size):
                mat_gt = target[j][:,1]
                for i in range(k):
                    if pred[j][i] in mat_gt:
                        rig += 1
                        break
            res.append(rig / batch_size * 100)   
    if len(res) > 1:
        return res
    else:
        return res[0]

def compute_sub_precision_finetune(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)

    res = []
    for k in topk:
        rig = 0
        if k == 1:
            for j in range(batch_size):
                sub_gt = target[j][:,2]
                if pred[j][0] in sub_gt:
                    rig += 1
            res.append(rig / batch_size * 100)
        elif k > 1:
            for j in range(batch_size):
                sub_gt = target[j][:,2]
                for i in range(k):
                    if pred[j][i] in sub_gt:
                        rig += 1
                        break
            res.append(rig / batch_size * 100)
    
    if len(res) > 1:
        return res
    else:
        return res[0]

def compute_distance(output, target, mat_dis, label2mat_id, topk=(1,)):
    """Computes the distance between predict material(s) and GT material."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    
    res = []
    for k in topk:
        dis_k = 0
        for j in range(batch_size):
            pred_one = pred[:,j:j+1].squeeze().cpu().numpy()
            gt_mat = int(label2mat_id[target[j].item()])
            gt_weight = mat_dis[gt_mat].cpu().numpy()
            if k == 1:
                pre_mat = int(label2mat_id[pred_one[0]])
                dis_k += gt_weight[pre_mat - 1]
            elif k > 1:
                for i in range(k):
                    pre_mat = int(label2mat_id[pred_one[i]])
                    if i == 0:
                        distance = gt_weight[pre_mat - 1]
                        min_dis = distance
                    else:
                        distance = gt_weight[pre_mat - 1]
                        min_dis = min(min_dis, distance)
                dis_k += min_dis    
        if k == 1:
            avg_dis_k = dis_k / batch_size
            res.append(avg_dis_k)
        else:
            avg_dis_k = dis_k / batch_size
            res.append(avg_dis_k)        
    if len(res) > 1:
        return res
    else:
        return res[0]


def compute_finetune_distance(output, target, mat_dis, topk=(1,)):
    """Computes the distance between predict material(s) and GT material."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)

    res = []
    dis = 0.

    for j in range(batch_size):
        min_dis = torch.tensor(999.).cuda()
        gt = target[j]
        pre_mat = pred[j][0]
        gt_sum = gt.sum(dim=1)
        have_idx = gt_sum.nonzero().squeeze()
        for part_idx in have_idx:
            gt_mat = gt[part_idx][1]
            gt_dis = mat_dis[gt_mat.item()].float()
            current_dis = gt_dis[pre_mat]
            min_dis = min(current_dis, min_dis)
        dis += min_dis

    avg_dis_k = dis / batch_size
    res.append(avg_dis_k.item())
      
    return res[0]


def material_to_substance_output(mat_output, subst_mat_labels):
    mat_output = F.softmax(mat_output, dim=1)
    subst_output = torch.zeros(mat_output.size(0), len(SUBSTANCES)).cuda()
    for subst_label, mat_labels in subst_mat_labels.items():
        subst_output[:, subst_label] = mat_output[:, mat_labels].sum(dim=1)
    return torch.log(subst_output)


def decay_learning_rate(optimizer, epoch, init_lr, decay_epochs, decay_frac):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (decay_frac ** (epoch // decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def visualize_input(batch_dict, type='side_by_side'):
    rand_idx = random.randrange(0, batch_dict['image'].size(0))
    # WARNING: transforms do operations in-place, we MUST clone here.
    rand_im = denormalize_transform(batch_dict['image'][rand_idx].clone())

    if type == 'side_by_side':
        return torch.cat(
            (rand_im[:3],
             rand_im[3].unsqueeze(0).expand(*rand_im[:3].size())), dim=2)
    elif type == 'overlay':
        mask_vis = torch.cat((
            rand_im[3].unsqueeze(0),
            torch.zeros(1, *rand_im[3].size()),
            torch.zeros(1, *rand_im[3].size()),
        ), dim=0)
        return rand_im[:3]/2 + mask_vis/2

def visualize_input_path1(batch_dict, type='side_by_side'):
    rand_idx = random.randrange(0, batch_dict['image_path1'].size(0))
    # WARNING: transforms do operations in-place, we MUST clone here.
    rand_im = denormalize_transform(batch_dict['image_path1'][rand_idx].clone())

    if type == 'side_by_side':
        return torch.cat(
            (rand_im[:3],
             rand_im[3].unsqueeze(0).expand(*rand_im[:3].size())), dim=2)
    elif type == 'overlay':
        mask_vis = torch.cat((
            rand_im[3].unsqueeze(0),
            torch.zeros(1, *rand_im[3].size()),
            torch.zeros(1, *rand_im[3].size()),
        ), dim=0)
        return rand_im[:3]/2 + mask_vis/2

def visualize_input_path2(batch_dict, type='side_by_side'):
    rand_idx = random.randrange(0, batch_dict['image_path2'].size(0))
    # WARNING: transforms do operations in-place, we MUST clone here.
    rand_im = denormalize_transform(batch_dict['image_path2'][rand_idx].clone())

    if type == 'side_by_side':
        return torch.cat(
            (rand_im[:3],
             rand_im[3].unsqueeze(0).expand(*rand_im[:3].size())), dim=2)
    elif type == 'overlay':
        mask_vis = torch.cat((
            rand_im[3].unsqueeze(0),
            torch.zeros(1, *rand_im[3].size()),
            torch.zeros(1, *rand_im[3].size()),
        ), dim=0)
        return rand_im[:3]/2 + mask_vis/2