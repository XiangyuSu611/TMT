'''
    train material predictor, 
    by default, first-stage triplet learning is performed
    when execute --classification, classification learning will be performed 
    we recommend to perform triple training first, and continue the training of classification learning through the --resume option
'''


import os
import argparse
import time
import shutil
import random
import sys
sys.path.append("./")
sys.path.append("./src")

import torch
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import pylab as pl
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as transforms_F

from src.material_prediction.config import SUBSTANCES
from src.material_prediction.data import transforms, rendering_dataset
from src.material_prediction.networks import utils as simi_utils
from src.material_prediction.networks.losses import TripletLossHuman
from src.material_prediction.networks.network import FLModel


INPUT_SIZE = 224
SHAPE = (384, 384)

parser = argparse.ArgumentParser(description='Material Similarity Training')
parser.add_argument('--snapshot-dir',
                    metavar='DIR', help='path to dataset',
                    default='data/training_data/material_prediction/')
parser.add_argument('--similarity-matrix-path',
                    metavar='DIR', help='path to similarity matrix',
                    default='data/training_data/material_prediction/total_similarity_matrix_sqrt.csv')
parser.add_argument('-j', '--workers',
                    default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 12)')
parser.add_argument('--epochs',
                    default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--tri-adjust-epoch',
                    nargs='+', default=[10, 20, 30, 40],
                    type=int, help='milestones to adjust the learning rate in triplet learning')
parser.add_argument('--classi-adjust-epoch',
                    nargs='+', default=[100, 180],
                    type=int, help='milestones to adjust the learning rate in classification learning')
parser.add_argument('--num-classes', default=100, type=int,
                    help='number of classes in the problem')
parser.add_argument('--emb-size',
                    default=128, type=int, help='size of the embedding')
parser.add_argument('--input-channels',
                    default=4, type=int, help='size of the input')
parser.add_argument('-b', '--batch-size',
                    default=150, type=int,
                    metavar='N', help='mini-batch size (default: 120)')
parser.add_argument('--triplet-learning-rate',
                    default=1e-3, type=float,
                    help='initial triplet learning rate')
parser.add_argument('--classifi-learning-rate',
                    default=5e-4, type=float,
                    help='initial classification learning rate')
parser.add_argument('--wd', '--weight-decay',
                    default=4e-4, type=float,
                    metavar='W', help='weight decay (default: 4e-4)',
                    dest='weight_decay')
parser.add_argument('--betas',
                    nargs='+', default=[0.9, 0.999], type=float,
                    help='beta values for ADAM')
parser.add_argument('--momentum',
                    default=0.9, type=float,
                    help='momentum in the SGD')
parser.add_argument('--margin',
                    default=0.3, type=float,
                    help='triplet loss margin')
parser.add_argument('--checkpoint-folder',
                    default='src/material_prediction/checkpoint/',
                    type=str, help='folder to store the trained models')
parser.add_argument('--model-name',
                    default='resnet_similarity', type=str,
                    help='name given to the model')
parser.add_argument('--resume',
                    default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--classification',
                    default=False, action='store_true',
                    help='choose whether to start classification learning')   
parser.add_argument('--dis-loss',
                    default=True, action='store_true',
                    help='Whether to use material similaritu distance loss')                  
parser.add_argument('-e', '--evaluate',
                    dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed',
                    default=2851, type=int,
                    help='seed for initializing training.')
parser.add_argument('--weight_tri',
                    default=1.0, type=float,
                    help='weight of tri loss.')
parser.add_argument('--weight_simi',
                    default=1.0, type=float,
                    help='weight of simi loss.')
parser.add_argument('--weight_mat',
                    default=1.0, type=float,
                    help='weight of material loss.')
parser.add_argument('--weight_sub',
                    default=0.5, type=float,
                    help='weight of substance loss.')
parser.add_argument('--weight_dis',
                    default=1.0, type=float,
                    help='weight of distance loss.')
parser.add_argument('--mask_noise_p',
                    default=0.0, type=float,
                    help='noise for masks.')


class AverageMeter(object):
    """
    https://github.com/pytorch/examples/blob/master/imagenet/single.py
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

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


class RandomResize(object):
    def __init__(self, low, high, interpolation=Image.BILINEAR):
        self.low = low
        self.high = high
        self.interpolation = interpolation

    def __call__(self, img):
        size = np.random.randint(self.low, self.high)
        return transforms_F.resize(img, size, self.interpolation)


def train_model(loader, epoch, similarity_matrix):
    def update_progress_bar(progress_bar, losses):
        description = '[Epoch ' + str(epoch) + '-train]'
        if args.classification == False:
            description += ' Triplet loss: '
            description += '%.3f/ %.3f (AVG)' % (losses.val, losses.avg)
        else:
            description += ' Classi loss: '
            description += '%.3f/ %.3f (AVG)' % (losses.val, losses.avg)
        progress_bar.set_description(description)

    global model
    global criterion
    global optimizer

    # keep track of the loss value
    losses = AverageMeter()
    losses_human = AverageMeter()
    losses_preplexity = AverageMeter()
    losses_material = AverageMeter()
    losses_substance = AverageMeter()
    losses_distance = AverageMeter()

    progress_bar = tqdm(loader, total=len(loader))

    with torch.set_grad_enabled(True):
        for batch_idx, batch_dict in enumerate(progress_bar):       
            imgs = batch_dict['image'].to(device, dtype)
            targets_mat = batch_dict['material_label'].to(device)
            targets_sub = batch_dict['substance_label'].to(device)

            # forward through the model and compute error
            pred_mat, pred_sub, embeddings = model(imgs)
            loss, loss_human, loss_perplexity, loss_material, loss_substance, loss_distance = \
                criterion(pred_mat, pred_sub, embeddings, targets_mat, targets_sub)

            losses.update(loss.item(), imgs.size(0))
            
            if args.classification == True:
                # not compute loss_human and loss_perplexity
                losses_human.update(loss_human, imgs.size(0))
                losses_preplexity.update(loss_perplexity, imgs.size(0))
                losses_material.update(loss_material.item(), imgs.size(0))
                losses_substance.update(loss_substance.item(), imgs.size(0))
                if args.dis_loss == True:
                    losses_distance.update(loss_distance.item(), imgs.size(0))
                else:
                    losses_distance.update(loss_distance, imgs.size(0))
            else:
                # not compute loss_material and loss_substance
                losses_human.update(loss_human.item(), imgs.size(0))
                losses_preplexity.update(loss_perplexity.item(), imgs.size(0))
                losses_material.update(loss_material, imgs.size(0))
                losses_substance.update(loss_substance, imgs.size(0))
                losses_distance.update(loss_distance, imgs.size(0))

            # compute gradient and update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_progress_bar(progress_bar, losses)

    return losses.avg, losses_human.avg, losses_preplexity.avg, losses_material.avg, losses_substance.avg, losses_distance.avg


def evaluate_model(loader, epoch, similarity_matrix):
    
    def update_progress_bar(progress_bar, current_agr, avg_agr, mat_prec1):
        description = '[Epoch ' + str(epoch) + '-val]'
        description += ' agreement: ' + '%.3f/ %.3f (AVG)' % (current_agr, avg_agr)
        if args.classification == True:
            description += '| mat_prec@1: ' + '%.2f/ %.2f (AVG)' %(mat_prec1.val, mat_prec1.avg)
        progress_bar.set_description(description)
    
    global model
    global criterion
    global optimizer
    
    correct = AverageMeter()
    total = AverageMeter()
    mat_prec1 = AverageMeter()
    mat_prec5 = AverageMeter()
    sub_prec = AverageMeter()
    mat_dist = AverageMeter()

    # keep track of the loss value
    losses = AverageMeter()
    losses_human = AverageMeter()
    losses_preplexity = AverageMeter()
    losses_material = AverageMeter()
    losses_substance = AverageMeter()
    losses_distance = AverageMeter()

    progress_bar = tqdm(loader, total=len(loader))

    with torch.set_grad_enabled(False):
        for batch_idx, batch_dict in enumerate(progress_bar):
            imgs = batch_dict['image'].to(device)
            targets_mat = batch_dict['material_label'].to(device)
            targets_sub = batch_dict['substance_label'].to(device)

            # forward through the model and compute accuracy
            pred_mat, pred_sub, embeddings = model(imgs)
            if batch_idx == 86:
                print('aaa')
            batch_correct_number, batch_total_number = criterion.get_simi_accuracy(embeddings, targets_mat)
            correct.update(batch_correct_number)
            total.update(batch_total_number)
            current_agr = batch_correct_number / ( batch_total_number + 1e-8) 
            avg_agr = correct.sum / total.sum

            # compute val loss
            loss, loss_human, loss_perplexity, loss_material, loss_substance, loss_distance = \
                criterion(pred_mat, pred_sub, embeddings, targets_mat, targets_sub)

            losses.update(loss.item(), imgs.size(0))

            if args.classification == True:
                # not compute loss_human and loss_perplexity
                losses_human.update(loss_human, imgs.size(0))
                losses_preplexity.update(loss_perplexity, imgs.size(0))
                losses_material.update(loss_material.item(), imgs.size(0))
                losses_substance.update(loss_substance.item(), imgs.size(0))
                if args.dis_loss == True:
                    losses_distance.update(loss_distance.item(), imgs.size(0))
                else:
                    losses_distance.update(loss_distance, imgs.size(0))
            else:
                # not compute loss_material and loss_substance
                losses_human.update(loss_human.item(), imgs.size(0))
                losses_preplexity.update(loss_perplexity.item(), imgs.size(0))
                losses_material.update(loss_material, imgs.size(0))
                losses_substance.update(loss_substance, imgs.size(0))
                losses_distance.update(loss_distance, imgs.size(0))

            # if args.classification == True, start to compute prediction accuracy
            if args.classification == True:
                pred_mat_acc1, pred_mat_acc5, pred_sub_acc, pred_mat_dist \
                    = criterion.get_pred_accuracy(pred_mat, pred_sub, targets_mat, targets_sub, similarity_matrix)
                mat_prec1.update(pred_mat_acc1, imgs.size(0))
                mat_prec5.update(pred_mat_acc5, imgs.size(0))
                sub_prec.update(pred_sub_acc, imgs.size(0))
                mat_dist.update(pred_mat_dist, imgs.size(0))

            update_progress_bar(progress_bar, current_agr, avg_agr, mat_prec1)

    return avg_agr, mat_prec1.avg, mat_prec5.avg, sub_prec.avg, mat_dist.avg, \
            losses.avg, losses_human.avg, losses_preplexity.avg, losses_material.avg, losses_substance.avg, losses_distance.avg


def save_checkpoint(state, is_best, folder, model_name='checkpoint', ):
    """
    if the current state is the best it saves the pytorch model
    in folder with name filename
    """
    path = os.path.join(folder, model_name)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, 'model')

    torch.save(state, path + '.pth.tar')
    if is_best:
        shutil.copyfile(path + '.pth.tar', path + '_best.pth.tar')


if __name__ == '__main__':

    # get input arguments
    args = parser.parse_args()

    # set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # set device and dtype
    dtype = torch.float
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        # # comment this if we want reproducibility
        # torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.enabled = True

        # this might affect performance but allows reproducibility
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True

    # define dataset
    # use photoshape dataloder.
    print(' * Loading datasets')
    snapshot_dir = Path(args.snapshot_dir)
    train_path = Path(snapshot_dir, 'training')
    validation_path = Path(snapshot_dir, 'validation')
    meta_path = Path(snapshot_dir, 'meta.json')
    color_binner = None

    with meta_path.open('r') as f:
        meta_dict = json.load(f)

    mat_id_to_label = {int(k): v
        for k, v in meta_dict['mat_id_to_label'].items()}

    num_materials = max(mat_id_to_label.values()) + 1 
    num_substances = len(SUBSTANCES)
    
    train_dataset = rendering_dataset.MaterialRendDataset(
        train_path,
        meta_dict,
        color_binner=color_binner,
        shape=SHAPE,
        lmdb_name=snapshot_dir.name,
        image_transform=transforms.train_image_transform(INPUT_SIZE, pad=0),
        mask_transform=transforms.train_mask_transform(INPUT_SIZE, pad=0),
        mask_noise_p=args.mask_noise_p)

    validation_dataset = rendering_dataset.MaterialRendDataset(
        validation_path,
        meta_dict,
        color_binner=color_binner,
        shape=SHAPE,
        lmdb_name=snapshot_dir.name,
        image_transform=transforms.inference_image_transform(
            INPUT_SIZE, INPUT_SIZE),
        mask_transform=transforms.inference_mask_transform(
            INPUT_SIZE, INPUT_SIZE))

    loader_train = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        pin_memory=True,
    )

    loader_val = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=False,
    )

    # load similarity matrix.
    mat_dis_path = args.similarity_matrix_path
    mat_dis = pd.read_csv(mat_dis_path, header=None)
    label2matDis = {}
    label_num = 0
    for i in range(1, mat_dis.values.shape[0]):
        list_i = mat_dis.values[i]
        label2matDis[label_num] = torch.from_numpy(list_i[1:]).cuda()
        label_num += 1
    
    # create model
    model = FLModel(
        models.resnet34(pretrained=True),
        layers_to_remove=1,
        num_features=args.emb_size,
        num_materials=num_materials,
        num_substances=num_substances,
        input_size=args.input_channels,
    )
    model = model.to(device)

    # define loss function
    criterion = TripletLossHuman(
        margin=args.margin,
        unit_norm=True,
        device=device,
        seed=args.seed,
        weight_human=args.weight_tri,
        weight_per=args.weight_simi,
        weight_mat=args.weight_mat,
        weight_sub=args.weight_sub,
        weight_dis=args.weight_dis,
        is_classification=args.classification,
        ues_distance_loss=args.dis_loss,
    )

    # define optimizer
    if args.classification == True:
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.classifi_learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            weight_decay=args.weight_decay,
            lr=args.triplet_learning_rate,
            momentum=args.momentum,
            nesterov=True,
        )

    # define LR scheduler
    if args.classification == True:
        adjust_epoch = args.classi_adjust_epoch
        gamma = 0.2
    else:
        adjust_epoch = args.tri_adjust_epoch
        gamma = 0.2
    
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=adjust_epoch,
        gamma=gamma,
    )
    lr_scheduler.last_epoch = args.start_epoch

    # define training status
    train_total_loss = []
    train_triplet_loss = []
    train_similarity_loss = []
    train_mat_loss = []
    train_sub_loss = []
    train_dis_loss = []

    val_total_loss = []
    val_triplet_loss = []
    val_similarity_loss = []
    val_mat_loss = []
    val_sub_loss = []
    val_dis_loss = []

    agr = []
    mat_acc1 = []
    mat_acc5 = []
    sub_acc = []
    mat_dist = []

    best_agreement = 0
    best_mat_acc = 0

    if args.resume:
        print(f' * Loading weights from {args.resume!s}')
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        with open(Path(args.resume).parent.parent / 'train_status.json', 'r') as f:
            train_status = json.load(f)
            current_epoch = train_status['epochs']
            train_total_loss = eval(train_status['train_total_loss'])
            val_total_loss = eval(train_status['val_total_loss'])
            train_triplet_loss = eval(train_status['train_triplet_loss'])
            train_similarity_loss = eval(train_status['train_simi_loss'])
            
            agr = eval(train_status['agr'])
            best_agreement = float(train_status['best_agr'])
            best_mat_acc = float(train_status['best_mat_acc@1'])

    if args.evaluate:
        # evaluation step
        model = model.eval()
        evaluate_model()

    else:
        # start training and evaluation loop
        for epoch in range(args.start_epoch + 1, args.epochs + 1):
            # train step
            model = model.train()
            current_train_loss, current_train_tri_loss, current_train_simi_loss, current_train_mat_loss, current_train_sub_loss, current_train_dis_loss = \
                train_model(loader_train, epoch, label2matDis)
            
            train_total_loss.append(current_train_loss)
            train_triplet_loss.append(current_train_tri_loss)
            train_similarity_loss.append(current_train_simi_loss)
            train_mat_loss.append(current_train_mat_loss)
            train_sub_loss.append(current_train_sub_loss)
            train_dis_loss.append(current_train_dis_loss)

            lr_scheduler.step()

            # evaluation step
            model = model.eval()
            current_agr, current_mat_acc1, current_mat_acc5, current_sub_acc, current_mat_dist, \
                current_val_loss, current_val_tri_loss, current_val_simi_loss, current_val_mat_loss, current_val_sub_loss, current_val_dis_loss = \
                    evaluate_model(loader_val, epoch, label2matDis)

            agr.append(current_agr)
            mat_acc1.append(current_mat_acc1)
            mat_acc5.append(current_mat_acc5)
            sub_acc.append(current_sub_acc)
            mat_dist.append(current_mat_dist)

            val_total_loss.append(current_val_loss)
            val_triplet_loss.append(current_val_tri_loss)
            val_similarity_loss.append(current_val_simi_loss)
            val_mat_loss.append(current_val_mat_loss)
            val_sub_loss.append(current_val_sub_loss)
            val_dis_loss.append(current_val_dis_loss)

            # save checkpoint model if it is the best
            if args.classification == True:
                is_best = current_mat_acc1 > best_mat_acc
                best_mat_acc = max(current_mat_acc1, best_mat_acc)
            else:
                is_best = current_agr > best_agreement
                best_agreement= max(current_agr, best_agreement)

            saved = False
            
            if is_best and saved == False:
                save_checkpoint(
                    {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'best_agreement': best_agreement,
                        'classification': args.classification,
                        'best_mat_acc': best_mat_acc,
                        'optimizer': optimizer.state_dict(),
                        'lr_schedule': lr_scheduler.state_dict(),
                    },
                    is_best, folder=args.checkpoint_folder,
                    model_name=args.model_name + '-' + str(epoch) + '_best'
                )

                model_name = args.model_name + '-' + str(epoch) + '_best'

                with (Path(args.checkpoint_folder) / model_name / 'best_train_status.json').open('w') as f:
                    json.dump({
                        'snapshot_dir': args.snapshot_dir,
                        'similarity_matrix_path': args.similarity_matrix_path,
                        'epochs':args.epochs,
                        'epoch_now': epoch,
                        'start_epoch':args.start_epoch,
                        'number_classes': args.num_classes,
                        'emb_size': args.emb_size,
                        'batch_size': args.batch_size,
                        'init_tri_learning_rate': args.triplet_learning_rate,
                        'init_cls_learning_rate': args.classifi_learning_rate,
                        'classification': args.classification,
                        'ues_distance_loss': args.dis_loss,

                        'train_total_loss': str(current_train_loss),
                        'train_triplet_loss': str(current_train_tri_loss),
                        'train_simi_loss': str(current_train_simi_loss),
                        'train_sub_loss': str(current_train_sub_loss),
                        'train_mat_loss': str(current_train_mat_loss),
                        'train_dis_loss': str(current_train_dis_loss),

                        'val_total_loss': str(current_val_loss),
                        'val_triplet_loss': str(current_val_tri_loss),
                        'val_simi_loss': str(current_val_simi_loss),
                        'val_sub_loss': str(current_val_mat_loss),
                        'val_mat_loss': str(current_val_sub_loss),
                        'val_dis_loss': str(current_val_dis_loss),

                        'agr': str(current_agr),
                        'mat_acc@1': str(current_mat_acc1),
                        'mat_acc@5': str(current_mat_acc5),
                        'sub_acc': str(current_sub_acc),
                        'mat_dis': str(current_mat_dist),

                    }, f, indent=4)

                saved = True 
            
            if saved == False:
                save_checkpoint(
                    {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'best_agreement': best_agreement,
                        'classification': args.classification,
                        'best_mat_acc': best_mat_acc,
                        'optimizer': optimizer.state_dict(),
                        'lr_schedule': lr_scheduler.state_dict(),
                    },
                    is_best, folder=args.checkpoint_folder,
                    model_name=args.model_name + '-' + str(epoch)
                )
                saved = True

            # plot graph.
            x = range(len(agr))

            pl.clf()
            pl.plot(x, agr, 'r', label=U'agr')
            pl.xlabel(U'epoch')
            pl.ylabel(U'value')
            pl.legend()
            pl.savefig(args.checkpoint_folder + '/agr.png')
            
            if args.classification == False:
                pl.plot(x, train_total_loss, 'b', label=U'train_total_loss')
                pl.plot(x, val_total_loss, 'g', label=U'val_total_loss')
                pl.xlabel(U'epoch')
                pl.ylabel(U'value')
                pl.legend()
                pl.savefig(args.checkpoint_folder + '/total_loss.png')

                pl.clf()
                pl.plot(x, train_triplet_loss, 'b', label=U'train_triplet_loss')
                pl.plot(x, val_triplet_loss, 'g', label=U'val_triplet_loss')
                pl.xlabel(U'epoch')
                pl.ylabel(U'value')
                pl.legend()
                pl.savefig(args.checkpoint_folder + '/triplet_loss.png')

                pl.clf()
                pl.plot(x, train_similarity_loss, 'b', label=U'train_similarity_loss')
                pl.plot(x, val_similarity_loss, 'g', label=U'val_similarity_loss')
                pl.xlabel(U'epoch')
                pl.ylabel(U'value')
                pl.legend()
                pl.savefig(args.checkpoint_folder + '/similarity_loss.png')

            if args.classification == True:
                pl.clf()
                pl.plot(x[args.start_epoch:], train_total_loss[args.start_epoch:], 'b', label=U'train_total_loss')
                pl.plot(x[args.start_epoch:], val_total_loss[args.start_epoch:], 'g', label=U'val_total_loss')
                pl.xlabel(U'epoch')
                pl.ylabel(U'value')
                pl.legend()
                pl.savefig(args.checkpoint_folder + '/total_loss.png')

                pl.clf()
                pl.plot(x[args.start_epoch:], train_mat_loss, 'b', label=U'train_material_loss')
                pl.plot(x[args.start_epoch:], val_mat_loss, 'g', label=U'val_material_loss')
                pl.xlabel(U'epoch')
                pl.ylabel(U'value')
                pl.legend()
                pl.savefig(args.checkpoint_folder + '/material_loss.png')

                pl.clf()
                pl.plot(x[args.start_epoch:], train_sub_loss, 'b', label=U'train_substance_loss')
                pl.plot(x[args.start_epoch:], val_sub_loss, 'g', label=U'val_substance_loss')
                pl.xlabel(U'epoch')
                pl.ylabel(U'value')
                pl.legend()
                pl.savefig(args.checkpoint_folder + '/substance_loss.png')

                if args.dis_loss == True:
                    pl.clf()
                    pl.plot(x[args.start_epoch:], train_dis_loss, 'b', label=U'train_distance_loss')
                    pl.plot(x[args.start_epoch:], val_dis_loss, 'g', label=U'val_distance_loss')
                    pl.xlabel(U'epoch')
                    pl.ylabel(U'value')
                    pl.legend()
                    pl.savefig(args.checkpoint_folder + '/distance_loss.png')

                pl.clf()
                pl.plot(x[args.start_epoch:], mat_acc1, 'y', label=U'mat@1_acc')
                pl.xlabel(U'epoch')
                pl.ylabel(U'value')
                pl.legend()
                pl.savefig(args.checkpoint_folder + '/mat@1_acc.png')

                pl.clf()
                pl.plot(x[args.start_epoch:], mat_acc5, 'y', label=U'mat@5_acc')
                pl.xlabel(U'epoch')
                pl.ylabel(U'value')
                pl.legend()
                pl.savefig(args.checkpoint_folder + '/mat@5_acc.png')

                pl.clf()
                pl.plot(x[args.start_epoch:], sub_acc, 'b', label=U'sub@1_acc')
                pl.xlabel(U'epoch')
                pl.ylabel(U'value')
                pl.legend()
                pl.savefig(args.checkpoint_folder + '/sub@1_acc.png')

                pl.clf()
                pl.plot(x[args.start_epoch:], mat_dist, 'b', label=U'mat@1_dist')
                pl.xlabel(U'epoch')
                pl.ylabel(U'value')
                pl.legend()
                pl.savefig(args.checkpoint_folder + '/mat@1_dist.png')

                pl.clf()
                pl.plot(x[args.start_epoch:], sub_acc, 'r', label=U'sub_acc')
                pl.plot(x[args.start_epoch:], mat_acc1, 'g', label=U'mat_acc@1')
                pl.plot(x[args.start_epoch:], mat_acc5, 'b', label=U'mat_acc@5')
                pl.xlabel(U'epoch')
                pl.ylabel(U'value')
                pl.legend()
                pl.savefig(args.checkpoint_folder + '/mat-and-sub-acc.png')
        
            # save json.
            with (Path(args.checkpoint_folder) / 'train_status.json').open('w') as f:
                json.dump({
                    'snapshot_dir': args.snapshot_dir,
                    'similarity_matrix_path': args.similarity_matrix_path,
                    'epochs':args.epochs,
                    'start_epoch':args.start_epoch,
                    'tri_adjust_epoch': args.tri_adjust_epoch,
                    'emb_size': args.emb_size,
                    'batch_size': args.batch_size,
                    'init_tri_learning_rate': args.triplet_learning_rate,
                    'init_cls_learning_rate': args.classifi_learning_rate,
                    'weight_decay': args.weight_decay,
                    'betas':args.betas,
                    'momentum':args.momentum,
                    'margin':args.margin,
                    'checkpoint_dir': args.checkpoint_folder,
                    'model_name': args.model_name,
                    'weight_tri': args.weight_tri,
                    'weight_simi': args.weight_simi,
                    'weight_mat': args.weight_mat,
                    'weight_sub': args.weight_sub,
                    'weight_dis': args.weight_dis,
                    'epoch_now': epoch,
                    'classification': args.classification,

                    'train_total_loss': str(train_total_loss),
                    'train_triplet_loss': str(train_triplet_loss),
                    'train_simi_loss': str(train_similarity_loss),
                    'train_sub_loss': str(train_sub_loss),
                    'train_mat_loss': str(train_mat_loss),
                    'train_dis_loss': str(train_dis_loss),

                    'val_total_loss': str(val_total_loss),
                    'val_triplet_loss': str(val_triplet_loss),
                    'val_simi_loss': str(val_similarity_loss),
                    'val_sub_loss': str(val_sub_loss),
                    'val_mat_loss': str(val_mat_loss),
                    'val_dis_loss': str(val_dis_loss),

                    'agr': str(agr),
                    'best_agr': str(best_agreement),
                    'mat_acc@1': str(mat_acc1),
                    'mat_acc@5': str(mat_acc5),
                    'sub_acc': str(sub_acc),
                    'mat_dis': str(mat_dist),
                    'best_mat_acc@1': str(best_mat_acc)

                }, f, indent=4) 