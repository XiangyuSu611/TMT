# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import torch
import torchvision.utils as vutils
import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util import util
from util.util import print_current_errors
from trainers.pix2pix_trainer import Pix2PixTrainer
import torch.nn.functional as F

opt = TrainOptions().parse()
print(' '.join(sys.argv))
dataloader = data.create_dataloader(opt)
len_dataloader = len(dataloader)
dataloader.dataset[11]
iter_counter = IterationCounter(opt, len(dataloader))
trainer = Pix2PixTrainer(opt, resume_epoch=iter_counter.first_epoch)
save_root = os.path.join(os.path.dirname(opt.checkpoints_dir), 'output', opt.name)

for epoch in iter_counter.training_epochs():
    opt.epoch = epoch
    # add noise.
    if not opt.maskmix:
        print('inject nothing')
    elif opt.maskmix and opt.noise_for_mask and epoch > opt.mask_epoch:
        print('inject noise')
    else:
        print('inject mask')
    print('real_reference_probability is :{}'.format(dataloader.dataset.real_reference_probability))
    print('hard_reference_probability is :{}'.format(dataloader.dataset.hard_reference_probability))
    
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        # ceshi daima
        if iter_counter.epoch_iter == 228:
            a = 1
        iter_counter.record_one_iteration()
        p = min(float(i + (epoch - 1) * len_dataloader) / 50 / len_dataloader, 1)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        # Training
        # train generator (every D_per_G epoch for generator)
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data=data_i, alpha=alpha)
        # train discriminator
        trainer.run_discriminator_one_step(data_i)
        if iter_counter.needs_printing():
            # return G loss and D loss.
            losses = trainer.get_latest_losses()
            try:
                print_current_errors(opt, epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter)
            except OSError as err:
                print(err)
        if iter_counter.needs_displaying():
            if not os.path.exists(save_root + opt.name):
                os.makedirs(save_root + opt.name)
            imgs_num = data_i['label'].shape[0]
            # print(data_i['path'])
            # print(data_i['ref_path'])
            # input label.
            if opt.dataset_mode == 'celebahq':
                data_i['label'] = data_i['label'][:,::2,:,:]
            elif opt.dataset_mode == 'celebahqedge':
                data_i['label'] = data_i['label'][:,:1,:,:]
            elif opt.dataset_mode == 'Renderedge':
                data_i['label'] = data_i['label'][:,:1,:,:]
            elif opt.dataset_mode == 'deepfashion':
                data_i['label'] = data_i['label'][:,:3,:,:]
            if data_i['label'].shape[1] == 3:
                label = data_i['label']
            else:
                label_vis = util.vis_tensor(data_i['label']) / 255
                label_ref_vis = util.vis_tensor(data_i['label_ref']) / 255
            cycleshow = None
            if opt.warp_cycle_w > 0:
                cycleshow = trainer.out['warp_cycle'] if opt.warp_patch else F.interpolate(trainer.out['warp_cycle'], scale_factor=opt.warp_stride)
                if opt.two_cycle:
                    cycleshow = torch.cat((cycleshow, F.interpolate(trainer.out['warp_i2r'], scale_factor=opt.warp_stride), F.interpolate(trainer.out['warp_i2r2i'], scale_factor=opt.warp_stride)), 0)
            if cycleshow is not None:
                imgs = torch.cat((label.cpu(), data_i['ref'].cpu(), trainer.out['warp_out'].cpu(), cycleshow.cpu(), trainer.get_latest_generated().data.cpu(), data_i['image'].cpu()), 0)
            else:
                # input label, ref_image, add-ref_seg, add-fake_ref_seg, warp_out, fake_image, gt_image.
                warp_ref_mask_256 = F.interpolate(trainer.out['warp_mask_to_ref'], size=[256,256], mode='bilinear')
                warped_label_vis = util.vis_generated_tensor(warp_ref_mask_256, data_i['label_ref']) / 255.
                imgs = torch.cat((label_vis.cpu(), data_i['ref'].cpu(), label_ref_vis.cpu(), warped_label_vis.cpu(), trainer.out['warp_out'].cpu(), trainer.get_latest_generated().data.cpu(), data_i['image'].cpu()), 0)
                # imgs = torch.cat((label_vis.cpu(), data_i['ref'].cpu(), label_ref_vis.cpu(), trainer.out['warp_out'].cpu(), trainer.get_latest_generated().data.cpu(), data_i['image'].cpu()), 0)
            try:
                vutils.save_image(imgs, save_root + opt.name + '/' + str(epoch) + '_' + str(iter_counter.total_steps_so_far) + '.png',  
                        nrow=imgs_num, padding=0, normalize=True)
            except OSError as err:
                print(err)
        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                (epoch, iter_counter.total_steps_so_far))
            try:
                trainer.save('latest')
                iter_counter.record_current_iter()
            except OSError as err:
                print(err)
    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
        epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
            (epoch, iter_counter.total_steps_so_far))
        try:
            trainer.save('latest')
            trainer.save(epoch)
        except OSError as err:
            print(err)

print('Training was successfully finished.')