'''
    defines all loss functions that will be used in the training process of the predictor,
    and a calculation function that provides accuracy,
    the triples used to calculate the triplet loss were sampled in advance and stored in the txt file
'''

import random
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.material_prediction.utils import compute_precision
from src.material_prediction.networks import utils as simi_utils


class TripletLossHuman(nn.Module):
    def __init__(self, margin=0.3, unit_norm=False, device=None, seed=None, \
            weight_human=1, weight_per=1, weight_mat=1, weight_sub=0.5, weight_dis=1, is_classification=False, ues_distance_loss=True):
        super(TripletLossHuman, self).__init__()

        # set seeds for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # user answers path
        train_triplet_path = 'data/training_data/material_prediction/triplet_label_train.json'
        test_triplet_path = 'data/training_data/material_prediction/triplet_label_test.json'

        # norm by row similarity matrix path
        similarity_matrix_path = 'data/training_data/material_prediction/similarity_matrix_norm_row.csv'

        # load norm by row similarity matrix.
        mat_dis = pd.read_csv(similarity_matrix_path, header=None)
        self.label2matDis = {}
        label_num = 0
        for i in range(1, mat_dis.values.shape[0]):
            list_i = mat_dis.values[i]
            self.label2matDis[label_num] = torch.from_numpy(list_i[1:]).to(device)
            label_num += 1

        with open(train_triplet_path) as f:
            train_triplet_data = json.load(f)
        with open(test_triplet_path) as f:
            test_triplet_data = json.load(f)

        # set loss variables
        self.margin = margin
        self.unit_norm = unit_norm
        self.weight_human = weight_human
        self.weight_per = weight_per
        self.weight_mat = weight_mat
        self.weight_sub = weight_sub
        self.weight_dis = weight_dis

        # triplet loss function used to model user answers
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin).to(device)
        self.ce_loss1 = nn.CrossEntropyLoss().to(device)
        self.ce_loss2 = nn.CrossEntropyLoss().to(device)
        self.softmax = nn.Softmax(dim=1).to(device)

        # get user answers and agreement (note that we only take different
        # agreements because an agreement of 2-2 will not be suitable for
        # training a triplet metric)
        self.user_answers_train = train_triplet_data['train_answer_diff']
        self.user_answers_test = test_triplet_data['test_answer_diff']

        self.user_answers_train = torch.tensor(self.user_answers_train)
        self.user_answers_test = torch.tensor(self.user_answers_test)

        self.user_answers_train = self.user_answers_train.to(device).long()
        self.user_answers_test = self.user_answers_test.to(device).long()
        self.is_classification = is_classification
        self.ues_distance_loss = ues_distance_loss

    def forward(self, preds_mat, preds_sub, inputs, targets_mat, targets_sub):
        """
        Args:
            preds_mat: predicted material tensor with shape (batch_size, num_of_mat_class)
            preds_sub: predicted substance tensor with shape (batch_size, num_of_sub_class)
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets_mat: ground truth material labels with shape (batch_size, )
            targets_mat: ground truth substance labels with shape (batch_size, )
        """
        if self.is_classification == False:
            if self.unit_norm:
                inputs = inputs / inputs.norm(dim=1, keepdim=True)

            # Compute pairwise distances
            dist = simi_utils.pairwise_dist(inputs)

            # move answers to correct device
            targets_mat = targets_mat.long()

            # get triplets answered by humans with representation in the batch
            is_there = torch.zeros_like(self.user_answers_train)
            for target_mat in torch.unique(targets_mat):
                is_there = is_there + (self.user_answers_train == target_mat).long()
            idx_triplets = (is_there.sum(dim=1) == 3).nonzero()
            if len(idx_triplets) == 0:
                warnings.warn('\nZero sampled triplets. '
                            'Consider increasing your batch size')
                return torch.zeros(1, requires_grad=True).to(targets_mat.device), \
                        torch.zeros(1, requires_grad=True).to(targets_mat.device), \
                        torch.zeros(1, requires_grad=True).to(targets_mat.device), \
                        0, \
                        0, \
                        0

            # if getting too much triplet, then clip the triplet number to 300.
            # if len(idx_triplets) > 300:
            #     rand_idx=torch.randperm(idx_triplets.size(0))
            #     idx_triplets = idx_triplets[rand_idx]
            #     idx_triplets = idx_triplets[:300]
            #     idx_triplets = torch.sort(idx_triplets)[0]

            dist_ap = torch.zeros(len(idx_triplets))
            dist_an = torch.zeros(len(idx_triplets))
            # target_agreement = torch.zeros(len(idx_triplets))

            dist_ap = dist_ap.to(inputs.device, inputs.dtype)
            dist_an = dist_an.to(inputs.device, inputs.dtype)
            # target_agreement = target_agreement.to(inputs.device, inputs.dtype)

            targets_triplet = self.user_answers_train[idx_triplets].squeeze()
            # keep targets_triplet to 2D tensor
            if len(targets_triplet.shape) == 1:
                targets_triplet = targets_triplet.unsqueeze(dim=0)
                
            for i, target_triplet in enumerate(targets_triplet):
                # get index of the triplet according to the given targets_mat
                triplet_idx = \
                    (targets_mat.view(1, -1) == target_triplet.view(-1, 1)).nonzero()

                # get single elements for each class.
                ix0 = triplet_idx[(triplet_idx[:, 0] % 3 == 0).nonzero()].squeeze()
                ix1 = triplet_idx[(triplet_idx[:, 0] % 3 == 1).nonzero()].squeeze()
                ix2 = triplet_idx[(triplet_idx[:, 0] % 3 == 2).nonzero()].squeeze()

                # At the end, the classes repeated have the same feature vector
                if len(ix0.shape) > 1: ix0 = ix0[0]
                if len(ix1.shape) > 1: ix1 = ix1[0]
                if len(ix2.shape) > 1: ix2 = ix2[0]

                # get distances and agreement
                dist_ap[i] = dist[ix0[1], ix1[1]]
                dist_an[i] = dist[ix0[1], ix2[1]]
                # target_agreement[i] = (self.user_agreement_train[idx, 0] /
                #                        self.user_agreement_train[idx].sum())

            tensor_y = torch.full(dist_ap.shape, -1).to(inputs.device)
            # Compute ranking hinge loss
            loss_human = self.ranking_loss(dist_ap, dist_an,
                                        tensor_y)

            # move from distances to probabilities
            s_ap = 1 / (dist_ap + 1)
            s_an = 1 / (dist_an + 1)

            # compute perplexity loss
            p_ap = s_ap / (s_ap + s_an)
            loss_perplexity = (-torch.log(p_ap + 1e-8)).mean()

            loss_material = 0
            loss_substance = 0
            loss_distance = 0
            # loss_material = self.ce_loss1(preds_mat, targets_mat)
            # loss_substance = self.ce_loss2(preds_sub, targets_sub)

            # get total loss and return
            loss = self.weight_human * loss_human + self.weight_per * loss_perplexity + \
                    self.weight_mat* loss_material + self.weight_sub * loss_substance + self.weight_dis * loss_distance

            return loss, loss_human, loss_perplexity, loss_material, loss_substance, loss_distance
        
        # if is_classification == True, then start to compute material loss and substance loss
        else:
            loss_human = 0
            loss_perplexity = 0

            loss_material = self.ce_loss1(preds_mat, targets_mat)
            loss_substance = self.ce_loss2(preds_sub, targets_sub)

            if self.ues_distance_loss == True:
                for j in range(preds_mat.shape[0]):
                    mat_out_softmax = self.softmax(preds_mat[j:j+1,:]).squeeze()
                    weight = self.label2matDis[targets_mat[j].item()].float()
                    if j == 0:
                        dis_material = torch.dot(mat_out_softmax, weight)
                    else:
                        dis_material = dis_material + torch.dot(mat_out_softmax, weight)
                loss_distance = dis_material / preds_mat.shape[0] # 由 batch_size 改为 preds_mat.shape[0]
            else:
                loss_distance = 0

            loss = self.weight_human * loss_human + self.weight_per * loss_perplexity + \
                    self.weight_mat* loss_material + self.weight_sub * loss_substance + self.weight_dis * loss_distance

            return loss, loss_human, loss_perplexity, loss_material, loss_substance, loss_distance

    def get_simi_accuracy(self, embeddings, targets_mat):
        if self.unit_norm:
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

        # Compute pairwise distances
        dist = simi_utils.pairwise_dist(embeddings)

        targets_mat = targets_mat.long()
        
        right_simi = 0

        # move answers to correct device
        targets_mat = targets_mat.long()

        # get triplets answered by humans with representation in the batch
        is_there = torch.zeros_like(self.user_answers_test)
        for target_mat in torch.unique(targets_mat):
            is_there = is_there + (self.user_answers_test == target_mat).long()
        idx_triplets = (is_there.sum(dim=1) == 3).nonzero()
        if len(idx_triplets) == 0:
            warnings.warn('\nZero sampled triplets. '
                          'Consider increasing your batch size')
            return 0, 0

        # if getting too much triplet, then clip the triplet number to 300.
        # if len(idx_triplets) > 300:
        #     rand_idx=torch.randperm(idx_triplets.size(0))
        #     idx_triplets = idx_triplets[rand_idx]
        #     idx_triplets = idx_triplets[:300]
        #     idx_triplets = torch.sort(idx_triplets)[0]

        dist_ap = torch.zeros(len(idx_triplets))
        dist_an = torch.zeros(len(idx_triplets))

        dist_ap = dist_ap.to(embeddings.device, embeddings.dtype)
        dist_an = dist_an.to(embeddings.device, embeddings.dtype)


        targets_triplet = self.user_answers_test[idx_triplets].squeeze()
        # keep targets_triplet to 2D tensor
        if len(targets_triplet.shape) == 1:
            targets_triplet = targets_triplet.unsqueeze(dim=0)

        for i, target_triplet in enumerate(targets_triplet):
            # get index of the triplet according to the given targets_mat
            triplet_idx = \
                (targets_mat.view(1, -1) == target_triplet.view(-1, 1)).nonzero()

            # get single elements for each class.
            ix0 = triplet_idx[(triplet_idx[:, 0] % 3 == 0).nonzero()].squeeze()
            ix1 = triplet_idx[(triplet_idx[:, 0] % 3 == 1).nonzero()].squeeze()
            ix2 = triplet_idx[(triplet_idx[:, 0] % 3 == 2).nonzero()].squeeze()

            # At the end, the classes repeated have the same feature vector
            if len(ix0.shape) > 1: ix0 = ix0[0]
            if len(ix1.shape) > 1: ix1 = ix1[0]
            if len(ix2.shape) > 1: ix2 = ix2[0]

            # get distances and agreement
            dist_ap[i] = dist[ix0[1], ix1[1]]
            dist_an[i] = dist[ix0[1], ix2[1]]
            # target_agreement[i] = (self.user_agreement_train[idx, 0] /
            #                        self.user_agreement_train[idx].sum())

        correct_number = (dist_ap < dist_an).sum().item()
        total_number = dist_ap.shape[0]
        
        return correct_number, total_number

    def get_pred_accuracy(self, preds_mat, preds_sub, targets_mat, targets_sub, similarity_matrix):
        '''
            check the prediction accuracy of the trained material predictor on the material,
            including the accuracy mat_acc for the material, and the accuracy sub_acc for the material category
            and the unnormalized raw similarity distance between the predicted material class and the GT material class
        '''
        mat_acc1, mat_acc5 = compute_precision(
            preds_mat, targets_mat, topk=(1, 5))
        
        sub_acc = compute_precision(preds_sub, targets_sub)

        preds_mat_softmax = self.softmax(preds_mat)
        preds_sub_softmax = self.softmax(preds_sub)
        preds_mat_label = torch.argmax(preds_mat_softmax, dim=1)
        preds_sub_label = torch.argmax(preds_sub_softmax, dim=1)

        mat_dist = 0.

        for index in range(targets_mat.shape[0]):
            pred_mat_label = preds_mat_label[index].item()
            target_mat = targets_mat[index].item()
            gt_dist = similarity_matrix[target_mat].float()
            mat_dist += gt_dist[pred_mat_label].item()

        return mat_acc1, mat_acc5, sub_acc, mat_dist / preds_mat.shape[0]

