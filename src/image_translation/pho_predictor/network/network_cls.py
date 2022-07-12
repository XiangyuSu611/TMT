import math
import torch
from torch import nn
from torchvision.models import resnet

import sys
sys.path.append('./src')
sys.path.append('.')
from torchvision import transforms, models


class FLModel(nn.Module):

    def __init__(self, pretrained, layers_to_remove, num_features, num_materials, num_substances,
                input_size=3, train_only_classifier=False,):
        super().__init__()

        # extract number of features in the last layer.
        in_features = list(pretrained.children())[-layers_to_remove].in_features
        # build the model.
        old_layers = list(pretrained.children())[:-layers_to_remove]

        if input_size != 3:
            first_conv = old_layers[0]
            old_layers[0] = nn.Conv2d(in_channels=input_size,
                                      out_channels=first_conv.out_channels,
                                      padding=first_conv.padding,
                                      kernel_size=first_conv.kernel_size,
                                      stride=first_conv.stride,
                                      bias=first_conv.bias,)

            old_layers[0].weight.data[:, :3, :, :].copy_(
                first_conv.weight.data[:, :3, :, :])

        # Pretrained network does not have alpha channel, so initialize it with random normal.
        for m in [old_layers[0]]:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data[:, 3, :, :].normal_(0, math.sqrt(2. / n))
        
        self.new_model = nn.Sequential(*old_layers)
        # input = 512 output = num_features (default 128)
        self.fc = nn.Linear(in_features, num_features)

        # needed only if we want to do classification
        # classificate mateiral.
        self.fc_material = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_materials)
        ) 
        # classificate substance.
        self.fc_substance = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_substances),
        )
        
        self.train_only_classifier = train_only_classifier
        
        if self.train_only_classifier:
            # do not need grad.
            for params in self.new_model.parameters():
                params.requires_grad = False
            for params in self.fc.parameters():
                params.requires_grad = False


    def forward(self, x):
        x = self.new_model(x)
        x = x.view(x.size(0), -1)
        embeddings = self.fc(x)
        embeddings = nn.ReLU()(embeddings)
        pred_mat = self.fc_material(embeddings)
        pred_sub = self.fc_substance(embeddings)
        return pred_mat, pred_sub, embeddings