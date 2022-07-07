import torch
from torch.autograd import Variable


imagenet_std = torch.FloatTensor((0.229, 0.224, 0.225)).view(3, 1, 1)
imagenet_mean = torch.FloatTensor((0.485, 0.456, 0.406)).view(3, 1, 1)


def imagenet_normalize(var):
    if var.is_cuda:
        std = imagenet_std.cuda()
        mean = imagenet_mean.cuda()
    else:
        std = imagenet_std
        mean = imagenet_mean

    if isinstance(var, Variable):
        var = var - Variable(mean)
        var = var / Variable(std)
    else:
        var = var - mean
        var = var / std
    return var


def imagenet_denormalize(var):
    if var.is_cuda:
        std = imagenet_std.cuda()
        mean = imagenet_mean.cuda()
    else:
        std = imagenet_std
        mean = imagenet_mean

    if isinstance(var, Variable):
        var = var * Variable(std)
        var = var + Variable(mean)
    else:
        var = var * std
        var = var + mean
    return var
