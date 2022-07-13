import os
import PIL
import torch
from torchvision import models, transforms


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
DEFAULT_TRF = transforms.Compose([
    transforms.Resize(size=(256)),
    transforms.CenterCrop(size=(IMG_SIZE)),
    transforms.ToTensor(),
])


def load_model(weights_path=None):
    """ Load model with IMAGENET weights if other pretrained weights
    are not given
    """
    model, layers_to_remove = models.resnet34(
        pretrained=weights_path is None), 1
    model = FTModel(model,
                    layers_to_remove=layers_to_remove,
                    num_features=128,
                    num_classes=100,
                    train_only_fc=False)

    if weights_path is not None:
        print('loading model weights')
        if os.path.isfile(weights_path):
            print(" => loading checkpoint '{}'".format(weights_path))
            checkpoint = torch.load(weights_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['state_dict'])
            print(" => loaded checkpoint '{}' (epoch {})"
                  .format(weights_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(weights_path))

    model.to(DEVICE)
    model.eval()

    return model


def pil_loader(image_path):
    """loads a path in a PIL image
    """
    return PIL.Image.open(image_path).convert('RGB')


def load_imgs(imgs_path, trf_test=None):
    """ Loads imgs from a folder. If no transforms are given it just
        resizes and performs a center crop of the image.
    """

    if trf_test is None:
        trf_test = DEFAULT_TRF

    # get all the image paths
    img_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(imgs_path)
                 for f in filenames]
    img_paths.sort()

    # prepare data structures
    imgs_tensor = torch.zeros(len(img_paths), 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

    # load all the images
    print('loading images')
    for i, img_path in enumerate(img_paths):
        imgs_tensor[i] = trf_test(pil_loader(img_path))

    return imgs_tensor, img_paths


def pairwise_dist(x):
    """ Computes pairwise distances between features
        x : torch tensor with shape Batch x n_features
    """
    n = x.size(0)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, x, x.t())
    dist = dist.clamp(min=1e-12).sqrt()  # numerical stability
    return dist

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