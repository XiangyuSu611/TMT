import math
from functools import partial

import torch
from torch.autograd import Variable


def chi_squared_dist(h1, h2):
    eps = 0.0000001
    h1 = h1 + eps
    h2 = h2 + eps
    s = (((h1 - h2).pow(2)) / (h1 + h2)).sum(dim=2).squeeze()
    return s.mean()


def gaussian_kernel(x, mu, sigma):
    return 1.0 / (sigma * math.sqrt(2*math.pi)) * torch.exp(-(x - mu)**2 / (2*sigma**2))


def triangle_kernel(batch, bin_val, span):
    k = 1 - torch.abs(batch - bin_val) / span
    k = (k >= 0).float() * k
    return k


def epanechnikov_kernel(batch, bin_val, span):
    k = 1 - ((batch - bin_val) / span).pow(2)
    k = 3/4 * (k >= 0).float() * k
    return k


def uniform_kernel(batch, bin_val, span):
    return ((batch - bin_val) <= span / 2) * ((batch - bin_val) >= -span / 2)


def batch_histogram(batch, bins=32, mask_batch=None,
                    kern_func=None):
    if kern_func is None:
        kern_func = partial(uniform_kernel, span=1/bins)

    eps = 0.0000001
    batch_size = (*batch.size()[:2], batch.size(2) * batch.size(3))
    binvals = Variable(torch.linspace(0, 1, bins).cuda())
    # Expand values so we compute histogram in parallel.
    binvals = binvals.view(1, 1, 1, bins).expand(*batch_size, bins)
    batch = batch.view(*batch_size, 1).expand(*batch_size, bins)
    hist_responses = kern_func(batch, binvals).float()
    if mask_batch is not None:
        mb_size = (mask_batch.size(0), mask_batch.size(1), mask_batch.size(2) * mask_batch.size(3))
        mask_batch = mask_batch.view(*mb_size, 1).expand(*batch_size, bins)
        hist_responses = hist_responses * mask_batch
    hist = hist_responses.sum(dim=2).unsqueeze(2)[:, :, 0, :] + eps
    hist /= hist.sum(dim=2).unsqueeze(2).expand(hist.size()) # L1 normalize (to PDF).
    return hist
