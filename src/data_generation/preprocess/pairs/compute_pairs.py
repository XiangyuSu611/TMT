"""
Generates exemplar-shape pairs.
"""
import sys
sys.path.append('/home/code/TMT/src')
import config
import click as click
import collections
import numpy as np
import json
import os
import time
import torch

from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm


@click.command()
@click.option('--batch-size', default=600, help='have better set as the 1/10 of the number of shapes')
@click.option('--num-workers', default=12)
@click.option('--start', default=0)
@click.option('--end', type=int)
@click.option('--topk', type=int, default=8)
@click.option('--prefetch', is_flag=True, help='')


def main(start, end, batch_size, num_workers, topk, prefetch):
    # data preparation.
    tqdm.write('Getting all shapes and exemplars.')
    shapes = sorted(Path(config.DATA_ROOT, 'shapes').iterdir())
    exemplars = sorted(Path(config.DATA_ROOT, 'exemplars').iterdir())
    # filter
    tqdm.write('Filtering shapes and exemplars with aligning features.')
    valid_shapes = [shape for shape in shapes
                    if os.path.exists(Path(shape, 'numpy', config.SHAPE_ALIGN_DATA_NAME))]
    valid_exemplars = [exemplar for exemplar in exemplars
                    if os.path.exists(Path(exemplar, 'numpy', config.EXEMPLAR_ALIGN_DATA_NAME))]
    tqdm.write(f'Using {len(valid_shapes)}/{len(shapes)} shapes, '
               f'{len(valid_exemplars)}/{len(exemplars)} exemplars')
    # generate dataset.
    dataset = AlignFeatureDataset(valid_shapes, prefetch=prefetch)
    shape_loader = DataLoader(
        dataset=dataset, sampler=SequentialSampler(dataset),
        shuffle=False, batch_size=batch_size, num_workers=num_workers)
    pair_matches_by_exemplar = collections.defaultdict(list)

    # Iterate over shapes in outer loop since loading them is considerably more expensive.
    exemplar_pbar = tqdm(valid_exemplars)
    new_pairs = []
    for exemplar in exemplar_pbar:
        exemplar_pbar.set_description(f'{exemplar}')
        process_exemplar(new_pairs, exemplar, shape_loader, topk=topk)
    pbar = tqdm(pair_matches_by_exemplar.items())

    # pbar.set_description('Choosing top-k matches')
    # n_pairs = 0
    # for exemplar_id, exemplar_pairs in pbar:
    #     exemplar_pairs = [pair for pair in exemplar_pairs if pair.distance
    #                     < config.ALIGN_DIST_THRES_GEN]
    #     exemplar_pairs.sort(key=lambda pair: pair.distance)
    #     for pair in exemplar_pairs[:config.ALIGN_TOP_K]:
    #         n_pairs += 1
    # tqdm.write(f'Saved {n_pairs} pairs')

    with open(config.PAIRS_JSON_PATH, 'w') as f1:
        json.dump(new_pairs, f1, indent=2)

def normalize_feats(batch):
    feat_norms = batch.norm(2, dim=1).view(batch.size(0), 1).expand(
        *batch.size())
    feats = (batch / feat_norms).cpu()
    return feats


def compute_feat_dists(batch_feats, image_feats):
    batch_feats = normalize_feats(batch_feats)
    image_feats = normalize_feats(image_feats.view(1, -1))
    return (batch_feats.cuda() - image_feats.cuda().expand(
        *batch_feats.size())).pow(2).sum(dim=1)


def process_exemplar(new_pairs, exemplar, shape_loader, topk, max_dist=40):
    shape_pbar = tqdm(shape_loader)
    best_pairs = []
    for batch_idx, batch in enumerate(shape_pbar):
        start_time1 = time.time()
        shape_ids, shape_feats, thetas, phis, fovs = batch
        shape_feats = shape_feats.float().cuda()
        shape_pbar.set_description(f'Batch {batch_idx}')

        exemplar_feat = np.load(Path(exemplar, 'numpy', config.EXEMPLAR_ALIGN_DATA_NAME), allow_pickle=True)['arr_0']
        exemplar_feat = (torch.from_numpy(exemplar_feat)
                         # Dimensions: (shape, viewpoint, feat_dims).
                         .view(1, 1, -1)
                         .float()
                         .cuda())

        # L2 dist from exemplar feature for all shapes/viewpoints.
        # Output shape: (batch_size, 456)
        dists = ((shape_feats - exemplar_feat.expand(*shape_feats.size()))
                 .pow(2).sum(dim=2))
        dists, vp_inds = torch.min(dists, dim=1)
        batch_best_dist, batch_best_ind = torch.min(dists, dim=0)
        dists.cpu()

        i = int(batch_best_ind.cpu().item())
        dist = float(batch_best_dist.cpu().item())
        if dist > max_dist:
            continue

        vp_ind = vp_inds[batch_best_ind]
        # If we haven't collect at least top-k pairs or this is better than the
        # worst collected one.
        if len(best_pairs) < topk or dist < best_pairs[-1].distance:
            new_pair = {
                    "id": len(new_pairs) + 1,
                    'exemplar': str(exemplar),
                    'shape': str(shape_ids[i]),
                    'azimuth': float(thetas[i][vp_ind]),
                    'elevation': float(phis[i][vp_ind]),
                    'fov': float(fovs[i][vp_ind]),
                    'distance': dist,
                    'feature_type': config.SHAPE_ALIGN_DATA_NAME
                    }
            new_pairs.append(new_pair)
            # Add pair to list and remove previous match.
            best_pairs.append(new_pair)
            # best_pairs.sort(key=lambda p: p.distance)
            best_pairs = best_pairs[:topk]
        end_time1 = time.time()
        shape_pbar.set_description(f'Batch {batch_idx}, load time: {end_time1 - start_time1}')
    


class AlignFeatureDataset(Dataset):
    def __init__(self, shapes, prefetch=False):
        self.shapes = shapes

        self.data_list = []
        self.prefetch = prefetch

        if prefetch:
            tqdm.write('Prefetching shape data')
            path = Path('/local1/kpar/data/terial/shape_data.pth')
            if path.exists():
                tqdm.write(f"Loading shape data from {path!s}")
                self.data_list = torch.load(str(path))
            else:
                tqdm.write('Loading shape data individually')
                pbar = tqdm(shapes)
                for shape in pbar:
                    pbar.set_description(f'Shape {shape.id}')
                    self.data_list.append(self._load_data(shape))
                tqdm.write(f"Saving shape data to {path!s}")
                torch.save(self.data_list, str(path))

    @staticmethod
    def _load_data(shape):
        data = np.load(Path(shape, 'numpy', config.SHAPE_ALIGN_DATA_NAME), allow_pickle=True)
        # Support new format.
        if 'arr_0' in data:
            data = data['arr_0'][()]
        feats = torch.from_numpy(data['feats'][()].astype(np.float16))
        thetas = torch.from_numpy(data['thetas'][()].astype(np.float16))
        phis = torch.from_numpy(data['phis'][()].astype(np.float16))
        fovs = torch.from_numpy(data['fovs'][()].astype(np.float16))
        return str(shape), feats, thetas, phis, fovs

    def __getitem__(self, index):
        """
        The shape of the feats returned is (batch_size, n_viewpoints, feat_dims)
        which means the batch size is the number of shapes. Each shape has 456
        different viewpoints.

        :param index:
        :return:
        """

        if self.prefetch:
            shape_id, feats, thetas, phis, fovs = self.data_list[index]
        else:
            shape = self.shapes[index]
            shape_id, feats, thetas, phis, fovs = self._load_data(shape)

        return (shape_id, feats.float(), thetas.float(), phis.float(),
                fovs.float())

    def __len__(self):
        return len(self.shapes)


if __name__ == '__main__':
    main()
