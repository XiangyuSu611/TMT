"""
Computes shape to exemplar flows.
"""
import sys
sys.append('/home/code/TMT/src')
import config, controllers
import click as click
import numpy as np
import matlab.engine
import os
from flow import resize_flow, visualize_flow, apply_flow
from pathlib import Path
from skimage.io import imsave
from skimage import transform
from skimage.color import rgb2gray
from skimage.morphology import disk, binary_closing
from typing import List
from tempfile import NamedTemporaryFile
from tqdm import tqdm


script_dir = os.path.dirname(os.path.realpath(__file__))
siftflow_path = Path(script_dir, '../../thirdparty/siftflow').resolve()


def bright_pixel_mask(image, percentile=80):
    image = rgb2gray(image)
    perc = np.percentile(np.unique(image), percentile)
    mask = image < perc
    return mask


def main():
    shape_dir = str(partnet_Shape().data_path)
    shape_dir = shape_dir[:shape_dir.rfind('/')]
    id_list = os.listdir(shape_dir)

    for i in range(0,len(id_list)):
        print(f"Initializing MATLAB engine.")
        engine = matlab.engine.start_matlab()
        engine.addpath(str(siftflow_path))
        engine.addpath(str(siftflow_path / 'mexDenseSIFT'))
        engine.addpath(str(siftflow_path / 'mexDiscreteFlow'))

        
        with session_scope() as sess:
            pairs, count = controllers.fetch_pairs_default(sess, filters=filters)
            # pairs, count = controllers.fetch_pairs(
            #     sess,
            #     by_shape=True,
            #     order_by=ExemplarpartShapePair.shape_id.asc(),
            # )

            print(f"Fetched {count} pairs for {id_list[i]}..")

            pairs = [
                pair for pair in pairs
                if (pair.data_exists(config.SHAPE_REND_SEGMENT_MAP_NAME)
                    and not pair.data_exists(config.FLOW_DATA_NAME))
            ]

            print(f"Computing flows for {len(pairs)} pairs.")

            pbar = tqdm(pairs)
            pair: ExemplarpartShapePair
            for pair in pbar:
                pbar.set_description(f'Pair {pair.id}')
                # if not pair.exemplar.data_exists(config.EXEMPLAR_SUBST_MAP_NAME, type='numpy'):
                #     logger.warning('pair %d does not have substance map', pair.id)
                #     continue

                if not pair.data_exists(config.SHAPE_REND_SEGMENT_MAP_NAME):
                    print(f'Pair {pair.id} does not have segment map')
                    continue

                if pair.data_exists(config.FLOW_DATA_NAME):
                    continue

                exemplar_im = transform.resize(
                    pair.exemplar.load_cropped_image(), config.SHAPE_REND_SHAPE,
                    anti_aliasing=True, mode='reflect')
                seg_vis = pair.load_data(config.SHAPE_REND_SEGMENT_VIS_NAME)

                vx, vy = compute_silhouette_flow(engine, pair)
                flow_vis = visualize_flow(vx, vy)

                vis.image(flow_vis.transpose((2, 0, 1)),
                        win='sil-flow',
                        opts={'title': 'sil-flow'})
                vis.image(
                    ((exemplar_im + apply_flow(seg_vis, vx, vy))/2).transpose((2, 0, 1)),
                    win='sil-flow-applied',
                    opts={'title': 'sil-flow-applied'})

                # vx, vy = compute_phong_flow(engine, exemplar_im, phong_im)
                #
                # flow_vis = visualize_flow(vx, vy)
                # vis.image(flow_vis.transpose((2, 0, 1)),
                #           win='phong-flow',
                #           opts={'title': 'phong-flow'})
                # vis.image(
                #     ((exemplar_im + apply_flow(seg_vis, vx, vy))/2).transpose((2, 0, 1)),
                #     win='phong-flow-applied',
                #     opts={'title': 'phong-flow-applied'})

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pair.save_data(config.FLOW_VIS_DATA_NAME, flow_vis)
                pair.save_data(config.FLOW_DATA_NAME, np.dstack((vx, vy)))


def compute_silhouette_flow(engine, pair):
    """
    Compute silhouette based flow.
    """
    with NamedTemporaryFile(suffix='.png') as exemplar_f, \
         NamedTemporaryFile(suffix='.png') as shape_f:
        # exemplar_sil = pair.exemplar.load_data(config.EXEMPLAR_SUBST_MAP_NAME, type='numpy')
        # exemplar_sil = (exemplar_sil != 5)[:, :, None]
        base_pattern = np.dstack((
            np.zeros(config.SHAPE_REND_SHAPE), *np.meshgrid(
                np.linspace(0, 1, config.SHAPE_REND_SHAPE[0]),
                np.linspace(0, 1, config.SHAPE_REND_SHAPE[1]))))

        exemplar_sil = bright_pixel_mask(
            pair.exemplar.load_cropped_image(), percentile=95)
        exemplar_sil = binary_closing(exemplar_sil, selem=disk(3))
        exemplar_sil = transform.resize(exemplar_sil, (500, 500),
                                        anti_aliasing=True, mode='reflect')
        shape_sil = pair.load_data(config.SHAPE_REND_SEGMENT_MAP_NAME) - 1
        shape_sil = (shape_sil > -1)
        shape_sil = binary_closing(shape_sil, selem=disk(3))

        exemplar_sil_im = exemplar_sil[:, :, None].repeat(repeats=3, axis=2).astype(float)
        shape_sil_im = shape_sil[:, :, None].repeat(repeats=3, axis=2).astype(float)

        exemplar_sil_im[exemplar_sil == 0] = base_pattern[exemplar_sil == 0]
        shape_sil_im[shape_sil == 0] = base_pattern[shape_sil == 0]

        vis.image(exemplar_sil_im.transpose((2, 0, 1)), win='exemplar-sil')
        vis.image(shape_sil_im.transpose((2, 0, 1)), win='shape-sil')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(exemplar_f.name, exemplar_sil_im)
            imsave(shape_f.name, shape_sil_im)

        vx, vy = engine.siftflow(str(exemplar_f.name), str(shape_f.name),
                                 nargout=2)
        vx, vy = resize_flow(np.array(vx), np.array(vy),
                             shape=config.SHAPE_REND_SHAPE)
    return vx, vy


def compute_phong_flow(engine, exemplar_im, phong_im):
    with NamedTemporaryFile(suffix='.png') as exemplar_f, \
            NamedTemporaryFile(suffix='.png') as shape_f:
        #
        # Phong based flow.
        #
        imsave(exemplar_f.name, exemplar_im)
        imsave(shape_f.name, phong_im)

        vx, vy = engine.siftflow(str(exemplar_f.name), str(shape_f.name),
                 nargout=2)
        vx, vy = resize_flow(np.array(vx), np.array(vy),
                             shape=config.SHAPE_REND_SHAPE)
    return vx, vy

if __name__ == '__main__':
    main()