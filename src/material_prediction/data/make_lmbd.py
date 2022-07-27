import io
import os 
import lmdb
import argparse

import ujson as json
from pathlib import Path

import msgpack
from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--snapshot-dir', dest='snapshot_dir', type=Path,
                    default='./data/training_data/image_translation/')
parser.add_argument('--lmdb-dir', dest='lmdb_dir', type=Path,
                    default='./data/training_data/material_prediction/')
parser.add_argument('--split-set', type=str, 
                    default='training', choices=['training','validation'])
parser.add_argument('--save-meta', action='store_true')
args = parser.parse_args()
SHAPE = (500, 500)

def main():
    snapshot_dir = args.snapshot_dir
    lmdb_dir = args.lmdb_dir
    # with session_scope() as sess:
    #     enabled_materials = (sess.query(models.Material)
    #                   .order_by(models.Material.substance,
    #                             models.Material.id)
    #                   .all())
    #     material_by_id = {
    #         m.id: m
    #         for m in enabled_materials
    #     }
    with open("./data/materials/materials.json", "r") as f:
        enabled_materials = json.load(f)
        material_by_id = {
            m["id"]: m 
            for m in enabled_materials
        }

    print(f"Fetched {len(enabled_materials)} enabled materials.")

    if input(f"This will create an LMDB database at {snapshot_dir!s} for the "
             f"{args.split_set.upper()} set. Continue? (y/n) ") != 'y':
        return

    snapshot_dir.mkdir(exist_ok=True, parents=True)

    mat_id_to_label = {
        material["id"]: i for i, material in enumerate(enabled_materials)
    }

    if args.save_meta:
        print('enter save_meta.')
        with (lmdb_dir / 'meta.json').open('w') as f:
            json.dump({
                'mat_id_to_label': mat_id_to_label,
            }, f, indent=2)

    tqdm.write(f"Fetching renderings from snapshot_dir.")
    tar_render_path = []
    if args.split_set == 'validation':
        renders_dir = 'validation'
    elif args.split_set == 'training':
        renders_dir = 'training'
    
    renders = os.listdir(snapshot_dir / renders_dir)
    for render_path in renders:
        if '.json' in render_path:
            tar_render_path.append(str(snapshot_dir) + '/' + renders_dir + '/' + render_path[0:render_path.find('.')])
        
    train_env = lmdb.open((str(lmdb_dir) + '/' + args.split_set),
                          map_size=30000*200000)
    
    with train_env.begin(write=True) as txn:
        pbar = tqdm(tar_render_path)
        for rend in pbar:
            index = int(rend[-8:])
            ldr_im = Image.open(rend + '.jpg')
            seg_map = Image.open(rend+ '.png')
            ldr_bytes = io.BytesIO()
            ldr_im.save(ldr_bytes, format='JPEG')
            seg_map_bytes = io.BytesIO()
            seg_map.save(seg_map_bytes, format='PNG') 
            
            with open(rend + '.json') as f:
                rend_params = json.load(f)
            mat_id_by_seg_name = rend_params['tar_segment']['material_ids']

            changed_segment_ids_dict = rend_params['segment']['segment_ids']
            for seg_name in changed_segment_ids_dict:
                changed_segment_ids_dict[seg_name] = int(seg_name[seg_name.find('_') + 1:])
            seg_material_ids = {
                seg_id: mat_id_by_seg_name[seg_name]
                for seg_name, seg_id in changed_segment_ids_dict.items()
                if seg_name in mat_id_by_seg_name
            }

            seg_substances = {
                seg_id: material_by_id[mat_id]["substance"]     
                for seg_id, mat_id in seg_material_ids.items()
            }
            
            payload = msgpack.packb({
                'rend_id': rend_params['shape_id'],
                'pair_id': rend_params['pair_id'],
                'ldr_image': ldr_bytes.getvalue(),
                'segment_map': seg_map_bytes.getvalue(),
                'seg_material_ids': seg_material_ids,
                'seg_substances': seg_substances,
            })
            txn.put(f'{index:08d}'.encode(), payload)

if __name__ == '__main__':
    main()
