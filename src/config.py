from pathlib import Path

"""
Directory configuration.
"""
LOCAL_ROOT = Path('/home/code/TMT')
DATA_ROOT = Path(LOCAL_ROOT, 'data')
THIRD_ROOT = Path(LOCAL_ROOT, 'src', 'thirdparty')

MINC_VGG16_WEIGHTS_PATH = Path('/home/code/TMT_release/weights/minc_vgg16.npy')
RADMAP_PATH = Path(DATA_ROOT, 'envmaps2/rnl.cross.exr')
PAIRS_JSON_PATH = Path(DATA_ROOT, 'pairs/pairs.json')

EXEMPLAR_SUBST_MAP_NAME = 'substance_map_minc_vgg16.map.v2.png'
EXEMPLAR_SUBST_VIS_NAME = 'substance_map_minc_vgg16.vis.v2.png'
EXEMPLAR_ALIGN_DATA_NAME = 'align_hog_8.npz'
SHAPE_ALIGN_DATA_NAME = 'align_hog_8.npz'
SHAPE_REND_SHAPE = (500, 500)
PAIR_FG_BBOX_NAME = f'shape_fg_bbox_{SHAPE_REND_SHAPE_STR}.png'
PAIR_RAW_SEGMENT_MAP_NAME = f'shape_segment_map_raw_{SHAPE_REND_SHAPE_STR}.png'
SHAPE_REND_PHONG_NAME = f'shape_rend_phong_{SHAPE_REND_SHAPE_STR}.png'
SHAPE_REND_SHAPE_STR = f'{SHAPE_REND_SHAPE[0]}x{SHAPE_REND_SHAPE[1]}'
SHAPE_REND_SEGMENT_MAP_NAME = f'shape_rend_segments_{SHAPE_REND_SHAPE_STR}.map.png'
SHAPE_REND_SEGMENT_VIS_NAME = f'shape_rend_segments_{SHAPE_REND_SHAPE_STR}.vis.png'
CAMEA_POSE_PIROR_PATH = Path('/home/code/TMT/src/data_generation/generation/camera_pose_piror.json')

"""
Alignment parameters.
"""
ALIGN_BIN_SIZE = 8
ALIGN_IM_SHAPE = (100, 100)
# ALIGN_DIST_THRES = 8.0
ALIGN_DIST_THRES = 10.0
ALIGN_DIST_THRES_GEN = 100.0
ALIGN_TOP_K = 7