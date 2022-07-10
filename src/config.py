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
"""
Alignment parameters.
"""
ALIGN_BIN_SIZE = 8
ALIGN_IM_SHAPE = (100, 100)
# ALIGN_DIST_THRES = 8.0
ALIGN_DIST_THRES = 10.0
ALIGN_DIST_THRES_GEN = 100.0
ALIGN_TOP_K = 7