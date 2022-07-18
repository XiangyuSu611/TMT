from pathlib import Path

"""
Directory configuration.
"""
LOCAL_ROOT = Path('/home/code/TMT')
DATA_ROOT = Path(LOCAL_ROOT, 'data')
BLENDER_JSON_PATH = Path(DATA_ROOT, 'training_data', 'json')
THIRD_ROOT = Path(LOCAL_ROOT, 'src', 'thirdparty')
MATERIAL_ROOT = Path(DATA_ROOT, 'materials')
MATERIAL_DIR_POLIIGON = MATERIAL_ROOT / 'polligon_0328'
MATERIAL_DIR_AITTALA = MATERIAL_ROOT / 'aittala-beckmann'
MATERIAL_DIR_VRAY = MATERIAL_ROOT / 'vray-materials-de'
MATERIAL_DIR_ADOBE_STOCK = MATERIAL_ROOT / 'adobe' / 'our'
MATERIAL_DIR_CCOTEXTURE = MATERIAL_ROOT / 'cc0texture'
MATERIAL_DIR_TEXTURE3D = MATERIAL_ROOT / '3Dtexture'
MATERIAL_DIR_TEXTUREHARVEN = MATERIAL_ROOT / 'texture_harven'
MATERIAL_DIR_SHARETEXTURE = MATERIAL_ROOT / 'shareTexture'

SHAPE_ROOT = Path(DATA_ROOT, 'shapes')
PAIR_ROOT = Path(DATA_ROOT, 'pairs')
MINC_VGG16_WEIGHTS_PATH = Path('/home/code/TMT_release/weights/minc_vgg16.npy')
RADMAP_PATH = Path(DATA_ROOT, 'envmaps2/rnl.cross.exr')
PAIRS_JSON_PATH = Path(DATA_ROOT, 'pairs/pairs.json')
ALIGN_POSE_PATH = Path(LOCAL_ROOT, 'src', 'data_generation', 'preprocess', 'shapes', 'align_pose.npy')

EXEMPLAR_SUBST_MAP_NAME = 'substance_map_minc_vgg16.map.v2.png'
EXEMPLAR_SUBST_VIS_NAME = 'substance_map_minc_vgg16.vis.v2.png'
EXEMPLAR_ALIGN_DATA_NAME = 'align_hog_8.npz'
SHAPE_ALIGN_DATA_NAME = 'align_hog_8.npz'
SHAPE_REND_SHAPE = (500, 500)
SHAPE_REND_SHAPE_STR = f'{SHAPE_REND_SHAPE[0]}x{SHAPE_REND_SHAPE[1]}'
PAIR_FG_BBOX_NAME = f'shape_fg_bbox_{SHAPE_REND_SHAPE_STR}.png'
PAIR_RAW_SEGMENT_MAP_NAME = f'shape_segment_map_raw_{SHAPE_REND_SHAPE_STR}.png'
SHAPE_REND_PHONG_NAME = f'shape_rend_phong_{SHAPE_REND_SHAPE_STR}0001.png'
SHAPE_REND_SEGMENT_MAP_NAME = f'shape_rend_segments_{SHAPE_REND_SHAPE_STR}.map0001.png'
SHAPE_REND_SEGMENT_VIS_NAME = f'shape_rend_segments_{SHAPE_REND_SHAPE_STR}.vis.png'
CAMEA_POSE_PIROR_PATH = Path('/home/code/TMT/src/data_generation/generation/camera_pose_piror.json')
FLOW_DATA_NAME = f'exemplar_rend_flow_silhouette_{SHAPE_REND_SHAPE_STR}.npz'
FLOW_VIS_DATA_NAME = f'exemplar_rend_flow_silhouette_{SHAPE_REND_SHAPE_STR}.png'
FLOW_EXEMPLAR_SILHOUETTE_VIS = f'flow_exemplar_silhouette_{SHAPE_REND_SHAPE_STR}.vis.png'
FLOW_SHAPE_SILHOUETTE_VIS = f'flow_shape_silhouette_{SHAPE_REND_SHAPE_STR}.vis.png'
PAIR_SEGMENT_OVERLAY_NAME = f'shape_segments_overlay.v2.png'
PAIR_PROXY_SUBST_MAP_NAME = f'pair_proxy_substances.map.png'
PAIR_PROXY_SUBST_VIS_NAME = f'pair_proxy_substances.vis.png'
PAIR_SHAPE_SUBST_MAP_NAME = f'pair_shape_substances.map.png'
PAIR_SHAPE_SUBST_VIS_NAME = f'pair_shape_substances.vis.png'
PAIR_FG_BBOX_NAME = f'shape_fg_bbox_{SHAPE_REND_SHAPE_STR}.png'
PAIR_RAW_SEGMENT_MAP_NAME = f'shape_segment_map_raw_{SHAPE_REND_SHAPE_STR}.png'
PAIR_SHAPE_WARPED_PHONG_NAME = f'shape_warped_phong_{SHAPE_REND_SHAPE_STR}_3.png'
PAIR_SHAPE_WARPED_SEGMENT_VIS_NAME = f'shape_warped_segments_{SHAPE_REND_SHAPE_STR}.vis.v2.png'
PAIR_SHAPE_WARPED_SEGMENT_MAP_NAME = f'shape_warped_segments_{SHAPE_REND_SHAPE_STR}.map.v2.png'
PAIR_SHAPE_CLEAN_SEGMENT_VIS_NAME = f'shape_clean_segments_{SHAPE_REND_SHAPE_STR}.vis.v2.png'
PAIR_SHAPE_CLEAN_SEGMENT_MAP_NAME = f'shape_clean_segments_{SHAPE_REND_SHAPE_STR}.map.v2.png'
PAIR_SHAPE_CLEAN_SEGMENT_MAP_NAME_OLD = f'shape_clean_segments_{SHAPE_REND_SHAPE_STR}.map.v2.png'
"""
Alignment parameters.
"""
ALIGN_BIN_SIZE = 8
ALIGN_IM_SHAPE = (100, 100)
# ALIGN_DIST_THRES = 8.0
ALIGN_DIST_THRES = 10.0
ALIGN_DIST_THRES_GEN = 100.0
ALIGN_TOP_K = 7

SUBSTANCES = [
    # chair and bed
    'fabric',
    'leather',
    'wood',
    'metal',
    'plastic',
    'background'
    # table
    # 'fabric',
    # 'stone',
    # 'wood',
    # 'metal',
    # 'plastic',
    # 'background',
]