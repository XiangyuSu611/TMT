"""
estimate camera pose using camera pose estimation network.
"""
import sys
sys.path.append('/home/code/TMT/material_transfer/cam_estimation')

import argparse
import cv2
import json
import numpy as np
import math
from cam_est import model_cam_old as model_cam
import model_normalization as model
import os
import socket
import skimage
import skimage.io
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tqdm import tqdm
from skimage.color import rgb2gray

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'cam_est'))
sys.path.append(os.path.join(BASE_DIR, 'data'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'preprocessing'))

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--max_epoch', type=int, default=1, help='Epoch to run [default: 201]')
parser.add_argument('--img_h', type=int, default=500, help='Image Height')
parser.add_argument('--img_w', type=int, default=500, help='Image Width')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate [default: 0.001]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--num_classes', type=int, default=1024, help='vgg dim')
parser.add_argument('--num_points', type=int, default=1, help='Point Number [default: 2048]')
parser.add_argument('--sdf_res', type=int, default=64, help='sdf grid')
parser.add_argument('--alpha', action='store_true')
parser.add_argument('--rot', action='store_true')
parser.add_argument('--tanh', action='store_true')
parser.add_argument('--cat_limit', type=int, default=168000, help="balance each category, 1500 * 24 = 36000")
parser.add_argument('--multi_view', action='store_true')
parser.add_argument('--num_sample_points', type=int, default=1, help='Sample Point Number [default: 2048]')
parser.add_argument('--shift', action="store_true")
parser.add_argument('--loss_mode', type=str, default="3D", help='loss on 3D points or 2D points')
parser.add_argument('--log_dir', default='./material_transfer/cam_estimation/SDF_DISN_log', help='Log dir [default: log]')
parser.add_argument('--cam_log_dir', default='./cam_est/cam_DISN', help='Log dir [default: log]')
parser.add_argument('--threedcnn', action='store_true')
parser.add_argument('--img_feat_onestream', action='store_true')
parser.add_argument('--img_feat_twostream', action='store_true')
parser.add_argument('--category', default="all", help='Which single class to train on [default: None]')
parser.add_argument('--binary', action='store_true')
parser.add_argument('--create_obj', action='store_true', help="create_obj or test accuracy on test set")
parser.add_argument('--store', action='store_true')
parser.add_argument('--view_num', type=int, default=24, help="how many views do you want to create for each obj")
parser.add_argument('--cam_est', action='store_true', help="if you are using the estimated camera image h5")
parser.add_argument('--augcolorfore', action='store_true')
parser.add_argument('--augcolorback', action='store_true')
parser.add_argument('--backcolorwhite', action='store_true')

FLAGS = parser.parse_args()

NUM_POINTS = FLAGS.num_points
BATCH_SIZE = FLAGS.batch_size
RESOLUTION = FLAGS.sdf_res+1
TOTAL_POINTS = RESOLUTION * RESOLUTION * RESOLUTION
GPU_INDEX = FLAGS.gpu
PRETRAINED_MODEL_PATH = FLAGS.log_dir
LOG_DIR = FLAGS.log_dir
SDF_WEIGHT = 10.

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

if FLAGS.cam_est:
    RESULT_OBJ_PATH = os.path.join("./demo/")
    print("RESULT_OBJ_PATH: ",RESULT_OBJ_PATH)
else:
    RESULT_OBJ_PATH = os.path.join("./demo/")

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_test.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

IMG_SIZE = FLAGS.img_h
HOSTNAME = socket.gethostname()
print("HOSTNAME:", HOSTNAME)

class NoStdStreams(object):
    def __init__(self,stdout = None, stderr = None):
        self.devnull = open(os.devnull,'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def sp2co(cam_json):
    azi = cam_json['azimuth']
    ele = cam_json['elevation']
    dis = 2.3
    x = dis * math.cos(azi) * math.sin(ele)
    y = dis * math.cos(ele)
    z = dis * math.sin(azi) * math.sin(ele)
    return [x,y,z]


def cam_evl(img_arr):
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            input_pls = model_cam.placeholder_inputs(1, NUM_POINTS, (IMG_SIZE, IMG_SIZE), num_pc=NUM_POINTS,
                                                        num_sample_pc=1, scope='inputs_pl')
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0, name='batch')
            print("--- cam Get model_cam and loss")
            
            # Get model and loss
            end_points = model_cam.get_model(input_pls, NUM_POINTS, is_training_pl, img_size=(IMG_SIZE, IMG_SIZE), bn=False, wd=2e-3, FLAGS=FLAGS)
            loss, end_points = model_cam.get_loss(end_points, sdf_weight=SDF_WEIGHT, FLAGS=FLAGS)
            tf.summary.scalar('loss', loss)

            print("--- Get training operator")
            # Create a session
            config = tf.ConfigProto()
            gpu_options = tf.GPUOptions()  # per_process_gpu_memory_fraction=0.99)
            config = tf.ConfigProto(gpu_options=gpu_options)
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)
            # Init variables
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver([v for v in tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES) if
                                    ('lr' not in v.name) and ('batch' not in v.name)])
            ckptstate = tf.train.get_checkpoint_state(FLAGS.cam_log_dir)

            if ckptstate is not None:
                LOAD_MODEL_FILE = os.path.join(FLAGS.cam_log_dir, os.path.basename(ckptstate.model_checkpoint_path))
                try:
                    with NoStdStreams():
                        saver.restore(sess, LOAD_MODEL_FILE)
                    print("model_cam loaded in file: %s" % LOAD_MODEL_FILE)
                except:
                    print("Fail to load overall modelfile: %s" % LOAD_MODEL_FILE)

            ops = {'input_pls': input_pls,
                    'is_training_pl': is_training_pl,
                    'step': batch,
                    'end_points': end_points}

            is_training = False
            batch_data = img_arr
            k = np.zeros([1,3,3], np.float32)

            feed_dict = {ops['is_training_pl']: is_training,
                        ops['input_pls']['imgs']: batch_data,
                        ops['input_pls']['K']: k}
            
            # 0 pred_trans 1 pred_RT
            output_list = [ops['end_points']['pred_trans_mat'], 
                            ops['end_points']['pred_RT']]

            output = sess.run(output_list, feed_dict=feed_dict)
            pred_trans_mat = output[0][0]
            pred_RT = output[1][0]
            pred_RT = np.transpose(pred_RT)
            R = pred_RT[0:3,0:3]
            T = pred_RT[:,3][0:3]
            location = np.dot(-(np.transpose(R)),T)
            
            return location

def bright_pixel_mask(image, percentile=80):
    image = rgb2gray(image)
    perc = np.percentile(np.unique(image), percentile)
    mask = image < perc
    return mask

def read_img_get_transmat():
    root_dir = '/home/code/TMT/src/material_transfer/exemplar/wild_photo/target/'
    cam_dir = '/home/code/TMT/src/material_transfer/exemplar/wild_photo/campose/'
    
    if not os.path.exists(cam_dir):
        os.makedirs(cam_dir) 
    file_list = os.listdir(root_dir)
    file_list.sort()
    
    pbar = tqdm(file_list)
    pbar.set_description('estimating cam paras...')
    for file in pbar:
        # if '.jpg' in file:
        if '.jpg' in file:
            if os.path.exists(cam_dir + file.replace('.jpg','.json')):
                continue
            # back and fore ground.
            jpg_file = skimage.io.imread(root_dir + file)
            jpg_file = skimage.img_as_float32(jpg_file)
            jpg_mask = bright_pixel_mask(jpg_file, percentile=95)
           
            img_file = root_dir + file
            img_arr = cv2.imread(img_file, cv2.IMREAD_UNCHANGED).astype(np.uint8)[:, :, :3]
            img_arr[jpg_mask == 0] = (0, 0, 0)
            dest_img = img_arr

            # # if jpg file is not a RGBA format, then trans to RGBA
            # if not img_arr.shape[2] > 3:
            #     dest_img = np.append(img_arr, np.full((500, 500, 1), 255), axis=2)
            #     dest_img[jpg_mask == 0] = (0,0,0,0)
            #     dest_img = dest_img.astype(np.uint8)
            # else:
            #     img_arr[jpg_mask == 0] = (0, 0, 0, 0)
            #     dest_img = img_arr

            batch_img = np.asarray([dest_img.astype(np.float32) / 255.])
            batch_data = {}
            batch_data['img'] = batch_img
            FLAGS.cam_est = True
            if FLAGS.cam_est:
                print("here we use our cam est network to estimate cam parameters:")
                # with open(root_dir + file.replace('jpg','json')) as f1:
                    # camera_gt = json.load(f1)['camera']
                # camera_loc_gt = sp2co(camera_gt)
                camera_loc_est = cam_evl(batch_img)        
                json_file = {}
                # json_file['cam_loc_gt'] = camera_loc_gt
                json_file['cam_loc_est'] = camera_loc_est.tolist()
                with open(cam_dir + file.replace('.jpg','.json'), 'w') as f:
                    json.dump(json_file, f, indent=2)


if __name__ == "__main__":
    read_img_get_transmat()