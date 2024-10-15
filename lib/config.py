import os
import yaml
import argparse

from easydict import EasyDict as edict



config = edict()

### General
config.EXPERIMENT_NAME = ''
config.LOG_DIR = 'log'
config.TENSORBOARD_DIR = 'tb'
config.CKPT_FREQ = 40   ### create a checkpoint every {CKPT_FREQ} epochs
config.RESUME_FILE = ''  ### checkpoint to resume (file name)
config.EXPORT_TRAINING_CURVES = True 
config.NUM_WORKERS_PER_GPU = 8

### Params of Model
config.MODEL = edict()
config.MODEL.NAME = 'POTR3D_B'
config.MODEL.MAX_NUM_PEOPLE = 10
config.MODEL.RECEPTIVE_FIELD = 81
config.MODEL.NUM_JOINTS = 17
# Params of Lifting module
config.MODEL.NUM_LAYERS = 8
config.MODEL.TOKEN_DIM = 512
config.MODEL.NUM_HEADS = 8
config.MODEL.MLP_RATIO = 2.
config.MODEL.QKV_BIAS = True
config.MODEL.QK_SCALE = None
config.MODEL.DROP_RATE = 0.
config.MODEL.ATTN_DROP_RATE = 0.
config.MODEL.DROP_PATH_RATE = 0.2
config.MODEL.NORM_LAYER = None

### Params of Training
config.TRAIN = edict()
config.TRAIN.EPOCHS = 200
config.TRAIN.BATCH_SIZE = 16
config.TRAIN.DROPOUT = 0
config.TRAIN.SHUFFLE = True
config.TRAIN.LR = 1e-4
config.TRAIN.LRD = 0.99 ## LR Decaying Rate
config.TRAIN.FLIP_AUGMENTATION = True
config.TRAIN.PRINTS_PER_EPOCH = 10
config.TRAIN.NO_EVAL = False

config.TRAIN.DATASET = edict()
config.TRAIN.DATASET.NAME = 'vid3dhp'
config.TRAIN.DATASET.TYPE = 'synth'
config.TRAIN.DATASET.SUBTYPE = 'aug1'
config.TRAIN.DATASET.SUBSET = 'train'
config.TRAIN.DATASET.DATASET_ROOT = './data/vid3dhp'
config.TRAIN.DATASET.DATASET_DIRNAME = ''
config.TRAIN.DATASET.CAM_LIST = None
config.TRAIN.DATASET.SEQ_LIST = None
config.TRAIN.DATASET.MIN_CLIP_LEN = 1000
config.TRAIN.DATASET.NUM_JOINTS = 17
config.TRAIN.DATASET.ROOTIDX = 14
config.TRAIN.DATASET.STRIDE = 1
config.TRAIN.DATASET.FPS = 30.
config.TRAIN.DATASET.INVALID_VALUE = -1.

### Params of Evaluation
config.TEST = edict()
config.TEST.MAKE_CHUNK = False
config.TEST.BATCH_SIZE = 16
config.TEST.PRINTS_PER_EPOCH = 10
config.TEST.SAVE_RESULT = True

config.TEST.DATASET = edict()
config.TEST.DATASET.NAME = 'vid3dhp'
config.TEST.DATASET.TYPE = 'synth'
config.TEST.DATASET.SUBTYPE = 'aug1'
config.TEST.DATASET.SUBSET = 'test'
config.TEST.DATASET.DATASET_ROOT = './data/vid3dhp'
config.TEST.DATASET.DATASET_DIRNAME = ''
config.TEST.DATASET.TRACKED_FILE = ''
config.TEST.DATASET.TRACKED_CS_THRE = 0.4
config.TEST.DATASET.CAM_LIST = None
config.TEST.DATASET.SEQ_LIST = None
config.TEST.DATASET.MIN_CLIP_LEN = 1000
config.TEST.DATASET.NUM_JOINTS = 17
config.TEST.DATASET.ROOTIDX = 14
config.TEST.DATASET.STRIDE = 1
config.TEST.DATASET.FPS = 30.
config.TEST.DATASET.INVALID_VALUE = -1.

### Params of Visualization
config.VIS = edict()
config.VIS.VIS_DIR = 'vis'
config.VIS.VID_NUM_FRAMES = 300
config.VIS.DEFAULT_NUM_SHOTS = 4
config.VIS.FIG_SIZE = 6
config.VIS.ELEV = 5.
config.VIS.AZIM = -90.
config.VIS.RADIUS_OF_3D_PLOT = 3.
config.VIS.BITRATE = 3000
config.VIS.FPS = 30
config.VIS.PRINT_FREQ = 30

config.VIS.DATASET = edict()
config.VIS.DATASET.NAME = 'vid3dhp'
config.VIS.DATASET.TYPE = 'synth'
config.VIS.DATASET.SUBTYPE = 'aug1'
config.VIS.DATASET.SUBSET = 'train'
config.VIS.DATASET.DATASET_ROOT = './data/vid3dhp'
config.VIS.DATASET.DATASET_DIRNAME = ''
config.VIS.DATASET.ROOTIDX = 14
config.VIS.DATASET.INVALID_VALUE = -1

### Params of Data Augmentation
config.AUG = edict()
config.AUG.TOT_FRAMES = 3.36e6
config.AUG.FRAMES_PER_CLIP = 1000
config.AUG.TEST_SET_RATIO = 0.2
config.AUG.RADIUS_OF_3D_BALL = 140
config.AUG.MIN_VALID_DEPTH = 500
config.AUG.NUM_PEOPLE_CANDI = [5,4,3,2]
config.AUG.PT_LIM = 6000
config.AUG.PR_LIM = 0.25
config.AUG.GPT_CANDI = [0.]
config.AUG.GPR_CANDI = [0.]
config.AUG.GPR_MARGIN = 0.05
config.AUG.PRINT_FREQ = 1000
config.AUG.DEFAULT_NOISE_LEVEL = 0.1
config.AUG.DEFAULT_NOISE_PROB = 0.2
config.AUG.SELF_OCCLUSION_NOISE_LEVEL = 0.2
config.AUG.SELF_OCCLUSION_NOISE_PROB = 0.3
config.AUG.SELF_OCCLUSION_DROPOUT_PROB = 0.05
config.AUG.INTER_OCCLUSION_NOISE_LEVEL = 1.
config.AUG.INTER_OCCLUSION_NOISE_PROB = 1.
config.AUG.INTER_OCCLUSION_DROPOUT_PROB = 0.5
config.AUG.INVALID_VALUE = -1.



def _update_dict(cfg, k, v):
    for vk, vv in v.items():
        if vk in cfg[k]:
            if isinstance(vv, dict):
                _update_dict(cfg[k], vk, vv)
            else:
                cfg[k][vk] = vv if vv != 'None' else None
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))

def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(config, k, v)
                else:
                    config[k] = v if v != 'None' else None
            else:
                raise ValueError("{} not exist in config.py".format(k))


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument(
        '--gpus', help='number of gpus', default=2, type=int)
    parser.add_argument(
        '--cur_path', help='current path', default='./', type=str)
    parser.add_argument(
        '--vis-mode', help='vis mode (image / video)', default='image', type=str)
    parser.add_argument(
        '--vis-target', help='target sequecne "seq_id,cam_no,clip_no + frame_no" to visualize', default=None, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args