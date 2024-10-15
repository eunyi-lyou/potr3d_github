# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
import hashlib
import logging
import os, shutil
import datetime
from pathlib import Path


def stable_sigmoid(x):
    sig_x = np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
    return sig_x

def log(log_out, write_str):
    with open(log_out, 'a') as f:
        f.write(str(write_str) + '\n')
    print(write_str)


def create_logger(cfg, cfg_path, mode='train', cur_path='./', logger_only=False):
    cfg_mode = eval(f'cfg.{mode.upper()}')
    cur_path = Path(cur_path)
    root_output_dir = (cur_path / cfg.LOG_DIR).resolve()  ##
    tensorboard_log_dir = (cur_path / cfg.TENSORBOARD_DIR).resolve()
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg_mode.DATASET.NAME + "_" + cfg_mode.DATASET.TYPE
    model = cfg.MODEL.NAME

    if cfg.EXPERIMENT_NAME != '':
        final_output_dir = root_output_dir / dataset / model
    else:
        final_output_dir = root_output_dir / dataset / model

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    tz = datetime.timezone(datetime.timedelta(hours=9)) ## KST
    time_str = str(datetime.datetime.now(tz))[:-16].replace(' ', '-').replace(':', '-')
    log_file = '{}_{}.log'.format(mode, time_str)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    if logger_only:
        tensorboard_log_dir = tensorboard_log_dir / dataset / model
        print('=> creating {}'.format(tensorboard_log_dir))
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(cfg_path, tensorboard_log_dir)

        tensorboard_log_dir = str(tensorboard_log_dir)

    return logger, str(final_output_dir), tensorboard_log_dir


def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """
    
    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)
        
    result = func(*args)
    
    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result
    
def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2**32 - 1) * (max_value - min_value)) + min_value

