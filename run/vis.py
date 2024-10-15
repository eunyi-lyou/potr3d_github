import re
from time import time
import os
from os.path import dirname, abspath
import sys
sys.path.append(dirname(abspath(dirname(__file__))))

from lib import dataset
from lib.utils.utils import *
from lib.config import parse_args, config



if __name__=='__main__':
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'vis', args.cur_path, logger_only=True)


    skeleton = eval('dataset.' + config.VIS.DATASET.NAME + '.skeleton')

    dataset_name = config.VIS.DATASET.NAME
    dataset_type = config.VIS.DATASET.TYPE + (f'_{config.VIS.DATASET.SUBTYPE}' if config.VIS.DATASET.TYPE=='synth' else '')
    dataset_subset = config.VIS.DATASET.SUBSET
    dataset_root = config.VIS.DATASET.DATASET_ROOT

    
    seq_id = None
    if args.vis_target is not None:
        seq_id = args.vis_target.split(',')
        seq_id = tuple(seq_id[:1] + list(map(int, seq_id[1:])))

    dataset.jointsdataset.visualization(config.VIS, \
        dataset_name, dataset_type, dataset_subset, dataset_root, \
        skeleton, mode=args.vis_mode, seq_id=seq_id)