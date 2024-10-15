import re
from time import time
import os
from os.path import dirname, abspath
import sys
sys.path.append(dirname(abspath(dirname(__file__))))

from time import time
import pprint

import torch
import torch.nn as nn

from lib import dataset
from lib.camera.camera import *
from lib.models.potr3d import *
from lib.models.loss import *
from lib.utils.utils import *
from lib.config import parse_args, config



if __name__=='__main__':
    args = parse_args()
    
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'test', args.cur_path)

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    gpus = [i for i in range(args.gpus)]


    test_dataset = eval('dataset.' + config.TEST.DATASET.NAME)(config, make_chunk=config.TEST.MAKE_CHUNK, mode='test')
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE if not config.TEST.MAKE_CHUNK else 1,
        shuffle=False,
        num_workers=config.NUM_WORKERS_PER_GPU*len(gpus),
        pin_memory=True
    )

    test_num_batches = len(test_dataloader) // config.TEST.BATCH_SIZE


    # Declare model and resume checkpoint file if necessary
    model_pos = POTR3D(config)
    
    model_params = 0
    for parameter in model_pos.parameters():
        model_params += parameter.numel()
    logger.info('INFO: Trainable parameter count: {:,d}'.format(model_params))

    model_pos = nn.DataParallel(model_pos)

    if torch.cuda.is_available():
        model_pos_pos = model_pos.cuda()    
    if config.RESUME_FILE:
        chk_filename = config.RESUME_FILE
        logger.info(f'Loading checkpoint : {chk_filename}')
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model_pos.load_state_dict(checkpoint['model_pos'], strict=False)

    
    receptive_field = config.MODEL.RECEPTIVE_FIELD


    ## Start Evaluation
    prints_per_epoch_test = config.TEST.PRINTS_PER_EPOCH
    with torch.no_grad():
        start_time = time()
        
        model_pos.eval()

        preds = []
        db_preds = {}
        db_preds['train'] = []
        db_preds['test'] = []

        logger.info(f'Evaluation Starts...')
        # Evaluate on test set
        for i, (clip_id, cameras, target_3d, inputs_2d, target_vis, img_paths) in enumerate(test_dataloader):
            if config.TEST.SAVE_RESULT:
                db_preds_i = {}
                db_preds_i['id'] = tuple([clip_id[0][0], clip_id[1].item(), clip_id[2].item()])
                db_preds_i['positions_3d'] = target_3d[0].clone().detach().cpu().numpy()

                kp2d = inputs_2d[0].clone().detach().cpu().numpy()
                valid_idx = kp2d[0,:,0,0] != 0
                kp2d_w_obj = kp2d[:,valid_idx]
                valid_kps_idx = kp2d_w_obj[...,0] != config.TEST.DATASET.INVALID_VALUE
                kp2d_w_obj[valid_kps_idx] = image_coordinates(kp2d_w_obj[valid_kps_idx], w=cameras[0][-3].item(), h=cameras[0][-2].item())
                kp2d[:,valid_idx] = kp2d_w_obj
                db_preds_i['positions_2d'] = kp2d

                db_preds_i['img_path'] = [x[0] for x in img_paths]                
            inputs_2d, offset = test_dataset.eval_data_prepare(receptive_field, inputs_2d)
            cameras = cameras.repeat(len(inputs_2d), 1)

            if torch.cuda.is_available():
                cameras = cameras.cuda()
                inputs_2d = inputs_2d.cuda()


            B, F, N, K, _ = inputs_2d.shape

            preds_3d = []
            for j in range(0, B, config.TEST.BATCH_SIZE):
                preds_3d_j = model_pos(inputs_2d[j:(j+config.TEST.BATCH_SIZE)])
                preds_3d.append(preds_3d_j)

            preds_3d = torch.cat(preds_3d, dim=0)
            preds_3d = convert_to_abs_camcoord(preds_3d, config.TEST.DATASET.ROOTIDX, cameras)
            preds_3d = preds_3d.reshape(-1, N, K, 3)
            preds_3d = preds_3d[:len(preds_3d)-offset]

            preds.append(preds_3d.detach().cpu().numpy())
            if config.TEST.SAVE_RESULT:
                db_preds_i['preds_3d'] = preds_3d.detach().cpu().numpy()
                db_preds['test'].append(db_preds_i)

            del inputs_2d

            if (i+1) % ((test_num_batches + prints_per_epoch_test - 1) // prints_per_epoch_test) == 0:
                logger.info('\t\t{:d}% Completed...!'.format(int((i+1)/test_num_batches * 100)))
        

        elapsed = (time() - start_time) / 60
        
        metrics = test_dataset.evaluate(preds)
        logger.info(f'\t[Finished] ({elapsed:.1f} MIN Elapsed) Evaluation Results')
        for k, v in metrics.items():
            logger.info(f'\t\t{k}: {v:.1f} ' + ('mm' if re.match('MPJPE|MPJVE', k) else '%'))


        if config.TEST.SAVE_RESULT:
            np.savez_compressed(os.path.join(config.TEST.DATASET.DATASET_ROOT, f'data_{config.TEST.DATASET.NAME}_preds.npz'), db=db_preds)