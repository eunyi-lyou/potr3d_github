import re
import os
from os.path import dirname, abspath
import sys
sys.path.append(dirname(abspath(dirname(__file__))))
from time import time
import pprint

import torch
import torch.nn as nn
import torch.optim as optim

from lib import dataset
from lib.camera.camera import *
from lib.models.potr3d import *
from lib.models.loss import *
from lib.utils.eval_utils import AverageMeter
from lib.utils.utils import *
from lib.config import parse_args, config



if __name__=='__main__':
    args = parse_args()
    
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train', args.cur_path)

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    gpus = [i for i in range(args.gpus)]


    # Create dataset & dataloader
    train_dataset = eval('dataset.' + config.TRAIN.DATASET.NAME)(config, mode='train')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.NUM_WORKERS_PER_GPU*len(gpus),
        pin_memory=True
    )

    test_dataset = eval('dataset.' + config.TEST.DATASET.NAME)(config, make_chunk=config.TEST.MAKE_CHUNK, mode='test')
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE if not config.TEST.MAKE_CHUNK else 1,
        shuffle=False,
        num_workers=config.NUM_WORKERS_PER_GPU*len(gpus),
        pin_memory=True
    )

    train_num_batches = len(train_dataset) // config.TRAIN.BATCH_SIZE
    test_num_batches = len(test_dataset) // config.TEST.BATCH_SIZE


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

    
    # Declare training parameters
    min_loss = 100000

    lr = config.TRAIN.LR
    lr_decay = config.TRAIN.LRD
    optimizer = optim.AdamW(model_pos.parameters(), lr=lr, weight_decay=0.1)

    losses_mpjpe_train = AverageMeter()
    losses_mpjve_train = AverageMeter()

    epoch = 0
    num_epochs = config.TRAIN.EPOCHS
    lambda_mpjve = 1.0

    receptive_field = config.MODEL.RECEPTIVE_FIELD

    if config.RESUME_FILE:
        epoch = checkpoint['epoch']
        lr = checkpoint['lr']
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            logger.info('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')



    ## Training
    logger.info('** Note: reported losses are averaged over the dataset.')
    prints_per_epoch_train = config.TRAIN.PRINTS_PER_EPOCH
    prints_per_epoch_test = config.TEST.PRINTS_PER_EPOCH
    while epoch < num_epochs:
        start_time = time()

        model_pos.train()
        logger.info(f'[Epoch {epoch+1}/{num_epochs}]')
        logger.info(f'\t=>TRAINING...')

        for i, (clip_id, cameras, target_3d, inputs_2d, target_vis, img_paths) in enumerate(train_dataloader):
            target_vis = target_vis.bool()
            target_obj = torch.cat([target_3d[b][0,:,0,0] != 0 for b in range(len(target_3d))], dim=0)

            if torch.cuda.is_available():
                cameras = cameras.cuda()
                target_3d = target_3d.cuda()
                inputs_2d = inputs_2d.cuda()
                target_vis = target_vis.cuda()
                target_obj = target_obj.cuda()
            
            optimizer.zero_grad()

            # Predict 3D poses
            preds_3d = model_pos(inputs_2d)
            preds_3d = convert_to_abs_camcoord(preds_3d, config.TRAIN.DATASET.ROOTIDX, cameras)
            
            del inputs_2d
            torch.cuda.empty_cache()


            # Loss Calculation
            loss_mpjpe, cnt_mpjpe = L2_loss(preds_3d, target_3d, target_vis, target_obj, root_id=config.TRAIN.DATASET.ROOTIDX, root_weight=0.3, include_invisible=True, mode='mpjpe')
            loss_mpjve, cnt_mpjve = L2_loss(preds_3d, target_3d, target_vis, target_obj, root_id=config.TRAIN.DATASET.ROOTIDX, root_weight=0.3, include_invisible=True, mode='mpjve')

            loss_total = loss_mpjpe + lambda_mpjve * loss_mpjve
            loss_total.backward()
            optimizer.step()


            losses_mpjpe_train.update(loss_mpjpe.item(), cnt_mpjpe.item())
            losses_mpjve_train.update(loss_mpjve.item(), cnt_mpjve.item())
            
            if (i+1) % ((train_num_batches + prints_per_epoch_train - 1) // prints_per_epoch_train) == 0:
                loss_mpjpe_avg = losses_mpjpe_train.avg
                loss_mpjve_avg = losses_mpjve_train.avg
                loss_tot_avg = loss_mpjpe_avg + lambda_mpjve * loss_mpjve_avg

                logger.info('\t\t[{:d}%] Total Loss: {:.5f}m / MPJPE Loss: {:.5f}m / MPJVE Loss: {:.5f}m'.format(int((i+1)/train_num_batches * 100), loss_tot_avg, loss_mpjpe_avg, loss_mpjve_avg))

            del preds_3d, target_3d, target_vis, target_obj, cameras, loss_mpjpe, cnt_mpjpe, loss_mpjve, cnt_mpjve, loss_total
            torch.cuda.empty_cache()


        # End-of-epoch evaluation
        if not config.TRAIN.NO_EVAL:
            with torch.no_grad():
                model_pos.eval()

                preds = []
                offsets = []

                logger.info(f'\t=>VALIDATING (w TEST data)')
                # Evaluate on test set
                for i, (clip_id, cameras, target_3d, inputs_2d, target_vis, img_paths) in enumerate(test_dataloader):
                    inputs_2d, offset = test_dataset.eval_data_prepare(receptive_field, inputs_2d, without_skip=False)
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
                    offsets.append(offset)

                    del inputs_2d

                    # if (i+1) % ((test_num_batches + prints_per_epoch_test - 1) // prints_per_epoch_test) == 0:
                    #     logger.info('\t\t{:d}% Completed...!'.format(int((i+1)/test_num_batches * 100)))
                
                metrics = test_dataset.evaluate(preds, offsets)
                val_loss = metrics[list(filter(lambda x: 'MPJPE_abs' in x, metrics.keys()))[0]]
                logger.info(f'\t\t=> Evaluation Results')
                for k, v in metrics.items():
                    logger.info(f'\t\t\t{k}: {v:.1f} ' + ('mm' if re.match('MPJPE|MPJVE', k) else '%'))


        # End-of-epoch report
        elapsed = (time() - start_time) / 60

        if config.TRAIN.NO_EVAL:
            logger.info('\t[Finished] ({:.1f} MIN Elapsed) lr {:.6f} MPJPE_train(mm) {:.1f}'.format(
                elapsed,
                lr,
                losses_mpjpe_train.avg * 1000))
        else:
            logger.info('\t[Finished] ({:.1f} MIN Elapsed) lr {:.6f} MPJPE_train(mm) {:.1f} MPJPE_valid(mm) {:.1f}'.format(
                elapsed,
                lr,
                losses_mpjpe_train.avg * 1000,
                val_loss))

        
        ### Save checkpoint if necessary
        if (epoch+1) % config.CKPT_FREQ == 0:
            chk_path = os.path.join(final_output_dir, f'epoch_{epoch+1}_{config.MODEL.NAME.upper()}_{config.TRAIN.DATASET.NAME.upper()}_{config.TRAIN.DATASET.TYPE.upper()}_S{config.TRAIN.DATASET.STRIDE}.bin')
            logger.info(f'\tSaving checkpoint to {chk_path}')

            torch.save({
                'epoch': epoch+1,
                'lr': lr,
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos.state_dict(),
            }, chk_path)

        # save best checkpoint
        best_chk_path = os.path.join(final_output_dir, f'best_{config.MODEL.NAME.upper()}_{config.TRAIN.DATASET.NAME.upper()}_{config.TRAIN.DATASET.TYPE.upper()}_S{config.TRAIN.DATASET.STRIDE}.bin')
        if val_loss < min_loss:
            min_loss = val_loss
            logger.info("\tSaving best checkpoint so far")
            torch.save({
                'epoch': epoch+1,
                'lr': lr,
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos.state_dict(),
            }, best_chk_path)


        ### Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1
        

        losses_mpjpe_train.reset()
        losses_mpjve_train.reset()
