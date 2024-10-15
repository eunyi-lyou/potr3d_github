import os
from time import time
import datetime
from pathlib import Path
import logging
import numpy as np
import torch
from torch.utils.data import Dataset

from lib.camera.camera import normalize_screen_coordinates
from lib.utils.vis_utils import *


logger = logging.getLogger(__name__)

class JointsDataset(Dataset):
    def __init__(self, cfg, mode):
        cfg_mode = eval(f'cfg.{mode.upper()}')
        if mode=='test':
            assert cfg_mode.MAKE_CHUNK or cfg_mode.BATCH_SIZE == 1, "Batch size should be 1 when the chunking is required"
            assert cfg_mode.MAKE_CHUNK != cfg_mode.SAVE_RESULT, "Chunking is not allowed to save the result in the order of original data" 

        self.fps = cfg_mode.DATASET.FPS
        self.min_clip_len = cfg_mode.DATASET.MIN_CLIP_LEN
        self.num_joints = cfg_mode.DATASET.NUM_JOINTS
        self.root_idx = cfg_mode.DATASET.ROOTIDX
        self.invalid_value = cfg_mode.DATASET.INVALID_VALUE
        self.mode = mode
        self.dataset_root = cfg_mode.DATASET.DATASET_ROOT
        self.dataset_dirname = cfg_mode.DATASET.DATASET_DIRNAME
        self.dataset_name = cfg_mode.DATASET.NAME
        self.dataset_type = cfg_mode.DATASET.TYPE
        assert self.dataset_type in ('gt', 'synth', 'tracked'), f"Invalid dataset type option, {self.dataset_type}"
        
        self.max_num_people = cfg.MODEL.MAX_NUM_PEOPLE
        self.receptive_field = cfg.MODEL.RECEPTIVE_FIELD
        self.stride = cfg_mode.DATASET.STRIDE
        self.flip = mode=='train' and cfg.TRAIN.FLIP_AUGMENTATION


    ## Return number of total clips (whose length = receptive field)
    def __len__(self):
        return len(self.db)
    
    def __getitem__(self, idx):
        db_item = self.db[idx]
        clip_id = db_item['id']
        camera = db_item['cam']
        inputs_3d = db_item['positions_3d']
        inputs_2d = db_item['positions_2d']
        inputs_vis = db_item['vis']
        inputs_img_path = db_item['img_path']

        return clip_id, camera, inputs_3d, inputs_2d, inputs_vis, inputs_img_path
        

    def _chunk_db(self):
        chunked_db = []

        pairs = []
        for i, clip in enumerate(self.db):
            clip_len = clip['positions_3d'].shape[0]
            if self.mode == 'train':
                n_chunks = (clip_len + self.receptive_field - 1) // self.receptive_field
            else:
                ## in 'eval' or 'vis' mode, do not extrapolate
                n_chunks = clip_len // self.receptive_field
            offset = (n_chunks * self.receptive_field - clip_len) // 2
            bounds = np.arange(n_chunks + 1) * self.receptive_field - offset
            if_flip = np.full(len(bounds)-1, False, dtype=bool)
            pairs += zip(np.repeat(i, len(bounds)-1), bounds[:-1], bounds[1:], if_flip)
            if self.flip:
                pairs += zip(np.repeat(i, len(bounds)-1), bounds[:-1], bounds[1:], ~if_flip)

        for pair in pairs:
            seq_i, start_idx, end_idx, if_flip = pair

            seq_data = self.db[seq_i]

            low_idx = max(start_idx, 0)
            high_idx = min(end_idx, seq_data['positions_3d'].shape[0])
            pad_left = low_idx - start_idx
            pad_right = end_idx - high_idx

            seq_id = seq_data['id']
            seq_3d = seq_data['positions_3d'][low_idx:high_idx].copy()
            seq_2d = seq_data['positions_2d'][low_idx:high_idx].copy()
            seq_vis = seq_data['vis'][low_idx:high_idx]
            seq_img_path = seq_data['img_path'][low_idx:high_idx]
            seq_cam = seq_data['cam']
            ## Debug ##
            if isinstance(seq_cam['normalization_factor'], list) or isinstance(seq_cam['normalization_factor'], np.ndarray):
                seq_cam['normalization_factor'] = seq_cam['normalization_factor'][0]

            chunked_seq_3d = np.pad(seq_3d, ((pad_left, pad_right), (0,0), (0,0), (0,0)), 'edge')
            chunked_seq_2d = np.pad(seq_2d, ((pad_left, pad_right), (0,0), (0,0), (0,0)), 'edge')
            chunked_seq_vis = np.pad(seq_vis, ((pad_left, pad_right), (0,0), (0,0)), 'edge')
            chunked_seq_img_path = [seq_img_path[0]] * pad_left + seq_img_path + [seq_img_path[-1]] * pad_right
            # np.pad(seq_img_path, ((pad_left, pad_right)), 'edge')

            ## Normalize input 2D joints
            ## Except for Invalid joints(Valued as -1 while synthesis)
            valid_idx = seq_2d[0,:,0,0] != 0
            seq_2d_w_obj = seq_2d[:,valid_idx]
            valid_kps_idx = seq_2d_w_obj[...,0] != self.invalid_value
            seq_2d_w_obj[valid_kps_idx] = normalize_screen_coordinates(seq_2d_w_obj[valid_kps_idx], w=seq_cam['res_w'], h=seq_cam['res_h']) 
            seq_2d[:,valid_idx] = seq_2d_w_obj

            if if_flip:
                chunked_seq_3d[...,0] *= -1
                chunked_seq_3d[...,self.skeleton['joints_left'] + self.skeleton['joints_right'],:] = \
                    chunked_seq_3d[...,self.skeleton['joints_right'] + self.skeleton['joints_left'],:]
                chunked_seq_2d[...,0] *= -1
                chunked_seq_2d[...,self.skeleton['joints_left'] + self.skeleton['joints_right'],:] = \
                    chunked_seq_2d[...,self.skeleton['joints_right'] + self.skeleton['joints_left'],:]
                
                ## Flip horizontal center param
                seq_cam['intrinsic'][2] *= -1
                ## Flip horizontal distortion param if exists
                if len(seq_cam['intrinsic']) >= 8:
                    seq_cam['intrinsic'][8] *= -1 ## Tangential distortion paramters are reversed

            chunked_db.append({
                'id': seq_id,
                'positions_3d': chunked_seq_3d,
                'positions_2d': chunked_seq_2d,
                'vis': chunked_seq_vis,
                'img_path': chunked_seq_img_path,
                'cam': np.concatenate((
                    seq_cam['intrinsic'],
                    [seq_cam['res_w'], seq_cam['res_h'], seq_cam['normalization_factor']]
                ))
            })

        return chunked_db


    def _prepare_db(self):
        for seq_data in self.db:
            seq_2d = seq_data['positions_2d']
            seq_cam = seq_data['cam']
            ## Debug ##
            if isinstance(seq_cam['normalization_factor'], list) or isinstance(seq_cam['normalization_factor'], np.ndarray):
                seq_cam['normalization_factor'] = seq_cam['normalization_factor'][0]

            ## Normalize input 2D joints
            ## Except for Invalid joints(Valued as -1 while synthesis)
            valid_idx = seq_2d[0,:,0,0] != 0
            seq_2d_w_obj = seq_2d[:,valid_idx]
            valid_kps_idx = seq_2d_w_obj[...,0] != self.invalid_value
            seq_2d_w_obj[valid_kps_idx] = normalize_screen_coordinates(seq_2d_w_obj[valid_kps_idx], w=seq_cam['res_w'], h=seq_cam['res_h']) 
            seq_2d[:,valid_idx] = seq_2d_w_obj

            seq_data['cam'] = np.concatenate((
                seq_cam['intrinsic'],
                [seq_cam['res_w'], seq_cam['res_h'], seq_cam['normalization_factor']]
            ))
    
    
    def _make_tracked_dataset(self, cfg, db_file_gt, db_file_tracked, follow_tracked_keys=False):
        data_tracked = np.load(db_file_tracked, allow_pickle=True)['db'].item()['test']

        db_tracked = {}
        db_tracked['train'] = []
        db_tracked['test'] = []

        if cfg.DATASET.NAME != 'demo':
            db_gt = np.load(db_file_gt, allow_pickle=True)['db'].item()['test']
            for i in range(len(db_gt)):
                db_gt_i = db_gt[i].copy()

                flag = True
                for j in range(len(data_tracked)):
                    if db_gt_i['id'] == data_tracked[j]['id']:
                        ## CMU-Panoptic Dataset contains some choppy frames, which would be contagious to tracking
                        ## So we cut the videos until that choppy artifacts occur
                        if cfg.DATASET.NAME=='panoptic':
                            if db_gt_i['id'] == ('160422_haggling1',1,0):
                                valid_limit = 959
                            elif db_gt_i['id'] == ('160422_ultimatum1',0,2):
                                valid_limit = 1014
                            elif db_gt_i['id'] == ('160226_mafia1',1,2):
                                valid_limit = 1239
                            else:
                                valid_limit = len(db_gt_i['img_path'])
                        break
                    if j == len(data_tracked)-1:
                        flag = False
                        if not follow_tracked_keys:
                            raise ValueError(f'{db_file_gt}, {db_file_tracked} mismatched')
                        else:
                            continue
                if flag:
                    data = data_tracked[j]['positions_2d']
                    valid_idx = data[0,:,0,0] != 0
                    data_w_obj = data[:,valid_idx,:,:2]
                    data_w_obj_score = data[:,valid_idx,:,2:]
                    ibb_idx = (data_w_obj_score[...,0] >= cfg.DATASET.TRACKED_CS_THRE) * (data_w_obj[...,0] >= 0) * (data_w_obj[...,0] <= db_gt_i['cam']['res_w']-1) * (data_w_obj[...,1] >= 0) * (data_w_obj[...,1] <= db_gt_i['cam']['res_h']-1)
                    data_w_obj[np.logical_not(ibb_idx)] = cfg.DATASET.INVALID_VALUE
                    data[:,valid_idx,:,:2] = data_w_obj
                    
                    db_gt_i['positions_2d'] = data[...,:2][:valid_limit]
                    db_gt_i['positions_3d'] = db_gt_i['positions_3d'][:valid_limit]
                    db_gt_i['vis'] = db_gt_i['vis'][:valid_limit]
                    db_gt_i['obj'] = db_gt_i['obj'][:valid_limit]
                    db_gt_i['img_path'] = db_gt_i['img_path'][:valid_limit]
                    db_tracked['test'].append(db_gt_i)

        ## Paired GT data doesn't exist if dataset == 'demo'
        else:
            for i in range(len(data_tracked)):
                data = data_tracked[j]['positions_2d']
                valid_idx = data[0,:,0,0] != 0
                data_w_obj = data[:,valid_idx,:,:2]
                data_w_obj_score = data[:,valid_idx,:,2:]
                ibb_idx = (data_w_obj_score[...,0] >= cfg.DATASET.TRACKED_CS_THRE) * (data_w_obj[...,0] >= 0) * (data_w_obj[...,0] <= db_gt_i['cam']['res_w']-1) * (data_w_obj[...,1] >= 0) * (data_w_obj[...,1] <= db_gt_i['cam']['res_h']-1)
                data_w_obj[np.logical_not(ibb_idx)] = cfg.DATASET.INVALID_VALUE
                data[:,valid_idx,:,:2] = data_w_obj

                data['positions_2d'] = data[...,:2]
                db_tracked['test'].append(data)
            
        np.savez_compressed(self.db_file, db=db_tracked)

        return db_tracked
    
    
    @staticmethod
    def eval_data_prepare(receptive_field, unchunked_data, without_skip=True):
        if isinstance(unchunked_data, torch.Tensor):
            unchunked_data = torch.squeeze(unchunked_data, dim=0)
            if without_skip:
                out_num = (unchunked_data.shape[0] + receptive_field - 1) // receptive_field
                offset = (out_num * receptive_field) - unchunked_data.shape[0]
            else:
                out_num = unchunked_data.shape[0] // receptive_field
                offset = 0
            eval_data = torch.empty(out_num, receptive_field, *unchunked_data.shape[1:])
            for i in range(out_num):
                if i==(out_num-1):
                    padding_shape = (offset,) + (1,) * (len(unchunked_data.shape) - 1)
                    padding = unchunked_data[-1:].repeat(*padding_shape)
                    eval_data[i] = torch.cat((unchunked_data[i*receptive_field:(i+1)*receptive_field], padding), dim=0)
                else:
                    eval_data[i] = unchunked_data[i*receptive_field:(i+1)*receptive_field]
        elif isinstance(unchunked_data, np.ndarray):
            try:
                unchunked_data = np.squeeze(unchunked_data, dim=0)
            except:
                pass
            if without_skip:
                out_num = (unchunked_data.shape[0] + receptive_field - 1) // receptive_field
                offset = (out_num * receptive_field) - unchunked_data.shape[0]
            else:
                out_num = unchunked_data.shape[0] // receptive_field
                offset = 0
            eval_data = np.empty((out_num, receptive_field, *unchunked_data.shape[1:]))
            for i in range(out_num):
                if i==(out_num-1):
                    padding_shape = ((0,offset),) + ((0,0),) * (len(unchunked_data.shape) - 1)
                    eval_data[i] = np.pad(unchunked_data[i*receptive_field:(i+1)*receptive_field], padding_shape, 'edge')
                else:
                    eval_data[i] = unchunked_data[i*receptive_field:(i+1)*receptive_field]
        else:
            raise ValueError(f'invalid Data Type : {type(unchunked_data)}')
        
        return eval_data, offset
    
    
    @staticmethod
    def visualization(cfg, dataset_name, dataset_type, dataset_subset, dataset_root, skeleton, mode='image', seq_id=None):
        ## type: gt --> GT 2D + GT 3D
        ## type: synth --> synth 2D + synth 3D
        ## type: tracked --> tracked 2D + GT 3D
        ## type: pred (tbd) ; also can be used for debugging --> input 2D + pred 3D + (GT 3D if exists)

        assert mode in ('image', 'video'), "Incompatible mode"
        assert seq_id is None or len(seq_id) in (3, 4), "Incompatible seq_id input"

        
        vis_dir = Path(cfg.VIS_DIR).resolve() / dataset_name
        logger.info('=> creating {}'.format(vis_dir))
        vis_dir.mkdir(parents=True, exist_ok=True)

        tz = datetime.timezone(datetime.timedelta(hours=9)) ## KST
        time_str = str(datetime.datetime.now(tz))[:-16].replace(' ', '-').replace(':', '-')
        out_file = '{}_{}_{}.{}'.format(dataset_type, mode, time_str, 'jpg' if mode=='image' else 'mp4')
        final_out_file = str(vis_dir / out_file)
        logger.info('=> saving file as {}'.format(final_out_file))


        
        num_frames = cfg.VID_NUM_FRAMES if mode=='video' else 1 
        plots_per_shot = 2 ## Input 2D + Target 3D  (for Pred --> Input 2D + Recon 3D + (Target 3D if exists))
        if dataset_type.endswith('preds'):
            plots_per_shot = 3
        default_num_shots = cfg.DEFAULT_NUM_SHOTS

        limit = num_frames


        db_file = os.path.join(dataset_root, f'data_{dataset_name}_{dataset_type}.npz')
        assert os.path.isfile(db_file), f"Create {db_file} first"
        if dataset_name != 'vid3dhp':
            db_file_gt = os.path.join(dataset_root, f'data_{dataset_name}_gt.npz')
            assert os.path.isfile(db_file_gt), f"Create {db_file_gt} first"


        db = np.load(db_file, allow_pickle=True)['db'].item()
        db = db[dataset_subset]

    
        ## Select datas to display
        selects = []
        if seq_id is not None:
            for db_idx, data in enumerate(db):
                if data['id'] == seq_id[:3]:
                    if len(seq_id) == 3:  ## (seq, camno, clipno)
                        num_shots = default_num_shots if mode=='image' else 1
                        for _ in range(num_shots):
                            if len(data['img_path']) < num_frames:
                                limit = len(data['img_path'])
                                start_idx = 0
                                end_idx = limit
                            else:
                                start_idx = np.random.choice(len(data['img_path']) - num_frames + 1, 1)[0]
                                end_idx = start_idx + num_frames
                            selects.append((db_idx, start_idx, end_idx))
                    elif len(seq_id) == 4:  ## (seq, camno, clipno, frameno)
                        assert mode=='image'
                        selects.append([db_idx, seq_id[3], seq_id[3] + num_frames])
        else:
            db_idxs = np.random.choice(len(db), size=default_num_shots)
            for db_idx in db_idxs:
                if len(db[db_idx]['img_path']) < num_frames:
                    limit = len(db[db_idx]['img_path'])
                    start_idx = 0
                    end_idx = limit
                else:
                    start_idx = np.random.choice(len(db[db_idx]['img_path']) - num_frames + 1, 1)[0]
                    end_idx = start_idx + num_frames
                end_idx = start_idx + num_frames
                selects.append([db_idx, start_idx, end_idx])
        
        data_id, data_img, datas = [], [], []
        for db_idx, start_idx, end_idx in selects:
            end_idx = start_idx + limit - 1
            
            data_id.append(db[db_idx]['id'] + (start_idx, end_idx))
            data_img.append(db[db_idx]['img_path'][start_idx:end_idx])
            
            _datas = []
            pobj = db[db_idx]['positions_2d'][0,:,0,0] != 0
            tobj = db[db_idx]['positions_3d'][0,:,0,0] != 0
            _datas.append(db[db_idx]['positions_2d'][start_idx:end_idx, pobj])
            if dataset_type.endswith('preds'):
                _datas.append(db[db_idx]['preds_3d'][start_idx:end_idx, pobj])
            _datas.append(db[db_idx]['positions_3d'][start_idx:end_idx, tobj])            
            datas.append(_datas)


        ## Plot selected shots
        size = cfg.FIG_SIZE
        elev = cfg.ELEV
        azim = cfg.AZIM
        fps = cfg.FPS
        bitrate = cfg.BITRATE

        num_shots = len(data_img)

        plt.ioff()
        fig = plt.figure()
        fig = plt.figure(constrained_layout=True)
        fig.set_size_inches(size * plots_per_shot, size * num_shots)
        subfigs = fig.subfigures(nrows=num_shots, ncols=1)
        if not isinstance(subfigs, np.ndarray):
            subfigs = np.array([subfigs])
        # fig.tight_layout()

        fig.suptitle(f"(Left) Input 2D / {('(Mid) Recon 3D / ' if dataset_type.endswith('preds') else '') + '(Right) Target 3D'}\n", fontsize=20)

        initialized = False
        assets = []
        center = None
        def update_fig(f):
            nonlocal initialized, assets, center

            if not initialized:
                for i in range(num_shots):
                    subfigs[i].suptitle(f'{data_id[i][0]} / CAM {data_id[i][1]} / CLIP {data_id[i][2]} : FRAME {os.path.basename(data_img[i][0])} ~ {os.path.basename(data_img[i][-1])}', fontsize=15)
                    _assets = []
                    for j in range(plots_per_shot-1,-1,-1):
                        if j == 0:
                            ax = subfigs[i].add_subplot(1, plots_per_shot, j+1)
                            ax.get_xaxis().set_visible(False)
                            ax.get_yaxis().set_visible(False)
                            ax.set_axis_off()
                            _assets.insert(0, render_img(cfg, ax, datas[i][j][f], skeleton=skeleton, mode='2d', colormode='leftright', img_path=data_img[i][f]))
                        else:
                            ax = subfigs[i].add_subplot(1, plots_per_shot, j+1, projection='3d')
                            ax.view_init(elev=elev, azim=azim)
                            # ax.set_xticklabels([])
                            # ax.set_yticklabels([])
                            # ax.set_zticklabels([])
                            ax.set_xlabel('$X$')
                            ax.set_ylabel('$Z (Depth)$')
                            ax.set_zlabel('$Y$')

                            if j==(plots_per_shot-1):
                                center = datas[i][j][f][:,cfg.DATASET.ROOTIDX].mean(0)
                            _assets.insert(0, render_img(cfg, ax, datas[i][j][f], skeleton=skeleton, mode='3d', colormode='leftright', center=center))
                    assets.append(_assets)

                initialized = True
            else:
                for i in range(num_shots):
                    for j in range(plots_per_shot):
                        if j == 0:
                            update_img(cfg, assets[i][j], datas[i][j][f], skeleton=skeleton, mode='2d', colormode='leftright', img_path=data_img[i][f])
                        else:
                            if j==(plots_per_shot-1):
                                center = datas[i][j][f][:,cfg.DATASET.ROOTIDX].mean(0)
                            update_img(cfg, assets[i][j], datas[i][j][f], skeleton=skeleton, mode='3d', colormode='leftright', center=center)

            if (f+1) % cfg.PRINT_FREQ == 0:
                logger.info('\t{}/{} Completed...!'.format(f+1, limit))
        

        logger.info('=>VISUALIZING...')
        start_time = time()
        if mode == 'video':
            anim = FuncAnimation(fig, update_fig, frames=np.arange(0, limit), interval=1000 / fps, repeat=False)
            if final_out_file.endswith('.mp4'):
                Writer = writers['ffmpeg']
                writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
                anim.save(final_out_file, writer=writer)
            elif final_out_file.endswith('.gif'):
                anim.save(final_out_file, dpi=80, writer='imagemagick')
            else:
                raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
        else:
            update_fig(0)
            fig.savefig(final_out_file)
            
        elapsed = (time() - start_time) / 60
        logger.info(f'[Finished] {elapsed:.1f} MIN Elapsed')
        plt.close()
