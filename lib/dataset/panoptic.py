import numpy as np
import copy
import time
import re
import os, glob
import logging
import json_tricks as json
from scipy.spatial.transform import Rotation

from lib.dataset.JointsDataset import JointsDataset
from lib.camera.camera import *
from lib.utils.augment_utils import *
from lib.utils.eval_utils import *

import warnings
warnings.filterwarnings("error")



JOINTS_DEF_PANOPTIC = {
    'neck': 0,
    'headtop': 1,  
    'pelvis': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14,
}

SKELETON_DEF_PANOPTIC = {
    'parents': [2, 0, -1, 0, 3, 4, 2, 6, 7, 0, 9, 10, 2, 12, 13],
    'joints_left': [3, 4, 5, 6, 7, 8],
    'joints_right': [9, 10, 11, 12, 13, 14],
    'symmetry': [[3,4,5,6,7,8], [9,10,11,12,13,14]]
}

TRAIN_LIST = [
    '160224_haggling1',
    '160226_mafia2',
    '160224_mafia1',
    '160224_mafia2',
    '160224_ultimatum1',
    '160224_ultimatum2'
]
VAL_LIST = [
    '160226_haggling1',
    '160422_haggling1', 
    '160226_mafia1', 
    '160422_ultimatum1',
    '160906_pizza1'
]

logger = logging.getLogger(__name__)


class PanopticDataset(JointsDataset):
    joints = JOINTS_DEF_PANOPTIC
    skeleton = SKELETON_DEF_PANOPTIC

    def __init__(self, cfg, make_chunk=True, mode='train'):
        super().__init__(cfg, mode)
        cfg_mode = eval(f'cfg.{mode.upper()}')
        
        self.joints = PanopticDataset.joints
        self.skeleton = PanopticDataset.skeleton
        assert len(self.joints) == self.num_joints, "Joints Definition Error"
        assert len(self.joints) == len(self.skeleton['parents']), "Joints Definition Error"

        self.train_list = TRAIN_LIST
        self.val_list = VAL_LIST
        self.seq_list = cfg_mode.DATASET.SEQ_LIST
        self.cam_list = cfg_mode.DATASET.CAM_LIST

        self.cams = self.get_cam(self.dataset_root, self.dataset_dirname, self.cam_list, self.seq_list)

        self.db_file = os.path.join(self.dataset_root, f"data_panoptic_{self.dataset_type + (f'_{cfg_mode.DATASET.SUBTYPE}' if cfg_mode.DATASET.TYPE=='synth' else '')}.npz")
        logger.info(f'=> Getting {self.dataset_type.upper()} db...')
        if os.path.isfile(self.db_file):
            self.db = np.load(self.db_file, allow_pickle=True)['db'].item()
            logger.info('=> Lazy Loading Completed...!')
        else:
            if self.dataset_type == 'gt':
                self.db = self._get_db()
            
            elif self.dataset_type == 'synth':
                db_file_gt = os.path.join(self.dataset_root, f'data_panoptic_gt.npz')
                if not os.path.isfile(db_file_gt):
                    logger.info(f'=> Getting GT db first...')
                    _ = self._get_db()
                self.db = self._make_augmented_dataset(cfg)
            
            elif self.dataset_type == 'tracked':
                assert mode=='test', 'Tracked dataset is only compatible with test mode'
                db_file_gt = os.path.join(self.dataset_root, f'data_panoptic_gt.npz')
                if not os.path.isfile(db_file_gt):
                    logger.info(f'=> Getting GT db first...')
                    _ = self._get_db()
                db_file_tracked = os.path.join(self.dataset_root, cfg_mode.DATASET.TRACKED_FILE)
                self.db = self._make_tracked_dataset(cfg_mode, db_file_gt, db_file_tracked, follow_tracked_keys=True)
        
        self.db = self.db[cfg_mode.DATASET.SUBSET]

        
        if self.mode == 'train' and self.stride > 1:
            self.db = self.db[::self.stride]
        
        if make_chunk:
            self.db = self._chunk_db()
        else:
            self._prepare_db()


    @staticmethod
    def get_cam(dataset_root, dataset_dirname, cam_list, seq_list):
        cams = {}
        for seq in seq_list:
            cams_seq = {}
            
            cam_file = os.path.join(dataset_root, dataset_dirname, seq, 'calibration_{:s}.json'.format(seq))
            with open(cam_file) as cfile:
                calib = json.load(cfile, ignore_comments=True)
            
            for cam in calib['cameras']:
                if [cam['panel'], cam['node']] in cam_list:
                    sel_cam = {}
                    sel_cam['id'] = '{:02d}_{:02d}'.format(cam['panel'], cam['node'])

                    sel_cam['res_w'] = cam['resolution'][0]
                    sel_cam['res_h'] = cam['resolution'][1]

                    ## Intrinsic Parameters
                    sel_cam['center'] = np.array([cam['K'][0][2], cam['K'][1][2]], dtype='float32')
                    sel_cam['focal_length'] = np.array([cam['K'][0][0], cam['K'][1][1]], dtype='float32')
                    sel_cam['radial_distortion'] = np.array([cam['distCoef'][0], cam['distCoef'][1], cam['distCoef'][4]], dtype='float32')
                    ## Order of distortion parameters(p1, p2) is reversed for them to be used in lib.camera.project_to_2d
                    sel_cam['tangential_distortion'] = np.array([cam['distCoef'][3], cam['distCoef'][2]], dtype='float32')
                    
                    sel_cam['normalization_factor'] = np.sqrt(np.prod(sel_cam['focal_length'])) / sel_cam['res_w'] * 2
                    sel_cam['intrinsic'] = np.concatenate((
                        sel_cam['focal_length'],
                        sel_cam['center'],
                        sel_cam['radial_distortion'],
                        sel_cam['tangential_distortion']
                    ))

                    ## Extrinsic Parameters (for coordinate change func. defined in lib.camera.camera)
                    sel_cam['orientation'] = Rotation.from_matrix(np.array(cam['R'])).as_quat()[...,[3,0,1,2]].tolist()
                    sel_cam['translation'] = (np.array(cam['t']) / 100.0).reshape(-1).tolist() ## cm to meters

                    cams_seq[(cam['panel'], cam['node'])] = sel_cam
            
            cams[seq] = [x[1] for x in sorted(cams_seq.items())]

        return cams

    def _get_db(self):
        logger.info(f"=> MAKE...")
        db = {}
        db['train'] = []
        db['test'] = []
        
        for seq in self.seq_list:
            logger.info(f'\tPROCESSING {seq}..')
            if seq in self.train_list:
                target = 'train'
            elif seq in self.val_list:
                target = 'test'
            curr_anno = os.path.join(self.dataset_root, self.dataset_dirname, seq, 'hdPose3d_stage1')
            anno_files = sorted(glob.iglob('{:s}/*.json'.format(curr_anno)))
            cameras = self.cams[seq]
            for c, v in enumerate(cameras):
                i = 0
                num_persons = 0
                cnt = 1

                db_temp_3d = []
                db_temp_2d = []
                db_temp_vis = []
                db_temp_img_path = []
                for n in range(len(anno_files)):
                    file = anno_files[n]
                    with open(file) as dfile:
                        try:
                            bodies = json.load(dfile, ignore_comments=True)['bodies']
                        except:
                            raise ValueError(f'Error Occured with file {file}')

                    postfix = os.path.basename(file).replace('body3DScene', '')
                    prefix = v['id']
                    image = os.path.join(self.dataset_root, self.dataset_dirname, seq, 'hdImgs', prefix, prefix + postfix)
                    image = image.replace('json', 'jpg')

                    all_poses_3d = []
                    all_poses = []
                    all_poses_vis = []
                    for body in bodies:
                        pose3d = np.array(body['joints15']).reshape((-1, 4))
                        pose3d = pose3d[:self.num_joints]

                        joints_vis = pose3d[:, -1] > 0.1
                        ## Coordinate change from world to cam defined in meter
                        pose3d_cam = world_to_camera(pose3d[:, :3] / 100.0, R=v['orientation'], t=v['translation'])

                        pose2d = np.zeros((self.num_joints, 2))
                        ## Distortion parameters are defined in cm
                        pose2d = project_to_2d(pose3d_cam[None, :, :3] * 100.0, v['intrinsic'][None, :])[0]
                        x_check = np.logical_and(pose2d[:, 0] >= 0, pose2d[:, 0] <= v['res_w'] - 1)
                        y_check = np.logical_and(pose2d[:, 1] >= 0, pose2d[:, 1] <= v['res_h'] - 1)
                        check = np.logical_and(x_check, y_check)
                        joints_vis[np.logical_not(check)] = 0

                        if joints_vis[self.root_idx]:
                            all_poses_3d.append(pose3d_cam)
                            all_poses.append(pose2d)
                            all_poses_vis.append(joints_vis)

                    if n == 0:
                        num_persons = len(all_poses_3d)
                        cnt = 1
                    else:
                        if num_persons == len(all_poses_3d):
                            cnt += 1
                        else:
                            if (cnt >= self.min_clip_len) and (len(db_temp_3d) > 0):
                                assert num_persons <= self.max_num_people, "Enlarge max_num_people argument ; num_person : {:d} / max_num_people : {:d}".format(num_persons, self.max_num_people)
                                
                                db_padded_3d = np.zeros((len(db_temp_3d), self.max_num_people, self.num_joints, 3), dtype='float32')
                                db_padded_2d = np.zeros((len(db_temp_2d), self.max_num_people, self.num_joints, 2), dtype='float32')
                                db_padded_vis = np.zeros((len(db_temp_3d), self.max_num_people, self.num_joints), dtype='bool')
                                
                                db_padded_3d[:,:num_persons] = np.asarray(db_temp_3d)[:,:num_persons]
                                db_padded_2d[:,:num_persons] = np.asarray(db_temp_2d)[:,:num_persons]
                                db_padded_vis[:,:num_persons] = np.asarray(db_temp_vis)[:,:num_persons]

                                 
                                db[target].append({
                                    'id': tuple([seq, c, i]),
                                    'positions_3d': db_padded_3d,
                                    'positions_2d': db_padded_2d,
                                    'vis': db_padded_vis,
                                    'img_path': db_temp_img_path,
                                    'cam': {
                                        'intrinsic': v['intrinsic'],
                                        'res_w': v['res_w'],
                                        'res_h': v['res_h'],
                                        'normalization_factor': v['normalization_factor']
                                    }
                                })
                                i += 1

                            db_temp_3d = []
                            db_temp_2d = []
                            db_temp_vis = []
                            db_temp_img_path = []
                            num_persons = len(all_poses_3d)
                            cnt = 1
                    
                    if (num_persons > 0):
                        db_temp_3d.append(all_poses_3d)
                        db_temp_2d.append(all_poses)
                        db_temp_vis.append(all_poses_vis)
                        db_temp_img_path.append(image)
                
                if (cnt >= self.min_clip_len) and (len(db_temp_3d) > 0):
                    assert num_persons <= self.max_num_people, "Enlarge max_num_people argument ; num_person : {:d} / max_num_people : {:d}".format(num_persons, self.max_num_people)
                                
                    db_padded_3d = np.zeros((len(db_temp_3d), self.max_num_people, self.num_joints, 3), dtype='float32')
                    db_padded_2d = np.zeros((len(db_temp_2d), self.max_num_people, self.num_joints, 2), dtype='float32')
                    db_padded_vis = np.zeros((len(db_temp_3d), self.max_num_people, self.num_joints), dtype='bool')
                    
                    db_padded_3d[:,:num_persons] = np.asarray(db_temp_3d)[:,:num_persons]
                    db_padded_2d[:,:num_persons] = np.asarray(db_temp_2d)[:,:num_persons]
                    db_padded_vis[:,:num_persons] = np.asarray(db_temp_vis)[:,:num_persons]

                    db[target].append({
                        'id':  tuple([seq, c, i]),
                        'positions_3d': db_padded_3d,
                        'positions_2d': db_padded_2d,
                        'vis': db_padded_vis,
                        'img_path': db_temp_img_path,
                        'cam': {
                            'intrinsic': v['intrinsic'],
                            'res_w': v['res_w'],
                            'res_h': v['res_h'],
                            'normalization_factor': v['normalization_factor']
                        }
                    })

        db['train'].sort(key=lambda x: x['id'])
        db['test'].sort(key=lambda x: x['id'])
        np.savez_compressed(self.db_file, db=db)
        logger.info(f"=> Make Completed...!")

        return db


    def _make_augmented_dataset(self, cfg):
        ## Data Loading
        data = np.load(os.path.join(self.dataset_root, f'data_panoptic_gt.npz'), allow_pickle=True)['db'].item()
        data = data['train'] + data['test']

        cams = self.cams
        num_cams = len(self.cam_list)
        num_seqs = len(data)
        clip_idxs_dict = {}
        for i in range(len(data)):
            k = data[i]['id'][:2]
            if k[0] in self.train_list:
                clip_idxs_dict.setdefault(k, [])
                clip_idxs_dict[k].append(i)
        tot_num_clips = sum([len(v) for k,v in clip_idxs_dict.items()])

        ## Fit Ground Planes for each camera
        logger.info(f"=>\tFitting Ground Plane...")
        geometry_path = os.path.join(self.dataset_root, 'panoptic_geometry.npz')
        if os.path.isfile(geometry_path):
            geometry = np.load(geometry_path, allow_pickle=True)
            normals = geometry['normals'].item()
            ground_planes = geometry['ground_planes'].item()
            logger.info(f"=>\tLazy Loading Completed...!")
        else:
            normals = {}
            ground_planes = {}

            for k, idxs in clip_idxs_dict.items():
                feets = np.concatenate([data[idx]['positions_3d'][:,:,[self.joints['l-ankle'], self.joints['r-ankle']]].reshape(-1,3) for idx in idxs], axis=0)
                ones = np.ones((feets.shape[0], 1))
                feets_c = np.concatenate((feets,ones), axis=-1).astype('float32')

                ground_plane, total_loss, fitting_loss, norm_loss = fit_ground_plane(feets_c, lr=1e-3, num_iter=10000)
            
                normal = ground_plane[:3] / np.linalg.norm(ground_plane[:3])
                ## z-value of normal vector should be negative (unless, it means camera is located below the ground plane)
                normal *= -np.sign(normal[-1])
                ground_plane *= (normal[0] / ground_plane[0])


                ## To Adjust Numerical computation error (When ||normal_y|| ~= 1)
                if np.sign(normal[1])==1:
                    normal[1] *= -1
                    ground_plane = np.concatenate([normal[:3], -ground_plane[-1:]])

                fitting_loss = abs(feets_c @ ground_plane / np.linalg.norm(ground_plane)).mean()

                logger.info(f'\t\t===== [{k[0]} - CAM {k[1]}] =====')
                logger.info(f'\t\t[Normal] {normal}')
                logger.info(f'\t\t[Fitting Loss]: {fitting_loss:.3f}')
                logger.info(f"\t\t===============")

                normals[k] = normal
                ground_planes[k] = ground_plane

            np.savez_compressed(os.path.join(self.dataset_root, 'panoptic_geometry.npz'), normals=normals, ground_planes=ground_planes)
            logger.info(f"=>\tCompleted...!")


        ## Augmentation
        logger.info('=> MAKE...')
        augtype = eval(f'cfg.{self.mode.upper()}').DATASET.SUBTYPE
        assert augtype in ('aug1', 'aug2', 'aug3', 'aug4'), f"Invalid augtype {augtype}"

        ### PT (Default)
        PT_lim = cfg.AUG.PT_LIM
        ### PR
        PR_lim = 0.
        ### GPT
        GPT_candi = [0.] ## (mm)
        ### GPR
        GPR_candi = [0.] ## (rad)

        if augtype != 'aug1':
            PR_lim = np.pi * cfg.AUG.PR_LIM
            if augtype != 'aug2':
                GPT_candi = cfg.AUG.GPR_CANDI ## (mm)
                if augtype != 'aug3':
                    GPR_candi = [np.pi * rot for rot in cfg.AUG.GPR_CANDI]

        tot_frames = cfg.AUG.TOT_FRAMES
        frames_per_seq = cfg.AUG.FRAMES_PER_CLIP
        tot_sequences = int(tot_frames / frames_per_seq)
        seq_per_comb = int(tot_sequences / len(GPT_candi) / len(GPR_candi) / num_cams / tot_num_clips)

        test_set_ratio = cfg.AUG.TEST_SET_RATIO
        test_seq_per_comb = int(seq_per_comb * test_set_ratio)
        train_seq_per_comb = seq_per_comb - test_seq_per_comb

        aug_print_freq = (cfg.AUG.PRINT_FREQ + frames_per_seq - 1) // frames_per_seq

        synth_db = {}
        synth_db['train'] = []
        synth_db['test'] = []
        
        for seq in self.train_list:
            logger.info(f'\tPROCESSING SEQ {seq}..')
            for c in range(num_cams):
                normal = normals[(seq,c)]
                ground_plane = ground_planes[(seq,c)]
                clip_idxs = clip_idxs_dict[(seq,c)]
                for i in clip_idxs:
                    clip = data[i]['positions_3d']
                    valid_idx = clip[0,:,0,0] != 0
                    clip = clip[:,valid_idx]
                    cam = data[i]['cam']
                    num_people = clip.shape[1]
                    for GPT_trans in GPT_candi:
                        for GPR_rot in GPR_candi:
                            s = 0
                            st = time.time()
                            while (s < seq_per_comb):
                                _kp3ds = [kp3d.squeeze(1) for kp3d in np.split(clip, num_people, axis=1)]
                                kp3ds = []
                                for kp3d in _kp3ds:
                                    ## (Warning) Potential Danger
                                    ## ; subjects too close to the camera might be affected by lens distortion
                                    ## we may use "compute_min_depths" func. (in lib.utils.augment_utils) later
                                    start_frame = np.random.choice(len(kp3d) - frames_per_seq + 1)
                                    kp3ds.append(kp3d[start_frame:(start_frame + frames_per_seq)])
                                kp3ds = np.stack(kp3ds, axis=1)
                                
                                success, meta = augmentation(kp3ds, cam, num_people, self.root_idx, ground_plane, normal, PT_lim, PR_lim, GPT_trans, GPR_rot, cfg)
                                if success:
                                    if s < train_seq_per_comb:
                                        target = 'train'
                                    else:
                                        target = 'test'
                                    
                                    db_padded_3d = np.zeros((len(meta['positions_3d']), self.max_num_people, self.num_joints, 3), dtype='float32')
                                    db_padded_2d = np.zeros((len(meta['positions_2d']), self.max_num_people, self.num_joints, 2), dtype='float32')
                                    db_padded_vis = np.zeros((len(meta['positions_3d']), self.max_num_people, self.num_joints), dtype='bool')
                                    
                                    db_padded_3d[:,:num_people] = np.asarray(meta['positions_3d'])[:,:num_people]
                                    db_padded_2d[:,:num_people] = np.asarray(meta['positions_2d'])[:,:num_people]
                                    db_padded_vis[:,:num_people] = np.asarray(meta['vis'])[:,:num_people]
                                    
                                    synth_db[target].append({
                                        'id': tuple([seq, c, s]),
                                        'positions_3d': db_padded_3d,
                                        'positions_2d': db_padded_2d,
                                        'vis': db_padded_vis,
                                        'img_path': [sorted(glob.glob(os.path.join(self.dataset_root, 'panoptic-toolbox', f'{seq}/hdImgs/{self.cam_list[c][0]:02d}_{self.cam_list[c][1]:02d}/*.jpg')))[0]] * frames_per_seq,
                                        'cam': {
                                            'intrinsic': cam['intrinsic'],
                                            'res_w': cam['res_w'],
                                            'res_h': cam['res_h'],
                                            'normalization_factor': cam['normalization_factor']
                                        }
                                    })

                                    if ((s+1) * frames_per_seq) % aug_print_freq == 0:
                                        et = time.time()
                                        logger.info(f"\t\t\tN: {num_people} / GPT: {GPT_trans} / GPR: {GPR_rot/np.pi:.2f}PI / cam: {c} --- NUM PERSONS: {num_people} --- NUM INTER OCCLUDED: {meta['num_inter_occluded']:d} / NUM SELF OCCLUDED: {meta['num_self_occluded']:d} / NUM INVISIBLE: {meta['num_invisible']:d}")
                                        logger.info(f"\t\t\t[GPR-eff] {meta['GPR-eff']/np.pi:.3f} / [V] {meta['PT_trans_v']:.3f} ({meta['PT_trans_v_val_range'][0]:.3f}~{meta['PT_trans_v_val_range'][1]:.3f}) / [U] {meta['PT_trans_u']:.3f} ({meta['PT_trans_u_val_range'][0]:.3f}~{meta['PT_trans_u_val_range'][1]:.3f})")
                                        logger.info(f"\t\t\tTIME SPENT: {et-st:.5f}SEC")
                                        st = time.time()

                                    s += 1

        np.savez_compressed(self.db_file, db=synth_db)
        logger.info('=> MAKE COMPLETED...!')
        return synth_db


    def evaluate(self, preds, mpjpe_threshold=0.500, seq_wise=['haggling','mafia','ultimatum','pizza']):
        assert len(preds)==len(self.db)  ### (B x [F, N, K, C])

        def _evaluate(i, total_gt):
            pred, _ = self.eval_data_prepare(self.receptive_field, preds[i], without_skip=False)

            target_clip = self.db[i]
            seq_id = target_clip['id']
            seq = seq_id[0]
            inputs_2d, _ = self.eval_data_prepare(self.receptive_field, target_clip['positions_2d'], without_skip=False)
            inputs_obj = inputs_2d[0,0,:,0,0] != 0
            target_3d, _ = self.eval_data_prepare(self.receptive_field, target_clip['positions_3d'], without_skip=False)
            target_vis, _ = self.eval_data_prepare(self.receptive_field, target_clip['vis'], without_skip=False)
            target_obj = target_3d[0,0,:,0,0] != 0

            eval_list, total_gt = evaluate_metrics(pred, inputs_obj, target_3d, target_vis, target_obj, total_gt=total_gt, root_id=self.root_idx, seq_id=seq)

            return eval_list, total_gt

        if seq_wise is None:
            eval_list = []
            total_gt = 0
            
            for i in range(len(self.db)):
                _eval_list, total_gt = _evaluate(i, total_gt)
                eval_list.extend(_eval_list)

            metrics = eval_list_to_metrics(eval_list, total_gt)
        else:
            seqs = '|'.join(seq_wise)
            metrics = {}
            
            for i in range(len(self.db)):
                eval_list, total_gt = _evaluate(i, total_gt=0)
                _metrics = eval_list_to_metrics(eval_list, total_gt)

                seq = re.findall(seqs, self.db[i]['id'][0])[0]

                for k in _metrics:
                    metrics.setdefault(k, {})
                    metrics[k].setdefault(seq, [0, 0])
                    if metrics[k][seq][0] == 0:
                        metrics[k][seq][1] = _metrics[k]
                    else:
                        metrics[k][seq][1] = (metrics[k][seq][1] * metrics[k][seq][0] + _metrics[k]) / (metrics[k][seq][0] + 1)
                    metrics[k][seq][0] += 1
                
            metrics = dict([(k, np.mean(list(map(lambda x: x[1], metrics[k].values())))) for k in metrics])

        return metrics



            
   