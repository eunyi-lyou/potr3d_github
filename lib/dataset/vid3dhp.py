import numpy as np
import time
import logging
import os, glob
import json_tricks as json
import scipy.io as sio
from scipy.spatial.transform import Rotation

from lib.dataset.JointsDataset import JointsDataset
from lib.camera.camera import *
from lib.utils.augment_utils import *
from lib.utils.eval_utils import *



JOINTS_DEF_VID3DHP = {
    'neck': 1,
    'headtop': 0,
    'r-shoulder': 2,
    'r-elbow': 3,
    'r-wrist': 4,
    'l-shoulder': 5,
    'l-elbow': 6,
    'l-wrist': 7,
    'r-hip': 8,
    'r-knee': 9,
    'r-ankle': 10,
    'l-hip': 11,
    'l-knee': 12,
    'l-ankle': 13,
    'pelvis': 14,
    'spine': 15,
    'head': 16
}

SKELETON_DEF_VID3DHP = {
    'parents': [16, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, -1, 14, 1],
    'joints_left': [5, 6, 7, 11, 12, 13],
    'joints_right': [2, 3, 4, 8, 9, 10],
    'symmetry': [[5,6,7,11,12,13], [2,3,4,8,9,10]]
}

CAM_LIST = list(range(14))


logger = logging.getLogger(__name__)


class Vid3DHPDataset(JointsDataset):
    joints = JOINTS_DEF_VID3DHP
    skeleton = SKELETON_DEF_VID3DHP

    def __init__(self, cfg, make_chunk=True, mode='train'):
        super().__init__(cfg, mode)
        cfg_mode = eval(f'cfg.{mode.upper()}')

        assert self.dataset_type=='synth', f"Invalid dataset type option, {self.dataset_type}"

        self.joints = Vid3DHPDataset.joints
        self.skeleton = Vid3DHPDataset.skeleton
        assert len(self.joints) == self.num_joints, "Joints Definition Error"
        assert len(self.joints) == len(self.skeleton['parents']), "Joints Definition Error"

        self.cam_list = CAM_LIST

        self.cams = self.get_cam(self.dataset_root, self.cam_list)

        self.db_file = os.path.join(self.dataset_root, f"data_vid3dhp_{self.dataset_type + (f'_{cfg_mode.DATASET.SUBTYPE}' if cfg_mode.DATASET.TYPE=='synth' else '')}.npz")
        logger.info(f'=> Getting {self.dataset_type.upper()} db...')
        if os.path.isfile(self.db_file):
            self.db = np.load(self.db_file, allow_pickle=True)['db'].item()
            logger.info('=> Lazy Loading Completed...!')
        else:
            self.db = self._make_augmented_dataset(cfg)
        if self.mode == 'train' and self.stride > 1:
            self.db = self.db[::self.stride]

        self.db = self.db[cfg_mode.DATASET.SUBSET]
        
        if make_chunk:
            self.db = self._chunk_db()
        else:
            self._prepare_db()


    @staticmethod
    def get_cam(dataset_root, cam_list, seq_list=None):
        _cams = []
        ## All camera calibration files are the same
        with open(os.path.join(dataset_root, '3DHP', 'S1/Seq1/camera.calibration'), 'r') as f:   
            o = f.readline().strip()
            while o != '':
                _cams.append(o)
                o = f.readline().strip()
        _cams = _cams[1:]

        cams = {}
        for i in range(len(cam_list)):
            cam_i = _cams[7*i:7*(i+1)]
            sel_cam = {}

            sel_cam['res_w'], sel_cam['res_h'] = [int(x) for x in cam_i[2].split(' ')[-2:]]

            ## Intrinsic Parameters
            K = np.array([x for x in cam_i[4].split(' ')[-16:]], dtype='float32').reshape(4,4)[:3,:3]
            sel_cam['focal_length'] = np.array(K[[0,1],[0,1]], dtype='float32')
            sel_cam['center'] = np.array(K[[0,1],[2,2]], dtype='float32')
            
            sel_cam['normalization_factor'] = np.sqrt(np.prod(sel_cam['focal_length'])) / sel_cam['res_w'] * 2
            sel_cam['intrinsic'] = np.concatenate((
                sel_cam['focal_length'],
                sel_cam['center'],
                ## Below parameters do not exist
                # sel_cam['radial_distortion'],
                # sel_cam['tangential_distortion']
            ))

            ## Extrinsic Parameters
            RT = np.array([x for x in cam_i[5].split(' ')[-16:]], dtype='float32').reshape(4,4)[:3]
            R,T = RT[:,:3], RT[:,3:]
            sel_cam['orientation'] = Rotation.from_matrix(R).as_quat()[...,[3,0,1,2]].tolist()
            sel_cam['translation'] = (T / 1000.0).reshape(-1).tolist() ## mm to meters

            cams[i] = sel_cam

        return cams
            

    def _make_augmented_dataset(self, cfg): 
        ## Data Loading
        num_subjects = 8
        cams = self.cams
        num_cams = len(cams)

        data_3dhp_gt_file = os.path.join(self.dataset_root, 'data_3dhp_gt.npz')
        if os.path.isfile(data_3dhp_gt_file):
            data = np.load(data_3dhp_gt_file, allow_pickle=True)['db'].item()
        else:
            data = {}
            for subject in [f'S{i+1}' for i in range(num_subjects)]:
                for seq in ['Seq1','Seq2']:
                    kp3ds = sio.loadmat(os.path.join(self.dataset_root, '3DHP', f'{subject}/{seq}/annot.mat'))['annot3']
                    for cam in range(num_cams):
                        kp3d = kp3ds[cam,0].reshape(-1,28,3)[:,[7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 4, 3, 6]] / 1000 ## mm to meter
                        data.setdefault(cam, [])
                        data[cam].append(kp3d)

            np.savez_compressed(data_3dhp_gt_file, db=data)
        

        ## Fit Ground Planes for each camera
        logger.info(f"=>\tFitting Ground Plane...")
        geometry_path = os.path.join(self.dataset_root, '3dhp_geometry.npz')
        if os.path.isfile(geometry_path):
            geometry = np.load(geometry_path, allow_pickle=True)
            normals = geometry['normals'].item()
            ground_planes = geometry['ground_planes'].item()
            logger.info(f"=>\tLazy Loading Completed...!")
        else:
            normals = {}
            ground_planes = {}

            for cam in data:
                data_cam = data[cam]
                feets = np.concatenate([_data_cam[:,[self.joints['l-ankle'], self.joints['r-ankle']]].reshape(-1,3) for _data_cam in data_cam], axis=0)
                ones = np.ones((feets.shape[0], 1))
                feets_c = np.concatenate((feets,ones), axis=-1).astype('float32')

                ground_plane, total_loss, fitting_loss, norm_loss = fit_ground_plane(feets_c, lr=1e-3, num_iter=10000)
            
                normal = ground_plane[:3] / np.linalg.norm(ground_plane[:3])
                ## z-value of normal vector should be negative (unless, it means camera is located below the ground plane)
                normal *= -np.sign(normal[-1])
                ground_plane *= (normal[0] / ground_plane[0])


                ## To Adjust Numerical computation error (When ||normal_y|| ~= 1)
                ## In cam 11,12,13, normal_y > 0 --> It's because their scenes are flipped upside-down
                if cam < 11:
                    if np.sign(normal[1])==1:
                        normal[1] *= -1
                        ground_plane = np.concatenate([normal[:3], -ground_plane[-1:]])

                fitting_loss = abs(feets_c @ ground_plane / np.linalg.norm(ground_plane)).mean()

                logger.info(f'\t\t===== [CAM {cam}] =====')
                logger.info(f'\t\t[Normal] {normal}')
                logger.info(f'\t\t[Fitting Loss]: {fitting_loss:.3f}')
                logger.info(f"\t\t===============")

                normals[cam] = normal
                ground_planes[cam] = ground_plane

            np.savez_compressed(os.path.join(self.dataset_root, '3dhp_geometry.npz'), normals=normals, ground_planes=ground_planes)
            logger.info(f"=>\tCompleted...!")


        ## Augmentation
        logger.info(f'=> MAKE...')
        augtype = eval(f'cfg.{self.mode.upper()}').DATASET.SUBTYPE
        assert augtype in ('aug1', 'aug2', 'aug3', 'aug4'), f"Invalid augtype {augtype}"
        
        num_people_candi = cfg.AUG.NUM_PEOPLE_CANDI

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
        seq_per_comb = int(tot_sequences / len(num_people_candi) / len(GPT_candi) / len(GPR_candi) / num_cams)

        test_set_ratio = cfg.AUG.TEST_SET_RATIO
        test_seq_per_comb = int(seq_per_comb * test_set_ratio)
        train_seq_per_comb = seq_per_comb - test_seq_per_comb

        aug_print_freq = (cfg.AUG.PRINT_FREQ + frames_per_seq - 1) // frames_per_seq

        synth_db = {}
        synth_db['train'] = []
        synth_db['test'] = []
        
        seq_no = 0
        for num_people in num_people_candi:
            for GPT_trans in GPT_candi:
                for GPR_rot in GPR_candi:
                    logger.info(f'\tPROCESSING SEQ {seq_no}..')
                    for c in range(num_cams):
                        data_cam = data[c]
                        cam = cams[c]
                        normal = normals[c]
                        ground_plane = ground_planes[c]

                        s = 0
                        st = time.time()
                        while (s < seq_per_comb):
                            idxs = np.random.choice(len(data_cam), num_people, replace=False)
                            kp3ds = []
                            for idx in idxs:
                                kp3d = data_cam[idx]
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
                                
                                synth_db[target].append({
                                    'id': tuple([seq_no, c, s - (0 if target=='train' else train_seq_per_comb)]),
                                    'positions_3d': meta['positions_3d'],
                                    'positions_2d': meta['positions_2d'],
                                    'vis': meta['vis'],
                                    'img_path': [sorted(glob.glob(os.path.join(self.dataset_root, '3DHP', f'S1/Seq1/imageSequence/video_{c}/*.jpg')))[0]] * frames_per_seq,
                                    'cam': {
                                        'intrinsic': cam['intrinsic'],
                                        'res_w': cam['res_w'],
                                        'res_h': cam['res_h'],
                                        'normalization_factor': cam['normalization_factor']
                                    }
                                })

                                if ((s+1) * frames_per_seq) % aug_print_freq == 0:
                                    et = time.time()
                                    logger.info(f"\t\tN: {num_people} / GPT: {GPT_trans} / GPR: {GPR_rot/np.pi:.2f}PI / cam: {c} --- NUM PERSONS: {num_people} --- NUM INTER OCCLUDED: {meta['num_inter_occluded']:d} / NUM SELF OCCLUDED: {meta['num_self_occluded']:d} / NUM INVISIBLE: {meta['num_invisible']:d}")
                                    logger.info(f"\t\t\t[GPR-eff] {meta['GPR-eff']/np.pi:.3f} / [V] {meta['PT_trans_v']:.3f} ({meta['PT_trans_v_val_range'][0]:.3f}~{meta['PT_trans_v_val_range'][1]:.3f}) / [U] {meta['PT_trans_u']:.3f} ({meta['PT_trans_u_val_range'][0]:.3f}~{meta['PT_trans_u_val_range'][1]:.3f})")
                                    logger.info(f"\t\t\tTIME SPENT: {et-st:.5f}SEC")
                                    st = time.time()

                                s += 1
                    seq_no += 1

        np.savez_compressed(self.db_file, db=synth_db)
        logger.info(f'=> MAKE COMPLETED...!')
        return synth_db


    def evaluate(self, preds, mpjpe_threshold=0.500):
        assert len(preds)==len(self.db)  ### (B x [F, N, K, C])

        eval_list = []
        total_gt = 0
        
        for i in range(len(self.db)):
            pred, _ = self.eval_data_prepare(self.receptive_field, preds[i])

            target_clip = self.db[i]
            seq_id = target_clip['id']
            seq = seq_id[0]
            inputs_2d, _ = self.eval_data_prepare(self.receptive_field, target_clip['positions_2d'])
            inputs_obj = inputs_2d[0,0,:,0,0] != 0
            target_3d, _ = self.eval_data_prepare(self.receptive_field, target_clip['positions_3d'])
            target_vis, _ = self.eval_data_prepare(self.receptive_field, target_clip['vis'])
            target_obj = target_3d[0,0,:,0,0] != 0

            _eval_list, total_gt = evaluate_metrics(pred, inputs_obj, target_3d, target_vis, target_obj, total_gt=total_gt, root_id=self.root_idx, seq_id=seq)
            eval_list.extend(_eval_list)

        metrics = eval_list_to_metrics(eval_list, total_gt)

        return metrics
















