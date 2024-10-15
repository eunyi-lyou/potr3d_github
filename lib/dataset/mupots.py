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

from data.mupots.evalutils import *



JOINTS_DEF_MUPOTS = {
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

SKELETON_DEF_MUPOTS = {
    'parents': [16, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, -1, 14, 1],
    'joints_left': [5, 6, 7, 11, 12, 13],
    'joints_right': [2, 3, 4, 8, 9, 10],
    'symmetry': [[5,6,7,11,12,13], [2,3,4,8,9,10]]
}

CAM_LIST = list(range(1))


logger = logging.getLogger(__name__)


class MuPoTSDataset(JointsDataset):
    joints = JOINTS_DEF_MUPOTS
    skeleton = SKELETON_DEF_MUPOTS

    def __init__(self, cfg, make_chunk=True, mode='test'):
        super().__init__(cfg, mode)
        cfg_mode = eval(f'cfg.{mode.upper()}')

        assert mode=='test', f"Only test mode is available"
        assert self.dataset_type in ('gt', 'tracked'), f"Invalid dataset type option, {self.dataset_type}"

        self.joints = MuPoTSDataset.joints
        self.skeleton = MuPoTSDataset.skeleton
        assert len(self.joints) == self.num_joints, "Joints Definition Error"
        assert len(self.joints) == len(self.skeleton['parents']), "Joints Definition Error"

        self.cam_list = CAM_LIST
        self.seq_list = cfg_mode.DATASET.SEQ_LIST

        self.cams = self.get_cam(self.dataset_root, self.cam_list)

        self.db_file = os.path.join(self.dataset_root, f"data_mupots_{self.dataset_type + (f'_{cfg_mode.DATASET.SUBTYPE}' if cfg_mode.DATASET.TYPE=='synth' else '')}.npz")
        logger.info(f'=> Getting {self.dataset_type.upper()} db...')
        if os.path.isfile(self.db_file):
            self.db = np.load(self.db_file, allow_pickle=True)['db'].item()
            logger.info('=> Lazy Loading Completed...!')
        else:
            if self.dataset_type == 'gt':
                self.db = self._get_db()
            
            elif self.dataset_type == 'tracked':
                db_file_gt = os.path.join(self.dataset_root, f'data_mupots_gt.npz')
                if not os.path.isfile(db_file_gt):
                    logger.info(f'=> Getting GT db first...')
                    _ = self._get_db()
                db_file_tracked = os.path.join(self.dataset_root, cfg_mode.DATASET.TRACKED_FILE)
                self.db = self._make_tracked_dataset(cfg_mode, db_file_gt, db_file_tracked)
        
        self.db = self.db[cfg_mode.DATASET.SUBSET]
        
        if make_chunk:
            self.db = self._chunk_db()
        else:
            self._prepare_db()
    
    
    @staticmethod
    def get_cam(dataset_root, cam_list, seq_list=None):
        cam_file = os.path.join(dataset_root, 'MuPoTS_cameras.json')
        with open(cam_file) as cfile:
            cams = json.load(cfile, ignore_comments=True)

        return cams


    def _get_db(self):
        logger.info(f"=> MAKE...")
        db = {}
        db['train'] = []
        db['test'] = []

        def load_annot(fname):
            def parse_pose(dt):
                res = {}
                annot2 = dt['annot2'][0,0]
                annot3 = dt['annot3'][0,0]
                annot3_univ = dt['univ_annot3'][0,0]
                is_valid = dt['isValidFrame'][0,0][0,0]
                res['annot2'] = annot2
                res['annot3'] = annot3
                res['annot3_univ'] = annot3_univ
                res['is_valid'] = is_valid == 1
                return res 
            data = sio.loadmat(fname)['annotations']
            results = []
            num_frames, num_inst = data.shape[0], data.shape[1]
            for i in range(num_frames):
                buff = []
                for j in range(num_inst):                
                    buff.append(parse_pose(data[i,j]))
                results.append(buff)
            return results
        
        for seq in self.seq_list:
            logger.info(f'\tPROCESSING {seq}..')
            cams = self.cams[seq]
            seq_dir = os.path.join(self.dataset_root, self.dataset_dirname, seq)
            for c in range(len(self.cam_list)):
                cam = cams[c]
                annot_file = os.path.join(seq_dir, 'annot.mat')
                annot = load_annot(annot_file)

                num_frames = len(annot)
                num_persons = len(annot[0])

                images = sorted(glob.glob(os.path.join(seq_dir, '*.jpg')))

                db_temp_3d = []
                db_temp_2d = []
                db_temp_vis = []
                db_temp_img_path = images
                for f in range(num_frames):
                    poses_3d = []
                    poses_2d = []
                    vis = []
                    for p in range(num_persons):
                        poses_3d.append(annot[f][p]['annot3'])
                        poses_2d.append(annot[f][p]['annot2'])
                        if annot[f][p]['is_valid']:
                            _vis = np.ones(self.num_joints, dtype='bool')
                        else:
                            _vis = np.zeros(self.num_joints, dtype='bool')
                        vis.append(_vis)
                    db_temp_3d.append(poses_3d)
                    db_temp_2d.append(poses_2d)
                    db_temp_vis.append(vis)
                
                db_padded_3d = np.zeros((len(db_temp_3d), self.max_num_people, self.num_joints, 3), dtype='float32')
                db_padded_2d = np.zeros((len(db_temp_2d), self.max_num_people, self.num_joints, 2), dtype='float32')
                db_padded_vis = np.zeros((len(db_temp_vis), self.max_num_people, self.num_joints), dtype='bool')
                
                db_padded_3d[:,:num_persons] = np.asarray(db_temp_3d).transpose(0,1,3,2)[:,:num_persons]
                db_padded_2d[:,:num_persons] = np.asarray(db_temp_2d).transpose(0,1,3,2)[:,:num_persons]
                db_padded_vis[:,:num_persons] = np.asarray(db_temp_vis)[:,:num_persons]

                db['test'].append({
                    'id': tuple([seq, c, 0]),
                    'positions_3d': db_padded_3d / 1000,  ## mm to meter
                    'positions_2d': db_padded_2d,
                    'vis': db_padded_vis,
                    'img_path': db_temp_img_path,
                    'cam': {
                        'intrinsic': cam['intrinsic'],
                        'res_w': cam['res_w'],
                        'res_h': cam['res_h'],
                        'normalization_factor': np.sqrt(np.prod(cam['intrinsic'][:2])) / cam['res_w'] * 2
                    }
                })

        db['train'].sort(key=lambda x: x['id'])
        db['test'].sort(key=lambda x: x['id'])
        np.savez_compressed(self.db_file, db=db)
        logger.info(f"=> Make Completed...!")

        return db


    def evaluate(self, preds, mpjpe_threshold=500):
        assert len(preds)==len(self.db)  ### (B x [F, N, K, C])

        _, o1, _, _ = mpii_get_joints('relavant')
        _, all_joints = mpii_joint_groups()

        evaluation_mode = 0  # 0 for all, 1 for matched 
        safe_traversal_order = [14, 15, 1, 0, 16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
  
        def _evaluate(root_align=True, threshold=250):
            gts = {}
            predictions = {}
            valid_frames = {}
            for i in range(len(self.db)):
                db_i = self.db[i]
                _gt = db_i['positions_3d'] * 1000  ## meter to mm
                _pred = preds[i] * 1000  ## meter to mm
                
                tobj = _gt[0,:,0,0] != 0
                pobj = db_i['positions_2d'][0,:,0,0] != 0

                gt = _gt[:,tobj]
                vis = db_i['vis'][:,tobj]
                pred = _pred[:,pobj]                

                seq = db_i['id'][0]
                gts_seq = gts.setdefault(seq, [])
                preds_seq = predictions.setdefault(seq, [])
                valid_frames_seq = valid_frames.setdefault(seq, [])

                num_frames = gt.shape[0]
                for f in range(num_frames):
                    validity = False
                    valid_person_idx = vis[f].sum(-1) != 0
                    if valid_person_idx.all():
                        validity = True

                    matches = match(gt[f], pred[f], root_id=self.root_idx, root_align=root_align, threshold=threshold)
                    gts_seq_i = []
                    preds_seq_i = []
                    for k in range(len(matches)):
                        pred_considered = False
                        if valid_person_idx[k]:
                            if matches[k] != -1:
                                gt_root = gt[f][k][self.root_idx:(self.root_idx+1)]
                                gt_rel = gt[f][k] - gt_root
                                pred_root = pred[f][matches[k]][self.root_idx:(self.root_idx+1)]
                                pred_rel = pred[f][matches[k]] - pred_root
                                pred_rel = norm_by_bone_length(pred_rel, gt_rel, o1, safe_traversal_order[1:])
                                
                                gt_n = gt_rel + gt_root
                                pred_n = pred_rel + pred_root

                                pred_considered = True
                            else:
                                gt_n = gt[f][k].copy()
                                pred_n = pred[f][matches[k]].copy()
                                # pred_n = 100000 * np.ones_like(gt_n)
                                if evaluation_mode == 0:
                                    pred_considered = True
                                validity = False
                        
                        if pred_considered:
                            gts_seq_i.append(gt_n)
                            preds_seq_i.append(pred_n)
                        else:
                            gts_seq_i.append([])
                            preds_seq_i.append([])
                    
                    gts_seq.append(gts_seq_i)
                    preds_seq.append(preds_seq_i)
                    valid_frames_seq.append(validity)

            return gts, predictions, valid_frames


        ## Calculate Metrics
        metrics = {}
        pck_rel, pck_abs, mpjpe_rel, mpjpe_abs, mpjve_rel, mpjve_abs = [], [], [], [], [], []
        
        # Calculate PCK_rel / PCK_abs / MPJPE_rel / MPJPE_abs / MPJVE_rel
        pck_threshold = 150
        match_threshold = 250
        gts, predictions, valid_frames = _evaluate(root_align=True, threshold=match_threshold)
        for seq in gts:
            gts_seq = gts[seq]
            preds_seq = predictions[seq]
            valid_frames_seq = np.array(valid_frames[seq])

            num_frames = len(gts_seq)
            num_persons = len(gts_seq[0])
            valid_frames_idx = np.zeros(num_frames, dtype='bool')
            valid_frames_idx[:-1][valid_frames_seq[:-1] * valid_frames_seq[1:]] = True

            _pck_rel, _pck_abs, _mpjpe_rel, _mpjpe_abs, _mpjve_rel = [], [], [], [], []
            for f in range(num_frames):
                for p in range(num_persons):
                    if len(gts_seq[f][p]) > 0:
                        error = np.linalg.norm(gts_seq[f][p] - preds_seq[f][p], axis=-1)[all_joints]
                        error_rel = np.linalg.norm((gts_seq[f][p] - gts_seq[f][p][self.root_idx:(self.root_idx+1)]) - (preds_seq[f][p] - preds_seq[f][p][self.root_idx:(self.root_idx+1)]), axis=-1)[all_joints]

                        _pck_abs.append(error)
                        _pck_rel.append(error_rel)

            pck_rel.append(_pck_rel)
            pck_abs.append(_pck_abs)

        metrics[f'PCK_rel @a\\{pck_threshold:.0f}mm'] = np.mean([(np.array(seq_data) < pck_threshold).sum() / len(seq_data) / len(all_joints) if len(seq_data) > 0 else np.inf for seq_data in pck_rel]) * 100
        metrics[f'PCK_abs @a\\{pck_threshold:.0f}mm'] = np.mean([(np.array(seq_data) < pck_threshold).sum() / len(seq_data) / len(all_joints) if len(seq_data) > 0 else np.inf for seq_data in pck_abs]) * 100


        # Calculate MPJVE_abs
        match_threshold = 1000
        gts, predictions, valid_frames = _evaluate(root_align=False, threshold=match_threshold)
        for seq in gts:
            gts_seq = gts[seq]
            preds_seq = predictions[seq]
            valid_frames_seq = np.array(valid_frames[seq])

            num_frames = len(gts_seq)
            valid_frames_idx = np.zeros(num_frames, dtype='bool')
            valid_frames_idx[:-1][valid_frames_seq[:-1] * valid_frames_seq[1:]] = True

            _mpjve_abs = []
            for f in range(num_frames):
                for p in range(num_persons):
                    if len(gts_seq[f][p]) > 0:
                        error = np.linalg.norm(gts_seq[f][p] - preds_seq[f][p], axis=-1)[all_joints]
                        error_rel = np.linalg.norm((gts_seq[f][p] - gts_seq[f][p][self.root_idx:(self.root_idx+1)]) - (preds_seq[f][p] - preds_seq[f][p][self.root_idx:(self.root_idx+1)]), axis=-1)[all_joints]

                        _mpjpe_abs.append(error.mean())
                        _mpjpe_rel.append(error_rel.mean())

                    if valid_frames_idx[f]:
                        dgtdt = ((gts_seq[f+1][p] - gts_seq[f+1][p][self.root_idx:(self.root_idx+1)]) - (gts_seq[f][p] - gts_seq[f][p][self.root_idx:(self.root_idx+1)])) #[all_joints]
                        dpreddt = ((preds_seq[f+1][p] - preds_seq[f+1][p][self.root_idx:(self.root_idx+1)]) - (preds_seq[f][p] - preds_seq[f][p][self.root_idx:(self.root_idx+1)])) #[all_joints]
                        assert len(dgtdt) == len(dpreddt), 'Length Mismatch'
                        error_rel = np.linalg.norm(dgtdt - dpreddt, axis=-1)
                        

                        dgtdt = ((gts_seq[f+1][p]) - (gts_seq[f][p])) #[all_joints]
                        dpreddt = ((preds_seq[f+1][p]) - (preds_seq[f][p])) #[all_joints]
                        assert len(dgtdt) == len(dpreddt), 'Length Mismatch'
                        error = np.linalg.norm(dgtdt - dpreddt, axis=-1)
                        
                        _mpjve_rel.append(error_rel.mean())
                        _mpjve_abs.append(error.mean()) 

            mpjpe_rel.append(_mpjpe_rel)
            mpjpe_abs.append(_mpjpe_abs)
            mpjve_rel.append(_mpjve_rel)
            mpjve_abs.append(_mpjve_abs)

        metrics[f'MPJPE_rel @{match_threshold:.0f}mm'] = np.mean([np.mean(seq_data) if len(seq_data) > 0 else np.inf for seq_data in mpjpe_rel])
        metrics[f'MPJPE_abs @{match_threshold:.0f}mm'] = np.mean([np.mean(seq_data) if len(seq_data) > 0 else np.inf for seq_data in mpjpe_abs])
        metrics[f'MPJVE_rel @{match_threshold:.0f}mm'] = np.mean([np.mean(seq_data) if len(seq_data) > 0 else np.inf for seq_data in mpjve_rel])
        metrics[f'MPJVE_abs @{match_threshold:.0f}mm'] = np.mean([np.mean(seq_data) if len(seq_data) > 0 else np.inf for seq_data in mpjve_abs])


        return metrics
















