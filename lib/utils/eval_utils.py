import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

from lib.utils.utils import *
from lib.camera.camera import *



def evaluate_metrics(preds, inputs_obj, target, target_vis, target_obj, total_gt=0, root_id=2, seq_id=None):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert preds.shape == target.shape, f'Different shapes, preds: {preds.shape} / target: {target.shape}'

    eval_list = []

    B = preds.shape[0]
    for b in range(B):
        ## (There may exist no-obj sequence due to PXTH thresholding)
        if inputs_obj.sum() == 0:
            total_gt += target.shape[2]
            continue

        target_kps = target[b][:,target_obj].transpose(1,0,2,3)  ## (N, F, K, C)
        pred_kps = preds[b][:,inputs_obj].transpose(1,0,2,3)  ## (N, F, K, C)
        vis_mask = target_vis[b][:,target_obj].transpose(1,0,2)[..., None]  ## (N, F, K, 1)

        m, n = target_kps.shape[0], pred_kps.shape[0]            
        
        ## Calculate MPJPE_abs/rel
        res_mpjpe_abs = np.zeros((m,n))
        res_mpjpe_rel = np.zeros((m,n))
        for i in range(m):
            for j in range(n):
                # Calculate MPJPE_abs
                res_mpjpe_abs[i,j] = np.linalg.norm((target_kps[i] - pred_kps[j]) * vis_mask[i], axis=-1).sum() / (vis_mask[i].sum())
                # Calculate MPJPE_rel
                res_mpjpe_rel[i,j] = np.linalg.norm(((target_kps[i] - target_kps[i,:,root_id:(root_id+1)]) - (pred_kps[j] - pred_kps[j,:,root_id:(root_id+1)])) \
                    * vis_mask[i], axis=-1).sum() / (vis_mask[i].sum())

        
        ## Calculate MPJVE_abs/rel
        pred_kps_vel = pred_kps[:,1:] - pred_kps[:,:-1]
        target_kps_vel = target_kps[:,1:] - target_kps[:,:-1]
        vis_mask_vel = vis_mask[:,1:] * vis_mask[:,:-1]
        
        res_mpjve_abs = np.zeros((m,n))
        res_mpjve_rel = np.zeros((m,n))
        if len(pred_kps_vel) > 0:
            for i in range(m):
                for j in range(n):
                    # Calculate MPJVE_abs
                    res_mpjve_abs[i,j] = np.linalg.norm((target_kps_vel[i] - pred_kps_vel[j]) * vis_mask_vel[i], axis=-1).sum() / (vis_mask_vel[i].sum())
                    # Calculate MPJVE_rel
                    res_mpjve_rel[i,j] = np.linalg.norm(((target_kps_vel[i] - target_kps_vel[i,:,root_id:(root_id+1)]) - (pred_kps_vel[j] - pred_kps_vel[j,:,root_id:(root_id+1)])) \
                        * vis_mask_vel[i], axis=-1).sum() / (vis_mask_vel[i].sum())


        ## Match Preds to GT with MPJPE_abs
        matched_pred = np.arange(n)
        try:
            matched_gt = np.argmin(res_mpjpe_rel, axis=0)
        except:
            raise ValueError(f'Matching Failed, number of target: {m}, number of preds: {n}')
        # matched_gt, matched_pred = linear_sum_assignment(res_mpjpe)

        mpjpe_abs = res_mpjpe_abs[matched_gt, matched_pred]
        mpjpe_rel = res_mpjpe_rel[matched_gt, matched_pred]
        mpjve_abs = res_mpjve_abs[matched_gt, matched_pred]
        mpjve_rel = res_mpjve_rel[matched_gt, matched_pred]

        for k in range(n):
            eval_list.append({
                'id': seq_id,
                'mpjpe_abs': mpjpe_abs[k],
                'mpjpe_rel': mpjpe_rel[k],
                'mpjve_abs': mpjve_abs[k],
                'mpjve_rel': mpjve_rel[k],
                'gt_id': matched_gt[k] + total_gt,
            })

        total_gt += m

    return eval_list, total_gt


def eval_list_to_metrics(eval_list, total_gt, mpjpe_threshold=0.500):
    eval_list.sort(key=lambda x: x['mpjpe_abs'])
    
    gt_det = []
    mpjpes_abs, mpjpes_rel, mpjves_abs, mpjves_rel = [], [], [], []
    for i, item in enumerate(eval_list):
        if item['mpjpe_abs'] < mpjpe_threshold and item['gt_id'] not in gt_det:
            mpjpes_abs.append(item['mpjpe_abs'])
            mpjpes_rel.append(item['mpjpe_rel'])
            mpjves_abs.append(item['mpjve_abs'])
            mpjves_rel.append(item['mpjve_rel'])
            gt_det.append(item['gt_id'])

    metrics = {
        f'MPJPE_abs @{mpjpe_threshold*1000:.0f}mm': np.mean(mpjpes_abs) * 1000 if len(mpjpes_abs) > 0 else np.inf,
        f'MPJPE_rel @{mpjpe_threshold*1000:.0f}mm': np.mean(mpjpes_rel) * 1000 if len(mpjpes_rel) > 0 else np.inf,
        f'MPJVE_abs @{mpjpe_threshold*1000:.0f}mm': np.mean(mpjves_abs) * 1000 if len(mpjves_abs) > 0 else np.inf,
        f'MPJVE_rel @{mpjpe_threshold*1000:.0f}mm': np.mean(mpjves_rel) * 1000 if len(mpjves_rel) > 0 else np.inf,
    }

    mpjpe_thresholds = np.arange(25, 155, 25) / 1000
    for t in mpjpe_thresholds:
        ap, rec = _eval_list_to_ap(eval_list, total_gt, mpjpe_threshold=t)
        metrics[f'AP @{t*1000:.0f}mm'] = ap*100  
        metrics[f'Recall @{t*1000:.0f}mm'] = rec*100
    for t in mpjpe_thresholds:
        ap_rel, rec_rel = _eval_list_to_ap(eval_list, total_gt, root_align=True, mpjpe_threshold=t)
        metrics[f'AP_rel @a\\{t*1000:.0f}mm'] = ap_rel*100 
        metrics[f'Recall_rel @a\\{t*1000:.0f}mm'] = rec_rel*100 

    return metrics


def _eval_list_to_ap(eval_list, total_gt, root_align=False, mpjpe_threshold=0.500):
    if not root_align:
        eval_list.sort(key=lambda x: x['mpjpe_abs'])
    else:
        eval_list.sort(key=lambda x: x['mpjpe_rel'])
    
    tp = np.zeros(len(eval_list))
    fp = np.zeros(len(eval_list))
    gt_det = []
    for i, item in enumerate(eval_list):
        val = item['mpjpe_abs'] if not root_align else item['mpjpe_rel']
        if (val <= mpjpe_threshold) and (item['gt_id'] not in gt_det):
            gt_det.append(item['gt_id'])
            tp[i] = 1
        else:
            fp[i] = 1

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    recall = tp / (total_gt + 1e-5)
    precise = tp / (tp + fp + 1e-5)
    for n in range(len(eval_list)-2, -1, -1):
        precise[n] = max(precise[n], precise[n+1])

    precise = np.concatenate(([0], precise, [0]))
    recall = np.concatenate(([0], recall, [1]))
    index = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])  

    return ap, recall[-2]



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count