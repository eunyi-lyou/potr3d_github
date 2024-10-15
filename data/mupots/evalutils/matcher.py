import numpy as np 
from data.mupots.evalutils import norm_pose


def match(gt, pred, root_id=14, root_align=False, threshold=250):
    matches = []
    gt = gt.copy()  
    pred = pred.copy()
    if root_align:
        gt -= gt[:,root_id:(root_id+1)]
        pred -= pred[:,root_id:(root_id+1)]
    for i in range(len(gt)):
        gt_i = gt[i]
        diffs = []
        for j in range(len(pred)):
            pred_j = pred[j]
            # pred_j = norm_pose.procrusted(pred_j, gt_i)
            diff = np.linalg.norm(pred_j - gt_i, axis=-1).mean()
            diffs.append(diff)
        diffs = np.float32(diffs)
        idx = np.argmin(diffs)
        if diffs.min() > threshold:
            matches.append(-1)
        else:
            matches.append(idx)
    return matches
