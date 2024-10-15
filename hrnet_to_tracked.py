import torch
import numpy as np
import json_tricks as json


# hrnet_res = json.load(open('../hrnet/output/coco/pose_hrnet/w48_384x288_ft_3/results/keypoints_mupots_results_0_det_bbox.json', 'r'))
hrnet_res = json.load(open('/home/shawn/data/projects/3Dpose/hrnet/output/coco/pose_hrnet/w32_384x288_ft_mucococo_7/results/mupots_res_gt_track.json', 'r'))

ids = np.array([0, 201, 251, 401, 301, 261, 541, 431, 551, 501, 251, 701, 251, 341, 626, 801, 501, 401, 126, 431, 501]).astype(np.int)

max_num_people = 10
num_joints = 17
num_people = np.array([2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]).astype(np.int)

cs_thre = 0.3

dets = {}
prev_img_id = -1
person_idx = 0
ts = -1

for d in hrnet_res:
    if d['image_id'] >= np.cumsum(ids)[ts+1]:
        ts += 1
        
        if f'TS{ts+1}' not in dets:
            dets[f'TS{ts+1}'] = np.zeros((ids[ts+1], max_num_people, num_joints, 2))
            dets[f'TS{ts+1}'][:,:num_people[ts]] = -1.

    if d['image_id'] != prev_img_id:
        person_idx = 0
        prev_img_id = d['image_id']
    else:
        person_idx += 1

    if ts < 5:
        res_w = 2048
        res_h = 2048
    else:
        res_w = 1920
        res_h = 1080

    data = dets[f'TS{ts+1}']

    frame_no = d['image_id'] - np.cumsum(ids)[ts]
    xs = np.array(d['keypoints'][0::3])
    ys = np.array(d['keypoints'][1::3])
    cs = np.array(d['keypoints'][2::3])

    mask = (cs > cs_thre) * (xs >= 0) * (xs <= res_w - 1) * (ys >= 0) * (ys <= res_h - 1)


    if mask.sum() == 0:
        print(ts, frame_no, d['image_id'])
    
    data[frame_no, person_idx, mask] = np.concatenate((xs[...,None][mask], ys[...,None][mask]), axis=-1)

    

        

np.savez_compressed('data/data_2d_mupots_tracked_hrnet.npz', tracks=dets)