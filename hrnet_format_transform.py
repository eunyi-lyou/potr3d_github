import torch
import sys
import re
import numpy as np
import json_tricks as json

np.random.seed(2234)


## Convert HRNet result file into the detection format


def demo():
    demo_res = json.load(open('../hrnet/output/coco/pose_hrnet/w48_384x288_ft_demo_7/results/demo_last_results.json'))

    vids_ids = ['demo_22', 'demo_23', 'demo_24', 'demo_25', 'demo_26', 'demo_27', 'demo_28', 'demo_29']
    
    ids = np.array([0, 278, 400, 291, 197, 2058, 492, 493, 1061])
    ids = np.cumsum(ids)


    dets = {}
    ts = 0
    prev_imgid = -1

    cs_thre = 0.4

    for ann in demo_res:
        imgid = ann['image_id']

        if imgid != prev_imgid:
            if imgid >= ids[ts+1]:
                for _ in range(ids[ts+1] - prev_imgid - 1):
                    if det[-1]['outputs'] is not None and len(det[-1]['outputs']) > 0:
                        det[-1]['outputs'] = torch.from_numpy(np.array(det[-1]['outputs']).astype('float32'))
                        det[-1]['kps'] = torch.from_numpy(np.array(det[-1]['kps']).astype('float32'))
                    det.append({'outputs': None, 'kps': None})
                ts += 1
            
            if imgid > 0:
                if det[-1]['outputs'] is not None and len(det[-1]['outputs']) > 0:
                    det[-1]['outputs'] = torch.from_numpy(np.array(det[-1]['outputs']).astype('float32'))
                    det[-1]['kps'] = torch.from_numpy(np.array(det[-1]['kps']).astype('float32'))
                else:
                    det[-1]['outputs'] = None
                    det[-1]['kps'] = None
                # except:
                #     raise ValueError(f'{imgid}, {ts}, {ids[ts]}, {ids[ts+1]}')

            det = dets.setdefault(vids_ids[ts], [])
                
            for _ in range(imgid - max(ids[ts], prev_imgid+1)):
                det.append({'outputs': None, 'kps': None})
            
            det.append({})

            prev_imgid = imgid

        kps = det[-1].setdefault('kps', [])
        bboxes = det[-1].setdefault('outputs', [])

        ## kps
        kp = np.array(ann['keypoints']).reshape(17,3)
        
        ## bbox
        cx, cy = ann['center']
        sx, sy = ann['scale']
        score = ann['score']
        w = sx * 200 / 1.25
        h = sy * 200 / 1.25
        x0 = cx - w/2
        y0 = cy - h/2

        res_w, res_h = 1920, 1080
        
        x1 = np.max((0, x0))
        y1 = np.max((0, y0))
        x2 = np.min((res_w - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((res_h - 1, y1 + np.max((0, h - 1))))

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        if kp[:,2].mean() >= cs_thre and x2 >= x1 and y2 >= y1:
            bboxes.append([cx, cy, w, h, 1., score])
            kps.append(kp)



    if not isinstance(det[-1]['outputs'], np.ndarray):
        if len(det[-1]['outputs']) > 0:
            det[-1]['outputs'] = torch.from_numpy(np.array(det[-1]['outputs']).astype('float32'))
            det[-1]['kps'] = torch.from_numpy(np.array(det[-1]['kps']).astype('float32'))
        else:
            det[-1]['outputs'] = None
            det[-1]['kps'] = None


    np.savez_compressed('data_2d_demo_last_hrnet_det3.npz', detections=dets)


def mupots():
    # hrnet_res = json.load(open('../hrnet/output/coco/pose_hrnet/w48_384x288_ft_3/results/keypoints_mupots_results_0_det_bbox.json', 'r'))
    hrnet_res = json.load(open('/home/shawn/data/projects/3Dpose/hrnet/output/coco/pose_hrnet/w48_384x288_ft_7/results/mupots_results_det_bbox.json', 'r'))

    ids = np.array([0, 201, 251, 401, 301, 261, 541, 431, 551, 501, 251, 701, 251, 341, 626, 801, 501, 401, 126, 431, 501])
    ids = np.cumsum(ids)


    dets = {}
    ts = 0
    prev_imgid = -1

    cs_thre = 0.4

    for ann in hrnet_res:
        imgid = ann['image_id']

        if imgid != prev_imgid:
            if imgid >= ids[ts+1]:
                for _ in range(ids[ts+1] - prev_imgid - 1):
                    det.append({'outputs': None, 'kps': None})
                ts += 1
            
            if imgid > 0:
                if len(det[-1]['outputs']) > 0:
                    det[-1]['outputs'] = torch.from_numpy(np.array(det[-1]['outputs']).astype('float32'))
                    det[-1]['kps'] = torch.from_numpy(np.array(det[-1]['kps']).astype('float32'))
                else:
                    det[-1]['outputs'] = None
                    det[-1]['kps'] = None

            det = dets.setdefault(f'TS{ts+1}', [])
                
            for _ in range(imgid - max(ids[ts], prev_imgid+1)):
                det.append({'outputs': None, 'kps': None})
            
            det.append({})

            prev_imgid = imgid

        kps = det[-1].setdefault('kps', [])
        bboxes = det[-1].setdefault('outputs', [])

        ## kps
        kp = np.array(ann['keypoints']).reshape(17,3)
        
        ## bbox
        cx, cy = ann['center']
        sx, sy = ann['scale']
        score = ann['score']
        w = sx * 200 / 1.25
        h = sy * 200 / 1.25
        x0 = cx - w/2
        y0 = cy - h/2

        if ts <= 4:
            res_w, res_h = 2048, 2048
        else:
            res_w, res_h = 1920, 1080
        
        x1 = np.max((0, x0))
        y1 = np.max((0, y0))
        x2 = np.min((res_w - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((res_h - 1, y1 + np.max((0, h - 1))))

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        if kp[:,2].mean() >= cs_thre and x2 >= x1 and y2 >= y1:
            bboxes.append([cx, cy, w, h, 1., score])
            kps.append(kp)



    if not isinstance(det[-1]['outputs'], np.ndarray):
        if len(det[-1]['outputs']) > 0:
            det[-1]['outputs'] = torch.from_numpy(np.array(det[-1]['outputs']).astype('float32'))
            det[-1]['kps'] = torch.from_numpy(np.array(det[-1]['kps']).astype('float32'))
        else:
            det[-1]['outputs'] = None
            det[-1]['kps'] = None


    np.savez_compressed('data/data_2d_mupots_hrnet_det3.npz', detections=dets)




def panoptic_ver2():
    panoptic_res = json.load(open('/data/projects/3Dpose/hrnet/output/coco/pose_hrnet/w48_384x288_ft_7/results/panoptic_ver2_results_det_bbox.json', 'r'))
    panoptic_images = np.load(f'/data/dataset/panoptic-toolbox/data_2d_panoptic_ver2_gt.npz', allow_pickle=True)['images'].item()

    VAL_LIST = [
        '160226_haggling1',
        '160422_haggling1', 
        '160226_mafia1', 
        '160422_ultimatum1',
        '160906_pizza1'
    ]
    CAM_LIST = [(0,16), (0,30)]

    dets = {}

    img_path_dict = {}
    seqs = panoptic_images.keys()
    img_id = 0
    num_images_cum_s = {}
    num_images_cum_e = {}
    for seq in seqs:
        if seq not in VAL_LIST:
            continue
        dets_seq = dets.setdefault(seq, [])
        for c, cam_c in enumerate(CAM_LIST):
            dets_seq_cam = []
            num_clips = len(panoptic_images[seq][c])
            for v in range(num_clips):
                dets_seq_cam.append([])
                num_images_cum_s[(seq, c, v)] = img_id
                clip = panoptic_images[seq][c][v]
                for im in clip:
                    img_path_dict[img_id] = (seq, c, v)
                    img_id += 1
                num_images_cum_e[(seq, c, v)] = img_id
            dets_seq.append(dets_seq_cam)


    test_combs = [['160226_haggling1', 0], ['160422_haggling1', 1], ['160226_mafia1', 0], ['160226_mafia1', 1], ['160422_ultimatum1', 0], ['160422_ultimatum1', 1], ['160906_pizza1', 0], ['160906_pizza1', 1]]
    for comb in test_combs:
        seq, c = comb
        cam_c = CAM_LIST[c]
        num_clips = len(panoptic_images[seq][c])
        if not (seq=='160226_mafia1' and c==0):
            selected_idx = np.random.choice(num_clips, 1)[0]
        else:
            selected_idx = 0 ## avoiding too long sequence
        comb.append(selected_idx) 
    test_combs = [tuple(comb) for comb in test_combs]
    print(test_combs)      


    imgid = -1
    comb = None
    flag = False

    cs_thre = 0.4

    for ann in panoptic_res:
        cur_imgid = ann['image_id']

        cur_seq, cur_c, cur_v = img_path_dict[cur_imgid]
        cur_comb = (cur_seq, cur_c, cur_v)

        if cur_imgid != imgid:
            if flag:
                if len(dets[comb[0]][comb[1]][comb[2]][-1]['outputs']) > 0:
                    dets[comb[0]][comb[1]][comb[2]][-1]['outputs'] = torch.from_numpy(np.array(dets[comb[0]][comb[1]][comb[2]][-1]['outputs']).astype('float32'))
                    dets[comb[0]][comb[1]][comb[2]][-1]['kps'] = torch.from_numpy(np.array(dets[comb[0]][comb[1]][comb[2]][-1]['kps']).astype('float32'))
                else:
                    dets[comb[0]][comb[1]][comb[2]][-1]['outputs'] = None
                    dets[comb[0]][comb[1]][comb[2]][-1]['kps'] = None

            if cur_comb != comb:
                if flag:
                    for _ in range(num_images_cum_e[comb] - imgid - 1):
                        dets[comb[0]][comb[1]][comb[2]].append({'outputs': None, 'kps': None})
                    
                flag = False
                if cur_comb in test_combs:
                    print(*cur_comb)
                    flag = True           
            
            if flag:
                if cur_comb != comb:
                    for _ in range(cur_imgid - num_images_cum_s[cur_comb]):
                        dets[cur_comb[0]][cur_comb[1]][cur_comb[2]].append({'outputs': None, 'kps': None}) 
                else:
                    for _ in range(cur_imgid - imgid - 1):
                        dets[comb[0]][comb[1]][comb[2]].append({'outputs': None, 'kps': None})

                dets[cur_comb[0]][cur_comb[1]][cur_comb[2]].append({})            
            
            imgid = cur_imgid
            comb = cur_comb
            

        if flag:
            kps = dets[comb[0]][comb[1]][comb[2]][-1].setdefault('kps', [])
            bboxes = dets[comb[0]][comb[1]][comb[2]][-1].setdefault('outputs', [])

            ## kps
            kp = np.array(ann['keypoints']).reshape(15,3)
            
            ## bbox
            cx, cy = ann['center']
            sx, sy = ann['scale']
            score = ann['score']
            w = sx * 200 / 1.25
            h = sy * 200 / 1.25
            x0 = cx - w/2
            y0 = cy - h/2

            res_w, res_h = 1920, 1080
            
            x1 = np.max((0, x0))
            y1 = np.max((0, y0))
            x2 = np.min((res_w - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((res_h - 1, y1 + np.max((0, h - 1))))

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            if kp[:,2].mean() >= cs_thre and x2 >= x1 and y2 >= y1:
                bboxes.append([cx, cy, w, h, 1., score])
                kps.append(kp)


    if not isinstance(dets[comb[0]][comb[1]][comb[2]][-1]['outputs'], np.ndarray):
        if len(dets[comb[0]][comb[1]][comb[2]][-1]['outputs']) > 0:
            dets[comb[0]][comb[1]][comb[2]][-1]['outputs'] = torch.from_numpy(np.array(dets[comb[0]][comb[1]][comb[2]][-1]['outputs']).astype('float32'))
            dets[comb[0]][comb[1]][comb[2]][-1]['kps'] = torch.from_numpy(np.array(dets[comb[0]][comb[1]][comb[2]][-1]['kps']).astype('float32'))
        else:
            dets[comb[0]][comb[1]][comb[2]][-1]['outputs'] = None
            dets[comb[0]][comb[1]][comb[2]][-1]['kps'] = None


    np.savez_compressed('data/data_2d_panoptic_ver2_hrnetw48_det3.npz', detections=dets)


def panoptic_ver1():
    panoptic_res = json.load(open('/data/projects/3Dpose/hrnet/output/coco/pose_hrnet/w48_384x288_ft_panoptic_7/results/keypoints_panoptic_ver1_train_results_0.json', 'r'))
    panoptic_images = np.load(f'/data/dataset/panoptic-toolbox/data_2d_panoptic_ver1_gt.npz', allow_pickle=True)['images'].item()

    TRAIN_LIST = [
        '160422_ultimatum1',
        '160224_haggling1',  ## Empty (No Valid Clip)
        '160226_haggling1',
        '161202_haggling1',
        '160906_ian1',  ## Empty (No Valid Clip)
        '160906_ian2',
        '160906_ian3',
        '160906_band1',
        '160906_band2',
        '160906_band3'   ## hd_00_03.mp4 seems broken
    ]
    CAM_LIST = [(0, 3), (0, 6), (0, 12), (0, 13), (0, 23)]

    dets = {}

    img_path_dict = {}
    seqs = panoptic_images.keys()
    img_id = 0
    num_images_cum_s = {}
    num_images_cum_e = {}
    for seq in seqs:
        if seq not in TRAIN_LIST:
            continue
        dets_seq = dets.setdefault(seq, [])
        for c, cam_c in enumerate(CAM_LIST):
            dets_seq_cam = []
            num_clips = len(panoptic_images[seq][c])
            for v in range(num_clips):
                dets_seq_cam.append([])
                num_images_cum_s[(seq, c, v)] = img_id
                clip = panoptic_images[seq][c][v]
                for im in clip:
                    img_path_dict[img_id] = (seq, c, v)
                    img_id += 1
                num_images_cum_e[(seq, c, v)] = img_id
            dets_seq.append(dets_seq_cam)


    test_combs = [['160422_ultimatum1', 1], ['160422_ultimatum1', 3], ['160226_haggling1', 1], ['160226_haggling1', 3]]
    for comb in test_combs:
        seq, c = comb
        cam_c = CAM_LIST[c]
        num_clips = len(panoptic_images[seq][c])
        selected_idx = np.random.choice(num_clips, 1)[0]
        comb.append(selected_idx) 
    test_combs = [tuple(comb) for comb in test_combs]
    print(test_combs)      


    imgid = -1
    comb = None
    flag = False

    cs_thre = 0.4

    for ann in panoptic_res:
        cur_imgid = ann['image_id']

        cur_seq, cur_c, cur_v = img_path_dict[cur_imgid]
        cur_comb = (cur_seq, cur_c, cur_v)

        if cur_imgid != imgid:
            if flag:
                if len(dets[comb[0]][comb[1]][comb[2]][-1]['outputs']) > 0:
                    dets[comb[0]][comb[1]][comb[2]][-1]['outputs'] = torch.from_numpy(np.array(dets[comb[0]][comb[1]][comb[2]][-1]['outputs']).astype('float32'))
                    dets[comb[0]][comb[1]][comb[2]][-1]['kps'] = torch.from_numpy(np.array(dets[comb[0]][comb[1]][comb[2]][-1]['kps']).astype('float32'))
                else:
                    dets[comb[0]][comb[1]][comb[2]][-1]['outputs'] = None
                    dets[comb[0]][comb[1]][comb[2]][-1]['kps'] = None

            if cur_comb != comb:
                if flag:
                    for _ in range(num_images_cum_e[comb] - imgid - 1):
                        dets[comb[0]][comb[1]][comb[2]].append({'outputs': None, 'kps': None})
                    
                flag = False
                if cur_comb in test_combs:
                    print(*cur_comb)
                    flag = True           
            
            if flag:
                if cur_comb != comb:
                    for _ in range(cur_imgid - num_images_cum_s[cur_comb]):
                        dets[cur_comb[0]][cur_comb[1]][cur_comb[2]].append({'outputs': None, 'kps': None}) 
                else:
                    for _ in range(cur_imgid - imgid - 1):
                        dets[comb[0]][comb[1]][comb[2]].append({'outputs': None, 'kps': None})

                dets[cur_comb[0]][cur_comb[1]][cur_comb[2]].append({})            
            
            imgid = cur_imgid
            comb = cur_comb
            

        if flag:
            kps = dets[comb[0]][comb[1]][comb[2]][-1].setdefault('kps', [])
            bboxes = dets[comb[0]][comb[1]][comb[2]][-1].setdefault('outputs', [])

            ## kps
            kp = np.array(ann['keypoints']).reshape(15,3)
            
            ## bbox
            cx, cy = ann['center']
            sx, sy = ann['scale']
            score = ann['score']
            w = sx * 200 / 1.25
            h = sy * 200 / 1.25
            x0 = cx - w/2
            y0 = cy - h/2

            res_w, res_h = 1920, 1080
            
            x1 = np.max((0, x0))
            y1 = np.max((0, y0))
            x2 = np.min((res_w - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((res_h - 1, y1 + np.max((0, h - 1))))

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            if kp[:,2].mean() >= cs_thre and x2 >= x1 and y2 >= y1:
                bboxes.append([cx, cy, w, h, 1., score])
                kps.append(kp)


    if not isinstance(dets[comb[0]][comb[1]][comb[2]][-1]['outputs'], np.ndarray):
        if len(dets[comb[0]][comb[1]][comb[2]][-1]['outputs']) > 0:
            dets[comb[0]][comb[1]][comb[2]][-1]['outputs'] = torch.from_numpy(np.array(dets[comb[0]][comb[1]][comb[2]][-1]['outputs']).astype('float32'))
            dets[comb[0]][comb[1]][comb[2]][-1]['kps'] = torch.from_numpy(np.array(dets[comb[0]][comb[1]][comb[2]][-1]['kps']).astype('float32'))
        else:
            dets[comb[0]][comb[1]][comb[2]][-1]['outputs'] = None
            dets[comb[0]][comb[1]][comb[2]][-1]['kps'] = None


    np.savez_compressed('data/data_2d_panoptic_ver1_hrnetw48_det3.npz', detections=dets)



if __name__ == "__main__":
    mode = sys.argv[1]

    eval(mode)()
