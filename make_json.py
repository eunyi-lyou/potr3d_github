import numpy as np
import json_tricks as json
import pickle
import time


def log(log_out, write_str):
    with open(log_out, 'a') as f:
        f.write(str(write_str) + '\n')
    print(write_str)


if __name__=='__main__':
    scale_adjustment = 1.15
    tot_frames = 1344000 // 2
    stride = 1 ### stride=10000 --> 3.1M
    log_file = 'log_make_vid3dhp_json.txt'

    for augtype in ('aug2',):
        log(log_file, f'PROCESSING {augtype.upper()} >>>')
        data_2d = np.load(f'data/data_2d_vid3dhp_{augtype}_img_1_3M_230215_GP_Y_OCC_N.npz', allow_pickle=True)
        data_3d = np.load(f'data/data_3d_vid3dhp_{augtype}_img_1_3M_230215_GP_Y_OCC_N.npz', allow_pickle=True)
        log(log_file, f'LOADING COMPLETED...!')

        kp2ds = data_2d['positions_2d'].item()
        kp3ds = data_3d['positions_3d'].item()

        with open('3dhp_cameras.pkl', 'rb') as f:
            cams = pickle.load(f)
        num_cams = len(cams)


        
        pose_annots = []
        img_annots = []

        annot_id = 0
        img_id = 0

        st = time.time()
        for k in kp3ds:
            if 'TS' in k:
                continue
            
            et = time.time()
            log(log_file, f'\t{img_id/(tot_frames // stride)*100:.0f}% COMPLETED...!\t({et-st:.0f}SEC ELAPSED)')
            log(log_file, f'{len(pose_annots)}, {len(img_annots)}')
            st = time.time()
            for c in range(num_cams-3): ## cam 11,12,13 are inverted ; so eliminated
                cam = cams[c]
                num_clips = len(kp3ds[k][c])
                # for v in range(num_clips):
                for v in range(0, num_clips, stride):
                    kp2d = kp2ds[k][c][v]
                    kp3d = kp3ds[k][c][v]

                    valid_idx = kp2d[0,:,0,0] != 0

                    kp2d = kp2d[:,valid_idx]
                    kp3d = kp3d[:,valid_idx]

                    num_frames = len(kp2d)
                    num_persons = len(kp2d[0])
                    # for f in range(0, num_frames, stride):
                    for f in range(num_frames):
                        left_top = np.min(kp2d[f], axis=1)
                        right_bottom = np.max(kp2d[f], axis=1)

                        cx = (left_top[:,0] + right_bottom[:,0]) / 2
                        cy = (left_top[:,1] + right_bottom[:,1]) / 2

                        center = np.stack([cx, cy], axis=-1)

                        w = (right_bottom[:,0] - left_top[:,0]) * scale_adjustment
                        h = (right_bottom[:,1] - left_top[:,1]) * scale_adjustment

                        left_top = center - np.stack([w/2, h/2], axis=-1)

                        for p in range(num_persons):
                            pose_annots.append({
                                'id': annot_id,
                                'image_id': img_id,
                                'keypoints_img': kp2d[f][p].tolist(),
                                'keypoints_cam': kp3d[f][p].tolist(),
                                'bbox': [left_top[p][0], left_top[p][1], w[p], h[p]],
                                'keypoints_vis': np.ones_like(kp2d[f][p][:,0]).tolist(),
                            })
                            annot_id += 1

                        img_annots.append({
                            'id': img_id,
                            'width': cam['width'],
                            'height': cam['height'],
                            'file_name': None,
                            'f': cam['f'],
                            'c': cam['c'],
                            'R': cam['R'],
                            'T': cam['T']
                        })
                        img_id += 1

        
        log(log_file, f'FINISHED')
        log(log_file, f'{len(pose_annots)}, {len(img_annots)}')
        # data = {'annotations': pose_annots, 'images': img_annots}
        log(log_file, f'SAVING STARTS')
        # with open(f'/data/dataset/VirtualPose/vid3dhp_{augtype}.json', 'w') as f:
        #     json.dump(data, f, sort_keys=True, indent=4)
        
        for k in range(10):
            log(log_file, f'{k+1}/10')

            start_idx_1 = len(pose_annots) // 10 * k
            end_idx_1 = len(pose_annots) // 10 * (k+1)

            start_idx_2 = len(img_annots) // 10 * k
            end_idx_2 = len(img_annots) // 10 * (k+1)

            data_k = data = {'annotations': pose_annots[start_idx_1:end_idx_1], 'images': img_annots[start_idx_2:end_idx_2]}
            np.savez_compressed(f'/data/dataset/VirtualPose/vid3dhp_{augtype}_{k}_230221.npz', data=data_k)
        log(log_file, f'SAVED2')