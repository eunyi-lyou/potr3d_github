import numpy as np
import glob
import os
import math
import time
import scipy.io as sio
from scipy.optimize import linprog

import sys
sys.path.append('.')

from lib.utils.utils import *

import matplotlib
matplotlib.use('Agg')

import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D



np.random.seed(2234)
num_cam = 14

def get_cam():
    _cams = []
    with open(f'/data/dataset/3DHP/S1/Seq1/camera.calibration', 'r') as f:   ## All camera calibration files are the same
        o = f.readline().strip()
        while o != '':
            _cams.append(o)
            o = f.readline().strip()
    _cams = _cams[1:]

    cams = {}
    for i in range(num_cam):
        _cam_i = _cams[7*i:7*(i+1)]
        cam_i = {}
        cam_i['res_w'], cam_i['res_h'] = [int(x) for x in _cam_i[2].split(' ')[-2:]]
        cam_i['K'] = np.array([float(x) for x in _cam_i[4].split(' ')[-16:]]).reshape(4,4)[:3,:3]
        cams[i] = cam_i

    return cams


def coord_transform_cam_to_GP(points, uvn):
    u, v, n = uvn                            
    transformed = np.zeros_like(points)
    transformed[...,0] = points @ u
    transformed[...,1] = points @ v
    transformed[...,2] = points @ n
    return transformed

def coord_transform_GP_to_cam(points, xyz):
    x, y, z = xyz
    transformed = np.zeros_like(points)
    transformed[...,0] = points @ x
    transformed[...,1] = points @ y
    transformed[...,2] = points @ z
    return transformed


## Compute minimum depths throughout receptive field (1,000 frames)
# def compute_min_depths(data):
#     depths = data[:,root_id,2]

#     Lind = []
#     temp_min_depths = []

#     for i, d in enumerate(depths):
#         while (Lind and Lind[0] <= i-frames_per_seq):
#             Lind.pop(0)
        
#         while (Lind and depths[Lind[-1]] >= d):
#             Lind.pop(-1)

#         Lind.append(i)
#         temp_min_depths.append(depths[Lind[0]])

#     min_depths = np.array(temp_min_depths)[(frames_per_seq-1):]

#     return min_depths


def make(augtype, log_out, gp_aware=True, occ_aware=True):
    assert augtype in ('aug1', 'aug2', 'aug3', 'aug4'), "Invalid augtype"

    num_subjects = 8
    num_people_candi = [5, 4] #[5, 4, 3, 2]


    ### PT (Default)
    person_trans_candi = [1.]
    ### PR
    theta_lim = 0
    ### GPT
    z_trans_candi = [0.] ## (mm)
    ### GPR
    gp_phi_rot_candi = [0]

    if augtype != 'aug1':
        theta_lim = np.pi / 4
        if augtype != 'aug2':
            z_trans_candi = [-1000., 0., 1500., 3000.] ## (mm)
            if augtype != 'aug3':
                gp_phi_rot_candi = [-np.pi/6, 0, np.pi/6]



    combination1 = [(f'S{i+1}', 'Seq1') for i in range(num_subjects)]
    combination2 = [(f'S{i+1}', 'Seq2') for i in range(num_subjects)]

    tot_frames = 0.84e6 #3.36e6
    frames_per_seq = 500
    tot_sequences = int(tot_frames / frames_per_seq)
    seq_per_comb = int(tot_sequences / len(num_people_candi) / len(z_trans_candi) / len(gp_phi_rot_candi) / len(person_trans_candi) / num_cam)

    test_set_ratio = 0.2 #0.2 #0.2
    test_seq_per_comb = int(seq_per_comb * test_set_ratio)
    train_seq_per_comb = seq_per_comb - test_seq_per_comb

    radius_of_3d_ball = 140 ## (mm)
    min_valid_depth = 500 #(mm)
    phi_margin = np.pi / 20

    res_w = 2048
    res_h = 2048

    root_id = 14
    max_num_people = 10


    ## Loading Annotation Files
    annot_files = {}
    for subject in [f'S{i+1}' for i in range(num_subjects)]:
        annot_files[subject] = {}
        for seq in ['Seq1','Seq2']:
            data = sio.loadmat(f'/data/dataset/3DHP/{subject}/{seq}/annot.mat')['annot3']
            annot_files[subject][seq] = data


    # ## Compute minimum depths throughout receptive field (1,000 frames)
    # min_depths = {}

    # for subj in [f'S{i+1}' for i in range(8)]:
    #     min_depths_subj = min_depths.setdefault(subj, {})
    #     for seq in ['Seq1', 'Seq2']:
    #         min_depths_seq = min_depths_subj.setdefault(seq, [])
    #         for c in range(14):
    #             data = (annot_files[subj][seq][c,0].reshape(-1,28,3)[:,[7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 4, 3, 6]])
    #             temp_min_depths = compute_min_depths(data)

    #             min_depths_seq.append(temp_min_depths)
    
    
    ## Loading informations about Ground Plane
    plane_data = np.load(f'data/data_3dhp_normal_vectors.npz', allow_pickle=True)
    normal_vectors = plane_data['normal_vectors'].item()
    ground_planes = plane_data['ground_planes'].item()

    ## Loading Camera Parameters
    cams = get_cam()

    seq_no = 0

    dataset_3d = {}
    dataset_2d = {}
    dataset_vis = {}
    dataset_obj = {}
    dataset_img_path = {}
    
    ## 20 seqs (20,000 frames) / options
    for num_people in num_people_candi:
        for z_trans in z_trans_candi:
            for phi_rot in gp_phi_rot_candi:
                for p_trans in person_trans_candi:
                    p_trans_ub = p_trans * 6000 #6000
                    p_trans_lb = p_trans * -6000 #-6000

                    seq_dataset_3d = []
                    seq_dataset_2d = []

                    seq_dataset_vis = []
                    seq_dataset_obj = []
                    seq_dataset_img_path = []

                    ts_seq_dataset_3d = []
                    ts_seq_dataset_2d = []

                    ts_seq_dataset_vis = []
                    ts_seq_dataset_obj = []
                    ts_seq_dataset_img_path = []

                    for c in range(num_cam):
                        cam = cams[c]
                        fx, fy, cx, cy = cam['K'][[0, 1, 0, 1],[0, 1, 2, 2]]
                        XperZ_ub = (cam['res_w']-1 - cx) / fx
                        XperZ_lb = -cx / fx
                        YperZ_ub = (cam['res_h']-1 - cy) / fy
                        YperZ_lb = -cy / fy

                        seq_cam_dataset_3d = []
                        seq_cam_dataset_2d = []

                        seq_cam_dataset_vis = []
                        seq_cam_dataset_obj = []
                        seq_cam_dataset_img_path = []

                        ts_seq_cam_dataset_3d = []
                        ts_seq_cam_dataset_2d = []

                        ts_seq_cam_dataset_vis = []
                        ts_seq_cam_dataset_obj = []
                        ts_seq_cam_dataset_img_path = []

                        s = 0
                        st = time.time()
                        while (s < seq_per_comb):
                            subjects = np.random.choice(num_subjects, size=num_people, replace=False)
                            seqs = np.random.choice(2, size=num_people, replace=True)

                            combs = [combination1[subject] if seq==0 else combination2[subject] for subject, seq in zip(subjects, seqs)]

                            kp3ds = []
                            for comb in combs:
                                kp3d = annot_files[comb[0]][comb[1]][c,0].reshape(-1,28,3)[:,[7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 4, 3, 6]]
                                kp3ds.append(kp3d)
                            
                            ## Construct U-V-N frame of the Ground Plane
                            if gp_aware:
                                ## It won't work for Aug3/4, it only works for Aug1/2
                                n = normal_vectors[c]
                            else:
                                n = np.random.randn(3)
                                n /= np.linalg.norm(n)

                            x = np.array([1., 0., 0.])
                            u = x - x@n * n
                            u /= np.linalg.norm(u)
                            v = np.cross(n, u)

                            
                            ## Rotation of the ground plane
                            # Compute the center point on the ground plane
                            if phi_rot != 0 or theta_lim != 0:
                                GP = ground_planes[c]
                                GP *= (n[0] / GP[0])
                                p = np.mean([kp3d.mean((0,1)) for kp3d in kp3ds], axis=0)
                                p_proj_GP = p - (n@p + GP[-1]) * n
                                t = p_proj_GP

                                cur_angle = np.arccos(np.dot(n, [0, 0, -1]))
                                phi_lb = -np.pi/2 + phi_margin + np.sign(n[1]) * cur_angle
                                phi_ub = np.pi/2 - phi_margin + np.sign(n[1]) * cur_angle
                                phi = np.clip(phi_rot, phi_lb, phi_ub)
                                theta = np.random.uniform(-theta_lim, theta_lim)
                                R1 = np.array([
                                    [1, 0, 0],
                                    [0, np.cos(phi), -np.sin(phi)],
                                    [0, np.sin(phi), np.cos(phi)],
                                ])
                                R2 = np.array([
                                    [np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1],
                                ])
                                R = R1 @ R2

                                rotated_points_uvns = [coord_transform_cam_to_GP(kp3d - t, [u, v, n]) @ R.T for kp3d in kp3ds]
                                xyz_in_uvn = coord_transform_cam_to_GP(np.eye(3), [u, v, n])
                                kp3ds = [coord_transform_GP_to_cam(rotated_points_uvn, xyz_in_uvn) + t for rotated_points_uvn in rotated_points_uvns]

                                rotated_uvn = np.eye(3) @ R.T @ [u, v, n]                       
                                u, v, n = rotated_uvn

                            
                            ## Crop each sequence
                            _kp3ds = []
                            for i, comb in enumerate(combs):
                                kp3d = kp3ds[i]
                                # if phi_rot != 0:
                                #     min_depths_comb = compute_min_depths(kp3d)
                                # else:
                                #     min_depths_comb = min_depths[comb[0]][comb[1]][c]
                                # valid_frame_idx = np.where(min_depths_comb + z_trans > min_valid_depth)[0]
                                # start_frame = np.random.choice(valid_frame_idx)

                                ### (23.01.10) It's okay to skip {compute_min_depths} as U,V Direction PT makes it feasible
                                start_frame = np.random.choice(len(kp3d) - frames_per_seq + 1)
                                _kp3ds.append(kp3d[start_frame:(start_frame + frames_per_seq)])
                            
                            if len(_kp3ds) != num_people:
                                print('[Z] FAIL!')
                                continue

                            kp3ds = np.stack(_kp3ds, axis=1)


                            ## z-direction GP translation
                            z_translation = z_trans*np.array([[0., 0., 1.]])
                            kp3ds += z_translation[None,:,None,:]

                            
                            ## Make a valid augmentation (Roots are all visible throught sequence)
                            Xs = kp3ds[:,:,root_id,0]
                            Ys = kp3ds[:,:,root_id,1]
                            Zs = kp3ds[:,:,root_id,2]

                            ## v-direction translation
                            coeff_inequalities = np.stack(([-v[2]], [YperZ_lb*v[2] - v[1]], [-YperZ_ub*v[2] + v[1]], [XperZ_lb*v[2] - v[0]], [-XperZ_ub*v[2] + v[0]]), axis=0).reshape(-1,1)
                            const_inequalities = np.stack(([Zs - min_valid_depth], [Ys - YperZ_lb*Zs], [-Ys + YperZ_ub*Zs], [Xs - XperZ_lb*Zs], [-Xs + XperZ_ub*Zs]), axis=0).reshape(5,-1).min(-1)
                            kv_min_res = linprog([1],
                                            A_ub=coeff_inequalities,
                                            b_ub=const_inequalities,
                                            bounds=(p_trans_lb,p_trans_ub))
                            kv_max_res = linprog([-1],
                                            A_ub=coeff_inequalities,
                                            b_ub=const_inequalities,
                                            bounds=(p_trans_lb,p_trans_ub))
                            if kv_min_res.success and kv_max_res.success and kv_min_res.fun <= -kv_max_res.fun:
                                kv_min, kv_max = int(kv_min_res.fun), int(-kv_max_res.fun)
                                mu = (kv_min + kv_max) / 2
                                sigma = (kv_max - kv_min) / 6
                                kv = np.clip(np.random.uniform(kv_min, kv_max, size=(num_people,1)), kv_min, kv_max)
                                # kv = np.clip(np.random.uniform(mu, sigma, size=(num_people,1)), kv_min, kv_max)
                                # print(f'[V] SUCCESS! / kv:{kv[0,0]:.3f} , min:{kv_min:.3f} , max:{kv_max:.3f}')
                            else:
                                # print(f'[V] MIN: {kv_min_res.fun if kv_min_res.success else 713:.3f}, MAX: {-kv_max_res.fun if kv_max_res.success else 713:.3f}')
                                print('[V] FAIL!')
                                continue

                            v_translation = kv * v[None,:]
                            kp3ds += v_translation[None,:,None,:]

                            ## u-direction translation
                            coeff_inequalities = np.stack(([-u[2]], [YperZ_lb*u[2] - u[1]], [-YperZ_ub*u[2] + u[1]], [XperZ_lb*u[2] - u[0]], [-XperZ_ub*u[2] + u[0]]), axis=0).reshape(-1,1)
                            const_inequalities = np.stack(([Zs - min_valid_depth], [Ys - YperZ_lb*Zs], [-Ys + YperZ_ub*Zs], [Xs - XperZ_lb*Zs], [-Xs + XperZ_ub*Zs]), axis=0).reshape(5,-1).min(-1)
                            ku_min_res = linprog([1],
                                            A_ub=coeff_inequalities,
                                            b_ub=const_inequalities,
                                            bounds=(p_trans_lb,p_trans_ub))
                            ku_max_res = linprog([-1],
                                            A_ub=coeff_inequalities,
                                            b_ub=const_inequalities,
                                            bounds=(p_trans_lb,p_trans_ub))
                            if ku_min_res.success and ku_max_res.success and ku_min_res.fun <= -ku_max_res.fun:
                                ku_min, ku_max = int(ku_min_res.fun), int(-ku_max_res.fun)
                                mu = (ku_min + ku_max) / 2
                                sigma = (ku_max - ku_min) / 6
                                ku = np.clip(np.random.uniform(ku_min, ku_max, size=(num_people,1)), ku_min, ku_max)
                                # ku = np.clip(np.random.uniform(mu, sigma, size=(num_people,1)), ku_min, ku_max)
                                # print(f'[U] SUCCESS! / ku:{ku[0,0]:.3f} , min:{ku_min:.3f} , max:{ku_max:.3f}')
                            else:
                                # print(f'[U] MIN: {ku_min_res.fun if ku_min_res.success else 713:.3f}, MAX: {-ku_max_res.fun if ku_max_res.success else 713:.3f}')
                                print('[U] FAIL')
                                continue                        

                            u_translation = ku * u[None,:]
                            kp3ds += u_translation[None,:,None,:]

                            
                            ##
                            kp3d = kp3ds

                            ## Generate 2D considering Occlusions
                            kp2d_h = kp3d @ (cam['K'].T)
                            kp2d = kp2d_h[...,:2] / kp2d_h[...,2:3]

                            F,N,K,_ = kp2d.shape
                            
                            inter_occluded = np.zeros(kp2d.shape[:-1], dtype='bool')
                            inter_occluded_front = np.zeros(kp2d.shape[:-1], dtype='bool')
                            self_occluded = np.zeros(kp2d.shape[:-1], dtype='bool')
                            invisible = np.zeros(kp2d.shape[:-1], dtype='bool')

                            for f, _kp3d_i in enumerate(kp3d):
                                ord = np.argsort(_kp3d_i.reshape(-1,3)[:,-1])
                                kp3d_i = _kp3d_i.reshape(-1,3)[ord]

                                for i in range(N*K):
                                    ni = ord[i] // K
                                    ki = ord[i] % K

                                    if (kp3d_i[i][2] < 0) \
                                        or (kp2d[f,ni,ki,0] < 0 or kp2d[f,ni,ki,0] > res_w - 1) \
                                        or (kp2d[f,ni,ki,1] < 0 and kp2d[f,ni,ki,1] > res_h - 1):
                                        invisible[f,ni,ki] = True
                                        continue

                                    if occ_aware:
                                        for j in range(i+1, N*K):
                                            nj = ord[j] // K
                                            kj = ord[j] % K
                                            dist = np.linalg.norm(kp2d[f,ni,ki] - kp2d[f,nj,kj])
                                            if dist < radius_of_3d_ball * np.sqrt(cam['K'][0,0] * cam['K'][1,1]) / kp3d_i[i][2]:
                                                if ni==nj:    
                                                    self_occluded[f,nj,kj] = True
                                                else:
                                                    inter_occluded[f,nj,kj] = True
                                                    inter_occluded_front[f,ni,ki] = True  ## Front position's person is alos affected by the occlusion


                            if invisible.sum() > 0:
                                if (invisible[:,:,root_id].std((0,1)) != 0) or (np.logical_not(invisible)[:,:,root_id].sum() == 0):  ### Num of people within sequence is not constant / is zero --> Discard
                                    continue                            
                            
                            
                            if occ_aware:
                                self_occluded *= np.logical_not(inter_occluded)
                                inter_occluded_front *= np.logical_not(inter_occluded)
                                
                                ## Perturb GT 2D KP with random gaussian noise
                                randvar = np.random.random(kp2d.shape[:-1])[...,None]
                                p = 0.2
                                degree_of_noise = 0.1  ## 0.1 x radius_of_3d_ball
                                noise = degree_of_noise * radius_of_3d_ball * np.sqrt(cam['K'][0,0] * cam['K'][1,1])/kp3d[...,2:] * np.random.multivariate_normal([0,0], [[1,0],[0,1]], kp2d.shape[:-1])
                                kp2d = kp2d + (randvar < p) * noise
                                
                                if self_occluded.sum() > 0:
                                    randvar = np.random.random(kp2d[self_occluded].shape[:-1])[...,None]
                                    p1 = 0.3
                                    p2 = 0.05
                                    var = -1.
                                    degree_of_noise = 0.2  ## 0.2 x radius_of_3d_ball
                                    noise = degree_of_noise * radius_of_3d_ball * np.sqrt(cam['K'][0,0] * cam['K'][1,1])/kp3d[self_occluded][...,2:] * np.random.multivariate_normal([0,0], [[1,0],[0,1]], kp2d[self_occluded].shape[:-1])
                                    kp2d[self_occluded] = (randvar < p2) * var + (randvar >= p2) * (kp2d[self_occluded] + (randvar < p1) * noise)
                                
                                # if inter_occluded_front.sum() > 0:
                                #     randvar = np.random.random(kp2d[inter_occluded_front].shape[:-1])[...,None]
                                #     p = 0.3
                                #     var = -1.
                                #     degree_of_noise = 0.5  ## 1.0 x radius_of_3d_ball
                                #     noise = degree_of_noise * radius_of_3d_ball * np.sqrt(cam['K'][0,0] * cam['K'][1,1])/kp3d[inter_occluded_front][...,2:] * np.random.multivariate_normal([0,0], [[1,0],[0,1]], kp2d[inter_occluded_front].shape[:-1])
                                #     kp2d[inter_occluded_front] = (randvar < p) * var + (randvar >= p) * (kp2d[inter_occluded_front] + noise)
                                
                                if inter_occluded.sum() > 0:
                                    randvar = np.random.random(kp2d[inter_occluded].shape[:-1])[...,None]
                                    p = 0.5
                                    var = -1.
                                    degree_of_noise = 1.  ## 1.0 x radius_of_3d_ball
                                    noise = degree_of_noise * radius_of_3d_ball * np.sqrt(cam['K'][0,0] * cam['K'][1,1])/kp3d[inter_occluded][...,2:] * np.random.multivariate_normal([0,0], [[1,0],[0,1]], kp2d[inter_occluded].shape[:-1])
                                    kp2d[inter_occluded] = (randvar < p) * var + (randvar >= p) * (kp2d[inter_occluded] + noise)

                            ## Should be eliminated for VirtualPose
                            if invisible.sum() > 0:
                                var = -1.
                                kp2d[invisible] = var

                            if (s+1) % 1000 == 0:
                                et = time.time()
                                log(log_out, f"N: {num_people} / z_t: {z_trans} / rot_phi: {phi_rot/np.pi:.2f}PI / cam: {c} --- NUM PERSONS: {num_people - (invisible[0,:,root_id]).sum()} --- NUM INTER OCCLUDED: {inter_occluded.sum():d} / NUM SELF OCCLUDED: {self_occluded.sum():d} / NUM INVISIBLE: {invisible.sum():d}")
                                log(log_out, f"[Phi] {(phi if phi_rot !=0 else 0)/np.pi:.3f} / [V] {kv.mean():.3f} ({kv_min:.3f}~{kv_max:.3f}) / [U] {ku.mean():.3f} ({ku_min:.3f}~{ku_max:.3f})")
                                log(log_out, f"\tTIME SPENT: {et-st:.5f}SEC")
                                st = time.time()


                            kp3d_padded = np.zeros((F,max_num_people,K,3), dtype='float32')
                            kp2d_padded = np.zeros((F,max_num_people,K,2), dtype='float32')
                            vis_padded = np.zeros((F,max_num_people,K,), dtype='bool')
                            obj_padded = np.zeros((F,max_num_people,), dtype='bool')

                            kp3d_padded[:,:N] = kp3d
                            kp2d_padded[:,:N] = kp2d
                            vis_padded[:,:N] = np.logical_not(invisible)
                            obj_padded[:,:N] = np.logical_not(invisible)[:,:,0]
                            img_path = [sorted(glob.glob(f'/data/dataset/3DHP/{combs[0][0]}/{combs[0][1]}/imageSequence/video_{c}/*.jpg'))[0]] * F

                            
                            if s < train_seq_per_comb:
                                seq_cam_dataset_3d.append(kp3d_padded)
                                seq_cam_dataset_2d.append(kp2d_padded)
                                seq_cam_dataset_vis.append(vis_padded)
                                seq_cam_dataset_obj.append(obj_padded)
                                seq_cam_dataset_img_path.append(img_path)
                            else:
                                ts_seq_cam_dataset_3d.append(kp3d_padded)
                                ts_seq_cam_dataset_2d.append(kp2d_padded)
                                ts_seq_cam_dataset_vis.append(vis_padded)
                                ts_seq_cam_dataset_obj.append(obj_padded)
                                ts_seq_cam_dataset_img_path.append(img_path)

                            s += 1

                        seq_dataset_3d.append(seq_cam_dataset_3d)
                        seq_dataset_2d.append(seq_cam_dataset_2d)
                        seq_dataset_vis.append(seq_cam_dataset_vis)
                        seq_dataset_obj.append(seq_cam_dataset_obj)
                        seq_dataset_img_path.append(seq_cam_dataset_img_path)

                        ts_seq_dataset_3d.append(ts_seq_cam_dataset_3d)
                        ts_seq_dataset_2d.append(ts_seq_cam_dataset_2d)
                        ts_seq_dataset_vis.append(ts_seq_cam_dataset_vis)
                        ts_seq_dataset_obj.append(ts_seq_cam_dataset_obj)
                        ts_seq_dataset_img_path.append(ts_seq_cam_dataset_img_path)
                    
                    dataset_3d[str(seq_no+1)] = seq_dataset_3d
                    dataset_2d[str(seq_no+1)] = seq_dataset_2d
                    dataset_vis[str(seq_no+1)] = seq_dataset_vis
                    dataset_obj[str(seq_no+1)] = seq_dataset_obj
                    dataset_img_path[str(seq_no+1)] = seq_dataset_img_path

                    dataset_3d[f'TS{seq_no+1}'] = ts_seq_dataset_3d
                    dataset_2d[f'TS{seq_no+1}'] = ts_seq_dataset_2d
                    dataset_vis[f'TS{seq_no+1}'] = ts_seq_dataset_vis
                    dataset_obj[f'TS{seq_no+1}'] = ts_seq_dataset_obj
                    dataset_img_path[f'TS{seq_no+1}'] = ts_seq_dataset_img_path

                    seq_no += 1

                    np.savez_compressed(f'data/data_3d_vid3dhp_{augtype}_img_1_3M_230215_Abl_GP_{"Y" if gp_aware else "N"}_OCC_{"Y" if occ_aware else "N"}.npz', positions_3d=dataset_3d)
                    np.savez_compressed(f'data/data_2d_vid3dhp_{augtype}_img_1_3M_230215_Abl_GP_{"Y" if gp_aware else "N"}_OCC_{"Y" if occ_aware else "N"}.npz', positions_2d=dataset_2d, vis=dataset_vis, obj=dataset_obj, img_path=dataset_img_path)

    
    np.savez_compressed(f'data/data_3d_vid3dhp_{augtype}_img_1_3M_230215_Abl_GP_{"Y" if gp_aware else "N"}_OCC_{"Y" if occ_aware else "N"}.npz', positions_3d=dataset_3d)
    np.savez_compressed(f'data/data_2d_vid3dhp_{augtype}_img_1_3M_230215_Abl_GP_{"Y" if gp_aware else "N"}_OCC_{"Y" if occ_aware else "N"}.npz', positions_2d=dataset_2d, vis=dataset_vis, obj=dataset_obj, img_path=dataset_img_path)



if __name__ == "__main__":
    mode = sys.argv[1]
    
    if mode=='viz':
        seq = sys.argv[2]
        log_out = sys.argv[3]

        render(seq, log_out)

    elif mode=='make':
        augtype = sys.argv[2]
        log_out = sys.argv[3]
        gp_aware = eval(sys.argv[4])
        occ_aware = eval(sys.argv[5])
        print("Bool_True" if gp_aware else "Bool_False", "Bool_True" if occ_aware else "Bool_False")
        make(augtype, log_out, gp_aware=gp_aware, occ_aware=occ_aware)

