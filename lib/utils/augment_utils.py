import numpy as np
import glob
import os
import math
import time
import scipy.io as sio
from scipy.optimize import linprog
import torch

from lib.utils.utils import *
from lib.camera.camera import project_to_2d_linear



def fit_ground_plane(kp3ds, lr=1e-3, num_iter=10000):
    kp3ds = torch.from_numpy(kp3ds)

    ground_plane = torch.nn.Parameter(torch.empty(4).normal_(mean=0, std=1))
    optimizer = torch.optim.AdamW([ground_plane], lr=lr)

    best_loss = 1000
    best_ground_plane = None
    best_fitting_loss = 1000
    best_norm_loss = 1000

    for i in range(num_iter):
        optimizer.zero_grad()
        
        fitting_loss = torch.abs(kp3ds @ ground_plane / torch.norm(ground_plane)).mean()
        norm_loss = torch.abs(torch.norm(ground_plane) - 1)
        loss = fitting_loss + norm_loss

        loss.backward()
        optimizer.step()

        cur_loss = loss.item()
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_fitting_loss = fitting_loss
            best_norm_loss = norm_loss
            best_ground_plane = ground_plane.detach().cpu().numpy()
    
    return best_ground_plane, best_loss, best_fitting_loss, best_norm_loss


## Compute minimum depths throughout receptive field (L frames)
def compute_min_depths(data_depths, L):
    depths = data_depths ## (F, 1)

    Lind = []
    temp_min_depths = []

    for i, d in enumerate(depths):
        while (Lind and Lind[0] <= i-L):
            Lind.pop(0)
        
        while (Lind and depths[Lind[-1]] >= d):
            Lind.pop(-1)

        Lind.append(i)
        temp_min_depths.append(depths[Lind[0]])

    min_depths = np.array(temp_min_depths)[(L-1):]

    return min_depths


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


def augmentation(kp3ds, cam, num_people, root_id, ground_plane, normal, PT_lim, PR_lim, GPT_trans, GPR_rot, cfg):
    success = False
    meta = None

    kp3ds *= 1000 ## meters to mm

    radius_of_3d_ball = cfg.AUG.RADIUS_OF_3D_BALL ## (mm)
    min_valid_depth = cfg.AUG.MIN_VALID_DEPTH ## (mm)
    GPR_rot_margin = np.pi * cfg.AUG.GPR_MARGIN

    invalid_value = cfg.AUG.INVALID_VALUE

    default_noise_r = cfg.AUG.DEFAULT_NOISE_LEVEL
    default_noise_p = cfg.AUG.DEFAULT_NOISE_PROB
    self_occlusion_noise_r = cfg.AUG.SELF_OCCLUSION_NOISE_LEVEL
    self_occlusion_noise_p = cfg.AUG.SELF_OCCLUSION_NOISE_PROB
    self_occlusion_p_d = cfg.AUG.SELF_OCCLUSION_DROPOUT_PROB
    inter_occlusion_noise_r = cfg.AUG.INTER_OCCLUSION_NOISE_LEVEL
    inter_occlusion_noise_p = cfg.AUG.INTER_OCCLUSION_NOISE_PROB
    inter_occlusion_p_d = cfg.AUG.INTER_OCCLUSION_DROPOUT_PROB

    noise_vars = [default_noise_r, default_noise_p, self_occlusion_noise_r, self_occlusion_noise_p, self_occlusion_p_d, 
                inter_occlusion_noise_r, inter_occlusion_noise_p, inter_occlusion_p_d]

    ## U-V-N (GP Coord Basis) / X-Y-Z (Cam Coord Basis)
    x = np.array([1., 0., 0.])
    u = x - x@normal * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    
    fx, fy, cx, cy = cam['intrinsic'][:4]
    XperZ_ub = (cam['res_w']-1 - cx) / fx
    XperZ_lb = -cx / fx
    YperZ_ub = (cam['res_h']-1 - cy) / fy
    YperZ_lb = -cy / fy

    ## PR + GPR
    if PR_lim != 0 or GPR_rot != 0:
        ## Compute the center point on the ground plane
        p = np.mean([kp3d.mean((0,1)) for kp3d in kp3ds], axis=0)
        p_proj_GP = p - (normal@p + ground_plane[-1] * 1000) * normal ## ground plane[-1] ; meter to mm
        t = p_proj_GP

        cur_angle = np.arccos(np.dot(normal, [0, 0, -1]))
        phi_lb = -np.pi/2 + GPR_rot_margin + np.sign(normal[1]) * cur_angle
        phi_ub = np.pi/2 - GPR_rot_margin + np.sign(normal[1]) * cur_angle
        phi = np.clip(GPR_rot, phi_lb, phi_ub)
        theta = np.random.uniform(-PR_lim, PR_lim)
        ## GPR
        R1 = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)],
        ])
        ## PR
        R2 = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ])
        R = R1 @ R2
    
        rotated_points_uvns = [coord_transform_cam_to_GP(kp3d - t, [u, v, normal]) @ R.T for kp3d in kp3ds]
        xyz_in_uvn = coord_transform_cam_to_GP(np.eye(3), [u, v, normal])
        kp3ds = [coord_transform_GP_to_cam(rotated_points_uvn, xyz_in_uvn) + t for rotated_points_uvn in rotated_points_uvns]

        rotated_uvn = np.eye(3) @ R.T @ [u, v, normal]                       
        u, v, normal = rotated_uvn

    ## z-direction GPT
    z_translation = GPT_trans*np.array([[0., 0., 1.]])
    kp3ds += z_translation[None,:,None,:]

    ## Make a valid augmentation (Roots should be all visible throughout sequence)
    Xs = kp3ds[:,:,root_id,0]
    Ys = kp3ds[:,:,root_id,1]
    Zs = kp3ds[:,:,root_id,2]

    ## v-direction PT
    coeff_inequalities = np.stack(([-v[2]], [YperZ_lb*v[2] - v[1]], [-YperZ_ub*v[2] + v[1]], [XperZ_lb*v[2] - v[0]], [-XperZ_ub*v[2] + v[0]]), axis=0).reshape(-1,1)
    const_inequalities = np.stack(([Zs - min_valid_depth], [Ys - YperZ_lb*Zs], [-Ys + YperZ_ub*Zs], [Xs - XperZ_lb*Zs], [-Xs + XperZ_ub*Zs]), axis=0).reshape(5,-1).min(-1)
    kv_min_res = linprog([1],
                    A_ub=coeff_inequalities,
                    b_ub=const_inequalities,
                    bounds=(-PT_lim, PT_lim))
    kv_max_res = linprog([-1],
                    A_ub=coeff_inequalities,
                    b_ub=const_inequalities,
                    bounds=(-PT_lim, PT_lim))
    if kv_min_res.success and kv_max_res.success and kv_min_res.fun <= -kv_max_res.fun:
        kv_min, kv_max = int(kv_min_res.fun), int(-kv_max_res.fun)
        mu = (kv_min + kv_max) / 2
        sigma = (kv_max - kv_min) / 6
        kv = np.clip(np.random.normal(mu, sigma, size=(num_people, 1)), kv_min, kv_max)
    else:
        return success, meta

    v_translation = kv * v[None,:]
    kp3ds += v_translation[None,:,None,:]

    ## u-direction PT
    coeff_inequalities = np.stack(([-u[2]], [YperZ_lb*u[2] - u[1]], [-YperZ_ub*u[2] + u[1]], [XperZ_lb*u[2] - u[0]], [-XperZ_ub*u[2] + u[0]]), axis=0).reshape(-1,1)
    const_inequalities = np.stack(([Zs - min_valid_depth], [Ys - YperZ_lb*Zs], [-Ys + YperZ_ub*Zs], [Xs - XperZ_lb*Zs], [-Xs + XperZ_ub*Zs]), axis=0).reshape(5,-1).min(-1)
    ku_min_res = linprog([1],
                    A_ub=coeff_inequalities,
                    b_ub=const_inequalities,
                    bounds=(-PT_lim, PT_lim))
    ku_max_res = linprog([-1],
                    A_ub=coeff_inequalities,
                    b_ub=const_inequalities,
                    bounds=(-PT_lim, PT_lim))
    if ku_min_res.success and ku_max_res.success and ku_min_res.fun <= -ku_max_res.fun:
        ku_min, ku_max = int(ku_min_res.fun), int(-ku_max_res.fun)
        mu = (ku_min + ku_max) / 2
        sigma = (ku_max - ku_min) / 6
        ku = np.clip(np.random.normal(mu, sigma, size=(num_people,1)), ku_min, ku_max)
    else:
        return success, meta                        

    u_translation = ku * u[None,:]
    kp3ds += u_translation[None,:,None,:]


    ## Generate 2D considering Occlusions
    # kp2d_h = kp3ds @ (cam['K'].T)
    # kp2d = kp2d_h[...,:2] / kp2d_h[...,2:]
    kp2d = project_to_2d_linear(kp3ds, np.tile(cam['intrinsic'][:4], (kp3ds.shape[0],1)))

    F,N,K,_ = kp2d.shape
    
    inter_occluded = np.zeros(kp2d.shape[:-1], dtype='bool')
    self_occluded = np.zeros(kp2d.shape[:-1], dtype='bool')
    invisible = np.zeros(kp2d.shape[:-1], dtype='bool')

    for f, _kp3d in enumerate(kp3ds):
        ord = np.argsort(_kp3d.reshape(-1,3)[:,-1])
        kp3d = _kp3d.reshape(-1,3)[ord]

        for i in range(N*K):
            ni = ord[i] // K
            ki = ord[i] % K

            if (kp3d[i][2] < 0) \
                or (kp2d[f,ni,ki,0] < 0 or kp2d[f,ni,ki,0] > cam['res_w'] - 1) \
                or (kp2d[f,ni,ki,1] < 0 and kp2d[f,ni,ki,1] > cam['res_h'] - 1):
                invisible[f,ni,ki] = True
                continue

            for j in range(i+1, N*K):
                nj = ord[j] // K
                kj = ord[j] % K
                dist = np.linalg.norm(kp2d[f,ni,ki] - kp2d[f,nj,kj])
                if dist < radius_of_3d_ball * np.sqrt(np.prod(cam['intrinsic'][:2])) / kp3d[i][2]:
                    if ni==nj:    
                        self_occluded[f,nj,kj] = True
                    else:
                        inter_occluded[f,nj,kj] = True


    if invisible.sum() > 0:
        if (invisible[:,:,root_id].std((0,1)) != 0) or (np.logical_not(invisible)[:,:,root_id].sum() == 0):  ### Num of people within sequence is not constant / is zero --> Discard
            return success, meta                           


    self_occluded *= np.logical_not(inter_occluded)
    
    ## Perturb GT 2D KP with random gaussian noise
    randvar = np.random.random(kp2d.shape[:-1])[...,None]
    degree_of_noise = noise_vars[0]  ## r x radius_of_3d_ball
    p = noise_vars[1]
    noise = degree_of_noise * radius_of_3d_ball * np.sqrt(np.prod(cam['intrinsic'][:2]))/kp3ds[...,2:] * np.random.multivariate_normal([0,0], [[1,0],[0,1]], kp2d.shape[:-1])
    kp2d = kp2d + (randvar < p) * noise
    
    ## Perturb Occluded KP
    if self_occluded.sum() > 0:
        randvar = np.random.random(kp2d[self_occluded].shape[:-1])[...,None]
        degree_of_noise = noise_vars[2]  ## r x radius_of_3d_ball
        p1 = noise_vars[3]
        p2 = noise_vars[4]
        noise = degree_of_noise * radius_of_3d_ball * np.sqrt(np.prod(cam['intrinsic'][:2]))/kp3ds[self_occluded][...,2:] * np.random.multivariate_normal([0,0], [[1,0],[0,1]], kp2d[self_occluded].shape[:-1])
        kp2d[self_occluded] = (randvar < p2) * invalid_value + (randvar >= p2) * (kp2d[self_occluded] + (randvar < p1) * noise)

    if inter_occluded.sum() > 0:
        randvar = np.random.random(kp2d[inter_occluded].shape[:-1])[...,None]
        degree_of_noise = noise_vars[5]  ## r x radius_of_3d_ball
        p1 = noise_vars[6]
        p2 = noise_vars[7]
        noise = degree_of_noise * radius_of_3d_ball * np.sqrt(np.prod(cam['intrinsic'][:2]))/kp3ds[inter_occluded][...,2:] * np.random.multivariate_normal([0,0], [[1,0],[0,1]], kp2d[inter_occluded].shape[:-1])
        kp2d[inter_occluded] = (randvar < p2) * invalid_value + (randvar >= p2) * (kp2d[inter_occluded] + (randvar < p1) * noise)

    if invisible.sum() > 0:
        kp2d[invisible] = invalid_value


    success = True
    meta = {
        'positions_3d': kp3ds / 1000, ## mm to meters
        'positions_2d': kp2d,
        'vis': np.logical_not(invisible),

        'num_invisible': invisible.sum(),
        'num_self_occluded': self_occluded.sum(),
        'num_inter_occluded': inter_occluded.sum(),
        'PT_trans_v': kv.mean(),
        'PT_trans_v_val_range': [kv_min, kv_max],
        'PT_trans_u': ku.mean(),
        'PT_trans_u_val_range': [ku_min, ku_max],
        'GPR-eff': phi if GPR_rot !=0 else 0
    }

    return success, meta