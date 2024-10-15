# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch

from lib.utils.utils import wrap
from lib.camera.quaternion import qrot, qinverse


def normalize_screen_coordinates(X, w, h): 
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    assert X.shape[-1] == 2

    if isinstance(w, torch.Tensor):
        ones = torch.ones_like(w)
        return X/w*2 - torch.cat((ones, h/w), dim=-1)
    else:
        
        return X/w*2 - [1, h/w]  
    
def image_coordinates(X, w, h):
    # Reverse camera frame normalization
    assert X.shape[-1] == 2

    if isinstance(w, torch.Tensor):
        ones = torch.ones_like(w)
        return (X + torch.cat((ones, h/w), dim=-1))*w/2
    else:
        return (X + [1, h/w])*w/2    
    

def camera_to_world(X, R, t):
    Rt = wrap(qinverse, R) # Invert rotation
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t) # Rotate and translate

    
def world_to_camera(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t


def convert_to_abs_camcoord(X, root_id, camera_params):
    """
    Unnormalize root joint's coordinate,
    and recover target keypoints in absoulte camera coordinate.
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] in (7, 12)    ### (fx, fy, cx, cy, res_w, res_h, norm_factor) or (fx, fy, cx, cy, k1, k2, k3, p2, p1, res_w, res_h, norm_factor)
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)
    
    intrinsics = camera_params[...,:4]
    res_w = camera_params[...,-3:-2]
    res_h = camera_params[...,-2:-1]
    norm_factor = camera_params[...,-1:]

    X_root = X[...,root_id:(root_id+1),:].clone()
    X_root[...,2:] = X_root[...,2:].clone() * norm_factor
    X_root[...,:2] = (image_coordinates(X_root[...,:2].clone(), res_w, res_h) - intrinsics[...,2:]) * X_root[...,2:].clone() / intrinsics[...,:2]

    X[...,root_id,:] = 0
    X = X + X_root

    return X

    
def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9  ### (fx, fy, cx, cy, k1, k2, k3, p2, p1)
    assert X.shape[0] == camera_params.shape[0]
    
    convert_dtype = False
    if isinstance(X, np.ndarray):
        convert_dtype = True
        X = torch.from_numpy(X)
    if isinstance(camera_params, np.ndarray):
        camera_params = torch.from_numpy(camera_params)

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)
        
    ### cf. https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    f = camera_params[..., :2]   ### [fx, fy]
    c = camera_params[..., 2:4]   ### [cx, cy]
    k = camera_params[..., 4:7]  ### [k1, k2, k3]
    p = camera_params[..., 7:]  ### [p2, p1] 
    
    X_h = X[..., :2] / X[..., 2:]
    r2 = torch.sum(X_h[..., :2]**2, dim=-1, keepdim=True)

    radial = 1 + torch.sum(k * torch.cat((r2, r2**2, r2**3), dim=-1), dim=-1, keepdim=True)
    tan = 2 * torch.sum(p*X_h, dim=-1, keepdim=True)

    X_corr = X_h*(radial + tan) + 2*p*r2
    X_proj = f*X_corr + c

    if convert_dtype:
        X_proj = X_proj.numpy()
    
    return X_proj

def project_to_2d_linear(X, camera_params):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 4
    assert X.shape[0] == camera_params.shape[0]

    convert_dtype = False
    if isinstance(X, np.ndarray):
        convert_dtype = True
        X = torch.from_numpy(X)
    if isinstance(camera_params, np.ndarray):
        camera_params = torch.from_numpy(camera_params)
    
    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)
        
    f = camera_params[..., :2]   ### [fx, fy]
    c = camera_params[..., 2:4]   ### [cx, cy]
    
    X_h = X[..., :2] / X[..., 2:]
    X_proj = f*X_h + c

    if convert_dtype:
        X_proj = X_proj.numpy()
    
    return X_proj