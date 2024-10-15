import torch
import numpy as np

from lib.camera.camera import image_coordinates


def L2_loss(preds, target, vis, obj, root_id=14, root_weight=0.3, root_relative=False, include_invisible=False, mode='mpjpe'):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert preds.shape == target.shape

    B, F, N, K, C = preds.shape

    weights = torch.Tensor([1/K] * K)
    weights[root_id] = root_weight

    vis = vis.clone()
    if include_invisible:
        vis = torch.logical_or(vis, vis.any(-1, keepdims=True))
    if root_relative:
        preds = preds - preds[...,root_id:(root_id+1),:]
        target = target - target[...,root_id:(root_id+1),:]
        vis[...,root_id] = 0

    if mode == 'mpjve':
        preds = preds[:,1:] - preds[:,:-1]
        target = target[:,1:] - target[:,:-1]
        vis = vis[:,1:] * vis[:,:-1]

    error = 0
    count = vis.sum()

    for b in range(B):
        mask = obj[b]
        error += torch.sum(torch.norm(preds[b,:,mask] - target[b,:,mask], dim=-1) * vis[b,:,mask])

    return error/count, count
