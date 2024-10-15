import os
import numpy as np
import matplotlib

matplotlib.use('Agg')


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import subprocess as sp
import cv2


np.random.seed(2234)
_COLORS = list(np.random.choice(range(256), size=20*3).reshape(20,3) / 255.)



def render_img(cfg, ax, poses, skeleton, mode='2d', colormode='individual', img_path=None, center=None):
    """
    Render poses (and image if required) on given ax as an image
    poses : (N, K, C)
    """
    assert mode in ('2d', '3d'), "Incompatible colormode"
    assert colormode in ('individal', 'leftright'), "Incompatible colormode"
    assert poses.shape[1] <= len(_COLORS), f"Maximum number of people should be less than {len(_COLORS)}"
    
    label_size = 15
    point_size = 10
    line_size = 10

    invalid_value = cfg.DATASET.INVALID_VALUE

    assets = {}
    assets['ax'] = ax
    assets['image'] = None
    assets['lines'] = []
    assets['points'] = []
    assets['labels'] = []
    
    if mode=='2d':
        point_colors = np.full(poses.shape[1], 'black')
        point_colors[skeleton['joints_right']] = 'red'

        if img_path is not None:
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img = ax.imshow(image)
            assets['image'] = img
        for p, pose in enumerate(poses):
            _lines = []
            for j, j_parent in enumerate(skeleton['parents']):
                _lines.append([])
                if j_parent != -1:
                    if (pose[[j, j_parent], 0] == invalid_value).any():
                        continue

                    if colormode == 'individual':
                        col = _COLORS[p]
                    else:
                        col = 'red' if j in skeleton['joints_right'] else 'black'
                    _lines[-1] = ax.plot([int(pose[j,0]), int(pose[j_parent,0])],
                                        [int(pose[j,1]), int(pose[j_parent,1])], color=col)
                    
            assets['lines'].append(_lines)

            joint_mask = pose[:,0] != invalid_value
            if joint_mask.sum() > 0:
                tx, ty = pose[joint_mask].min(0)
                assets['labels'].append(ax.text(tx, ty, str(p), color=_COLORS[p], size=label_size))
                assets['points'].append(ax.scatter(*pose[joint_mask].T, point_size, color=point_colors[joint_mask], edgecolors='white', zorder=10))
            else:
                assets['labels'].append(None)
                assets['points'].append(None)

    else:
        center = poses[:,cfg.DATASET.ROOTIDX].mean(0) if center is None else center
        radius = cfg.RADIUS_OF_3D_PLOT

        ax.set_xlim3d([-radius/2 + center[0], radius/2 + center[0]])
        ax.set_zlim3d([radius/2 + center[1], -radius/2 + center[1]])
        ax.set_ylim3d([-radius/2 + center[2], radius/2 + center[2]])

        for p, pose in enumerate(poses):
            _lines = []
            for j, j_parent in enumerate(skeleton['parents']):
                _lines.append([])
                if j_parent != -1:
                    if colormode == 'individual':
                        col = _COLORS[p]
                    else:
                        col = 'red' if j in skeleton['joints_right'] else 'black'
                    _lines[-1] = ax.plot([pose[j,0], pose[j_parent,0]],
                                        [pose[j,1], pose[j_parent,1]],
                                        [pose[j,2], pose[j_parent,2]], zdir='y', color=col)

            tx, ty, tz = pose.min(0)
            
            assets['lines'].append(_lines)
            assets['labels'].append(ax.text(tx, tz, ty, str(p), color=_COLORS[p], size=label_size, zdir='x'))
            assets['center'] = center
            
    return assets


## data --> (F,N,K,C)
def update_img(cfg, assets, poses, skeleton, mode='2d', colormode='individual', img_path=None, center=None):
    """
    Render poses (and image sequence if required) on given ax as a video
    poses : (N, K, C)
    """
    assert mode in ('2d', '3d'), "Incompatible colormode"
    assert colormode in ('individal', 'leftright'), "Incompatible colormode"
    assert poses.shape[1] <= len(_COLORS), f"Maximum number of people should be less than {len(_COLORS)}"

    label_size = 15
    point_size = 10
    line_size = 10

    invalid_value = cfg.DATASET.INVALID_VALUE
    
    if mode=='2d':
        point_colors = np.full(poses.shape[1], 'black')
        point_colors[skeleton['joints_right']] = 'red'

        if img_path is not None:
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            assets['image'].set_data(image)
        for p, pose in enumerate(poses):
            for j, j_parent in enumerate(skeleton['parents']):
                if j_parent != -1:
                    if (pose[[j, j_parent], 0] == invalid_value).any():
                        if len(assets['lines'][p][j]) > 0:
                            assets['lines'][p][j][0].remove()
                        assets['lines'][p][j] = []
                        continue

                    if len(assets['lines'][p][j]) > 0:
                        assets['lines'][p][j][0].set_xdata(np.array([pose[j, 0], pose[j_parent, 0]]))
                        assets['lines'][p][j][0].set_ydata(np.array([pose[j, 1], pose[j_parent, 1]]))
                    else:
                        if colormode == 'individual':
                            col = _COLORS[p]
                        else:
                            col = 'red' if j in skeleton['joints_right'] else 'black'
                        assets['lines'][p][j] = assets['ax'].plot([int(pose[j,0]),int(pose[j_parent,0])], [int(pose[j,1]),int(pose[j_parent,1])], color=col)
                    
            assets['labels'][p].remove()
            assets['labels'][p] = None
            assets['points'][p].remove()
            assets['points'][p] = None

            joint_mask = pose[:,0] != invalid_value
            tx, ty = pose[joint_mask].min(0)
            if joint_mask.sum() > 0:
                assets['labels'][p] = assets['ax'].text(tx, ty, str(p), color=_COLORS[p], size=label_size)
                assets['points'][p] = assets['ax'].scatter(*pose[joint_mask].T, point_size, color=point_colors[joint_mask], edgecolors='white', zorder=10)

    else:
        center = poses[:,cfg.DATASET.ROOTIDX].mean(0) if center is None else center
        radius = cfg.RADIUS_OF_3D_PLOT

        assets['ax'].set_xlim3d([-radius/2 + center[0], radius/2 + center[0]])
        assets['ax'].set_zlim3d([radius/2 + center[1], -radius/2 + center[1]])
        assets['ax'].set_ylim3d([-radius/2 + center[2], radius/2 + center[2]])

        for p, pose in enumerate(poses):
            for j, j_parent in enumerate(skeleton['parents']):
                if j_parent != -1:
                    if colormode == 'individual':
                        col = _COLORS[p]
                    else:
                        col = 'red' if j in skeleton['joints_right'] else 'green'
                    assets['lines'][p][j][0].set_xdata(np.array([pose[j, 0], pose[j_parent, 0]]))
                    assets['lines'][p][j][0].set_ydata(np.array([pose[j, 1], pose[j_parent, 1]]))
                    assets['lines'][p][j][0].set_3d_properties(np.array([pose[j, 2], pose[j_parent, 2]]), zdir='y')

            tx, ty, tz = pose.min(0)
            assets['labels'][p].set_x(tx)
            assets['labels'][p].set_y(tz)
            assets['labels'][p].set_3d_properties(ty, zdir='x')

        assets['center'] = center