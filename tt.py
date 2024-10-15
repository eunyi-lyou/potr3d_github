import numpy as np
import sys
import os, glob
from pathlib import Path
import scipy.io as sio
import pickle
from easydict import EasyDict as edict
from lib.utils.vis_utils import *


config = edict()
config.RADIUS_OF_3D_PLOT = 4.
config.DATASET = edict()
config.DATASET.INVALID_VALUE = -1.
config.DATASET.ROOTIDX = 14

bev_skeleton = {
    'parents': [1,2,14,14,3,4,7,8,12,12,9,10,15,16,-1,14,12],
    'joints_left': [3,4,5,9,10,11],
    'joints_right': [0,1,2,6,7,8],
    'symmetry': [[3,4,5,9,10,11], [0,1,2,6,7,8]],
    'rootidx': 14
}


mupots_skeleton = {
    'parents': [16, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, -1, 14, 1],
    'joints_left': [5, 6, 7, 11, 12, 13],
    'joints_right': [2, 3, 4, 8, 9, 10],
    'symmetry': [[5,6,7,11,12,13], [2,3,4,8,9,10]],
    'rootidx': 14
}
fxs = [1500.9799492788811, 1500.8605847627498, 1493.0169231973466, 1492.4964211490396, 1494.751093087024, 1132.8454726004886, 1103.8105659586838, 1123.8970261035088, 1103.5365554667687, 1103.802134740414, 1114.3884708175303, 1113.6840903858147, 1102.1600498245136, 1124.119473712715, 1100.8391573135282, 1137.9504563337387, 1137.8203133580198, 1191.8365301432216, 1103.689283995902, 1191.8378621424363]
fys = [1495.9003438753227, 1495.8994125542592, 1495.7220938482521, 1490.0239074933459, 1498.4098509691555, 1132.5149281839203, 1101.450236791744, 1121.6128547290284, 1101.450354911498, 1101.4509428236727, 1112.8719381062135, 1114.1954171009895, 1103.2677652946732, 1121.6112076685376, 1101.978920346303, 1138.332582310877, 1138.3319658637479, 1190.9720764607541, 1103.2831108425546, 1190.9735951379007]
cxs = [1030.7205375378683, 1030.7210310766873, 1021.9302353498579, 1013.9198555554226, 996.1297923326164, 933.8571053118252, 942.2364413741751, 953.0105789807833, 942.105026757321, 941.9484164760886, 965.7885096353243, 924.1013058194874, 984.1041281827166, 952.831311322105, 964.8252264269842, 945.7841549055345, 945.7886150698768, 907.0356835221816, 980.0625785655135, 907.0354654383593]
cys = [1045.5236081955522, 1045.5235756917198, 1069.9068786386795, 987.332492684269, 1009.3103287004742, 576.3980582579932, 528.3335315403131, 521.4528992516214, 528.3335272771702, 528.3336052695937, 497.12171949365967, 513.2677519265864, 507.5564578849272, 521.4525247780649, 503.46923142354683, 511.14931758721417, 511.14915322116315, 579.7367279637436, 499.8267111030015, 579.7371596804576]
norm_factor = [focal_length / 2048 * 2 if i < 5 else focal_length / 1920 * 2 for i, focal_length in enumerate(fxs)]
res_ws = [2048 if i < 5 else 1920 for i in range(20)]
res_hs = [2048 if i < 5 else 1080 for i in range(20)]



d = pickle.load(open('../mupots_eval/mupots/pred/ablation_POTR3D_B_Aug2_0_7M_TS10.pkl', 'rb'))






def project_pose(pose, f, c):
    if len(pose) > 0:
        pose = pose[...,:2] / pose[...,2:]
        pose = pose * f + c
    else:
        pose = np.full((1,17,2), config.DATASET.INVALID_VALUE)
    return pose


def load_annot(fname):
    def parse_pose(dt):
        annot3 = dt['annot3'][0,0]
        return annot3 
    data = sio.loadmat(fname)['annotations']
    results = []
    num_frames, num_inst = data.shape[0], data.shape[1]
    for i in range(num_frames):
        buff = []
        for j in range(num_inst):
            buff.append(parse_pose(data[i,j]))
        results.append(buff)
    return np.array(results).transpose(0,1,3,2)


def plot_all(poses3d, poses2d, imgs, final_out_file, mode='video'):
    size = 6
    fps = 30.
    bitrate = 3000
    radius = 3.
    limit = len(imgs)

    num_views = 3
    
    fig = plt.figure(figsize=(size*(len(poses3d) + 2), size*num_views), constrained_layout=True)
    subfigs = fig.subfigures(nrows=1, ncols=(len(poses3d) + 2))
    titles = ['Input', 'GT', 'VirtualPose', 'BEV w/o s', 'BEV w s', 'POTR-3D(Ours)', '']
    view_names = ['Front View', 'Side View', 'Top View']
    view_params = [[5.,-90.], [5.,0.], [90., -90.]]

    H, W = cv2.imread(imgs[0]).shape[:2]

    initialized = False
    assets = []
    rc = None
    def update_fig(f):
        nonlocal initialized, assets, rc

        if mode=='image' or not initialized:
            for i, subfig in enumerate(subfigs):
                subfig.suptitle(titles[i], fontsize=35)
                if i == 0:
                    ax = subfig.add_subplot(1,1,1)
                    ax.set_axis_off()
                    assets.append(ax)
                elif i <= 5:
                    _assets = []
                    for j in range(3):
                        ax = subfig.add_subplot(3,1,j+1,projection='3d')
                        ax.view_init(elev=view_params[j][0], azim=view_params[j][1])
                        # ax.set_axis_off()
                        ax.tick_params(top=False, bottom=False, left=False, right=False,
                            labelleft=False, labelbottom=False)
                        _assets.append(ax)
                    assets.append(_assets)
                else:
                    _assets = []
                    for j in range(3):
                        ax = subfig.add_subplot(3,1,j+1)
                        ax.set_axis_off()
                        _assets.append(ax)
                    assets.append(_assets)
            initialized = True

        if mode=='image' or initialized:
            for i, subfig in enumerate(subfigs):
                if i == 0:
                    ax = assets[i]
                    ax.clear()
                    ax.set_axis_off()
                    image = cv2.cvtColor(cv2.imread(imgs[f]), cv2.COLOR_BGR2RGB)
                    ax.imshow(image)
                elif i <= 5:
                    skeleton = bev_skeleton if titles[i].startswith('BEV') else mupots_skeleton
                    if i==1:
                        rc = poses3d[i-1][f][:,skeleton['rootidx']].mean(0)
                    for j in range(3):
                        ax = assets[i][j]
                        ax.clear()
                        ax.view_init(elev=view_params[j][0], azim=view_params[j][1])
                        ax.tick_params(top=False, bottom=False, left=False, right=False,
                                labelleft=False, labelbottom=False)
                        ax.set_xlim3d([-radius/2 + rc[0], radius/2 + rc[0]])
                        ax.set_zlim3d([radius/2 + rc[1], -radius/2 + rc[1]])
                        ax.set_ylim3d([-radius/2 + rc[2], radius/2 + rc[2]])
                        for p, pose in enumerate(poses3d[i-1][f]):
                            for j, j_parent in enumerate(skeleton['parents']):
                                if j_parent != -1:
                                    col = 'red' if j in skeleton['joints_right'] else 'black'
                                    ax.plot([pose[j,0], pose[j_parent,0]],
                                            [pose[j,1], pose[j_parent,1]],
                                            [pose[j,2], pose[j_parent,2]], zdir='y', color=col, linewidth=2.5)
                else:
                    for j in range(3):
                        ax = assets[i][j]
                        ax.clear()
                        ax.set_axis_off()
                        img = np.ones((H,W,3))
                        ax.imshow(img)
                        ax.text(0, H/2, view_names[j], size=30)
            
            
        if (f+1) % (limit // 3) == 0:
            print(f'\t{(f+1)/limit*100}% Completed..!')            
        fig.savefig('../VirtualPose/demo_results/results_all.jpg')


    if mode == 'video':
        anim = FuncAnimation(fig, update_fig, frames=np.arange(0, limit), interval=1000 / fps, repeat=False)
        if final_out_file.endswith('.mp4'):
            Writer = writers['ffmpeg']
            writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
            anim.save(final_out_file, writer=writer)
        elif final_out_file.endswith('.gif'):
            anim.save(final_out_file, dpi=80, writer='imagemagick')
        else:
            raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    else:
        update_fig(0)
        Path(os.path.dirname(final_out_file)).mkdir(exist_ok=True)
        fig.savefig(final_out_file)
        
    plt.close()


def plot_all2(poses3d, poses2d, imgs, final_out_file, frameno, mode='image'):
    size = 6
    fps = 30.
    bitrate = 3000
    radius = 3.
    limit = len(imgs)

    num_views = 3

    ### Save Image Drawing
    plt.ioff()
    fig = plt.figure(figsize=(size, size))
    ax_in = fig.add_subplot(1, 1, 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    # for i in range(0, limit, 3):
    print(imgs[frameno])
    image = cv2.cvtColor(cv2.imread(imgs[frameno]), cv2.COLOR_BGR2RGB)
    ax_in.imshow(image, aspect='equal')
    plt.savefig(f'{os.path.dirname(final_out_file)}/recons_0_{frameno:06d}.jpg', bbox_inches='tight')
    ax_in.cla()
    plt.close()

    ### Save Recons.
    fig = plt.figure(figsize=(size, size))
    radius = 4.0
    ax = fig.add_subplot(1, 1, 1, projection='3d')


    # fig = plt.figure(figsize=(size*(len(poses3d) + 2), size*num_views), constrained_layout=True)
    # subfigs = fig.subfigures(nrows=1, ncols=(len(poses3d) + 2))
    titles = ['Input', 'GT', 'VirtualPose', 'BEV w/o s', 'BEV w s', 'POTR-3D(Ours)', '']
    view_names = ['Front View', 'Side View', 'Top View']
    view_params = [[5.,-90.], [5.,0.], [90., -90.]]

    H, W = cv2.imread(imgs[0]).shape[:2]

    initialized = False
    assets = []
    rc = None
    def update_fig(f):
        nonlocal initialized, assets, rc

        if mode=='image' or not initialized:
            for i, title in enumerate(titles):
                if i in (0,6):
                    continue
                else:
                    _assets = []
                    for j in range(3):
                        ax.view_init(elev=view_params[j][0], azim=view_params[j][1])
                        # ax.set_axis_off()
                        ax.tick_params(top=False, bottom=False, left=False, right=False,
                            labelleft=False, labelbottom=False)
                        _assets.append(ax)
                    assets.append(_assets)
            initialized = True

        if mode=='image' or initialized:
            for i, title in enumerate(titles):
                if i in (0,6):
                    continue
                else:
                    skeleton = bev_skeleton if titles[i].startswith('BEV') else mupots_skeleton
                    if i==1:
                        rc = poses3d[i-1][f][:,skeleton['rootidx']].mean(0)
                    for j in range(1):
                        ax.clear()
                        ax.view_init(elev=view_params[j][0], azim=view_params[j][1])
                        # ax.tick_params(top=False, bottom=False, left=False, right=False,
                        #         labelleft=False, labelbottom=False)
                        ax.set_xlabel('$X$')
                        ax.set_ylabel('$Z (Depth)$')
                        ax.set_zlabel('$Y$')
                        try:
                            ax.set_aspect('equal')
                        except NotImplementedError:
                            ax.set_aspect('auto')
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        ax.set_zticklabels([])
                        ax.dist = 7.5
                        ax.set_xlim3d([-radius/2 + rc[0], radius/2 + rc[0]])
                        ax.set_zlim3d([radius/2 + rc[1], -radius/2 + rc[1]])
                        ax.set_ylim3d([-radius/2 + rc[2], radius/2 + rc[2]])
                        for p, pose in enumerate(poses3d[i-1][f]):
                            for j, j_parent in enumerate(skeleton['parents']):
                                if j_parent != -1:
                                    col = 'red' if j in skeleton['joints_right'] else 'black'
                                    ax.plot([pose[j,0], pose[j_parent,0]],
                                            [pose[j,1], pose[j_parent,1]],
                                            [pose[j,2], pose[j_parent,2]], zdir='y', color=col) #, linewidth=1.0)
                    
                    plt.savefig(f'{os.path.dirname(final_out_file)}/recons_{i}_{f:06d}.jpg', bbox_inches='tight')
                    ax.cla()
            

    update_fig(frameno)
    # Path(os.path.dirname(final_out_file)).mkdir(exist_ok=True)
    # fig.savefig(final_out_file)
        
    plt.close()




def plot(poses3d, poses2d, imgs, final_out_file, mode='video'):
    center = np.nanmean([pose3d[:,config.DATASET.ROOTIDX].mean(0) if len(pose3d) > 0 else [np.nan]*3 for pose3d in poses3d], axis=0)
    limit = len(imgs)

    size = 6
    elev = 5.
    azim = -90
    fps = 30.
    bitrate = 3000

    num_shots = 1
    plots_per_shot = 2

    plt.ioff()
    fig = plt.figure()
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(size * plots_per_shot, size * num_shots)
    subfigs = fig.subfigures(nrows=num_shots, ncols=1)
    if not isinstance(subfigs, np.ndarray):
        subfigs = np.array([subfigs])
    # fig.tight_layout()

    initialized = True
    assets = []
    def update_fig(f):
        nonlocal initialized, assets

        pose3d = poses3d[f]
        pose2d = poses2d[f]

        if not initialized:
            for i in range(num_shots):
                _assets = []
                for j in range(plots_per_shot-1,-1,-1):
                    if j == 0:
                        _assets.insert(0, subfigs[i].add_subplot(1, plots_per_shot, j+1))
                    else:
                        _assets.insert(0, subfigs[i].add_subplot(1, plots_per_shot, j+1, projection='3d'))
                assets.append(_assets)

            initialized = True
        else:
            for i in range(num_shots):
                for j in range(plots_per_shot-1,-1,-1):
                    if j == 0:
                        ax = assets[i][j]
                        ax.clear()
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                        ax.set_axis_off()
                        if len(pose2d) >0 :
                            _ = render_img(config, ax, pose2d, skeleton=mupots_skeleton, mode='2d', colormode='leftright', img_path=imgs[f])
                    else:
                        ax = assets[i][j]
                        ax.clear()
                        ax.view_init(elev=elev, azim=azim)
                        # ax.set_xticklabels([])
                        # ax.set_yticklabels([])
                        # ax.set_zticklabels([])
                        ax.set_xlabel('$X$')
                        ax.set_ylabel('$Z (Depth)$')
                        ax.set_zlabel('$Y$')
                        if len(pose3d) > 0 :
                            _ = render_img(config, ax, pose3d, skeleton=mupots_skeleton, mode='3d', colormode='leftright', center=center)

    if mode == 'video':
        anim = FuncAnimation(fig, update_fig, frames=np.arange(0, limit), interval=1000 / fps, repeat=False)
        if final_out_file.endswith('.mp4'):
            Writer = writers['ffmpeg']
            writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
            anim.save(final_out_file, writer=writer)
        elif final_out_file.endswith('.gif'):
            anim.save(final_out_file, dpi=80, writer='imagemagick')
        else:
            raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    else:
        update_fig(0)
        fig.savefig(final_out_file)
        
    plt.close()



def get_data_gt(demo_no, mode='video'):
    poses3d = load_annot(f'/data/dataset/MuPoTS/TS{demo_no}/annot.mat') / 1000

    return poses3d, None, None, None, None

def get_data_potr3d(demo_no, mode='video'):
    # print("\tProcessing >>>", f'../mupots_eval/mupots/pred/ablation_POTR3D_B_Aug2_0_7M_TS{demo_no}')
    try:
        data = pickle.load(open(f'../mupots_eval/mupots/pred/ablation_POTR3D_B_Aug2_0_7M_TS{demo_no}.pkl', 'rb'))
    except:
        raise ValueError(f'Demo_{demo_no} is not processed yet')
    
    predP = data.copy()
    predD = predP[...,config.DATASET.ROOTIDX,2] * norm_factor[demo_no-1]
    rootXY = (predP[...,config.DATASET.ROOTIDX,:2] / 1000 + [1, res_hs[demo_no-1]/res_ws[demo_no-1]]) * res_ws[demo_no-1] / 2
    rootX = rootXY[...,0]
    rootY = rootXY[...,1]

    pred_root_3d = np.stack([(rootX - cxs[demo_no-1])*predD/fxs[demo_no-1], (rootY - cys[demo_no-1])*predD/fys[demo_no-1], predD], axis=-1)[...,None,:]

    predP[...,config.DATASET.ROOTIDX,:] = 0
    
    poses3d = (predP + pred_root_3d) / 1000
    poses2d = [project_pose(pose3d, [fxs[demo_no-1], fys[demo_no-1]], [cxs[demo_no-1], cys[demo_no-1]]) for pose3d in poses3d]
    
    imgs = sorted(glob.glob(f'/data/dataset/MuPoTS/TS{demo_no}/*.jpg'))

    final_out_file = f"../VirtualPose/demo_results/potr3d_result_pose_demo_{demo_no}.mp4"

    return poses3d, poses2d, imgs, final_out_file, mode


def get_data_virtualpose(demo_no, mode='video'):
    # print("\tProcessing >>>", f"../VirtualPose/demo_results/preds_3d_kpt_demo_{demo_no}")
    try:
        data = sio.loadmat(f"../VirtualPose/demo_results/preds_3d_kpt_demo_{demo_no}")
    except:
        raise ValueError(f'Demo_{demo_no} is not processed yet')
    keys = [x for x in sorted(data.keys()) if x.startswith('TS')]

    poses3d = [data[k]/1000 for k in keys]
    poses2d = [project_pose(pose3d, [fxs[demo_no-1], fys[demo_no-1]], [cxs[demo_no-1], cys[demo_no-1]]) for pose3d in poses3d]
    
    imgs = sorted(glob.glob(f'/data/dataset/MuPoTS/TS{demo_no}/*.jpg'))

    final_out_file = f"../VirtualPose/demo_results/result_pose_demo_{demo_no}.mp4"

    return poses3d, poses2d, imgs, final_out_file, mode


def get_data_romp(demo_no, if_smoothing=False, mode='video'):
    # # print("\tProcessing >>>", f"../ROMP/demo/demo_{demo_no}_resized/video_results_{'wo_smoothing' if not if_smoothing else 'w_smoothing_3'}.npz")
    # try:
    #     data = np.load(f"../ROMP/demo/demo_{demo_no}_resized/video_results_{'wo_smoothing' if not if_smoothing else 'w_smoothing_3'}.npz", allow_pickle=True)['results'].item()
    # except:
    #     raise ValueError(f'Demo_{demo_no} is not processed yet')
    # poses3d = [data[k]['joints'][:,-17:] + data[k]['cam_trans'][:,None] for k in data]
    # poses2d = [data[k]['pj2d_org'][:,-17:] for k in data]
    
    # imgs = sorted(glob.glob(f'../potr3d/demo/demo_{demo_no}_resized/*.jpg'))

    # final_out_file = f"../ROMP/demo/demo_{demo_no}_resized/results_pose_{'wo_smoothing' if not if_smoothing else 'w_smoothing_3'}.mp4"

    # return poses3d, poses2d, imgs, final_out_file, mode

    #################################### MUPOTS #######################################
    imgs = sorted(glob.glob(f'/data/dataset/MuPoTS/TS{demo_no}/*.jpg'))
    H,W,_ = cv2.imread(imgs[0]).shape
    fov = 60 # degree
    f = max(H,W)/2 * 1/np.tan(np.radians(fov/2))
    
    fxs = [1500.9799492788811, 1500.8605847627498, 1493.0169231973466, 1492.4964211490396, 1494.751093087024, 1132.8454726004886, 1103.8105659586838, 1123.8970261035088, 1103.5365554667687, 1103.802134740414, 1114.3884708175303, 1113.6840903858147, 1102.1600498245136, 1124.119473712715, 1100.8391573135282, 1137.9504563337387, 1137.8203133580198, 1191.8365301432216, 1103.689283995902, 1191.8378621424363]
    fys = [1495.9003438753227, 1495.8994125542592, 1495.7220938482521, 1490.0239074933459, 1498.4098509691555, 1132.5149281839203, 1101.450236791744, 1121.6128547290284, 1101.450354911498, 1101.4509428236727, 1112.8719381062135, 1114.1954171009895, 1103.2677652946732, 1121.6112076685376, 1101.978920346303, 1138.332582310877, 1138.3319658637479, 1190.9720764607541, 1103.2831108425546, 1190.9735951379007]
    f_gt = np.sqrt(fxs[demo_no-1] * fys[demo_no-1])
    
    try:
        data = np.load(f"../ROMP/demo/MuPoTS_TS{demo_no}/video_results_{'wo_smoothing' if not if_smoothing else 'w_smoothing_3'}.npz", allow_pickle=True)['results'].item()
        poses3d = [data[k]['joints'][:,-17:] + data[k]['cam_trans'][:,None] for k in data]
        poses2d = [data[k]['pj2d_org'][:,-17:] for k in data]
    except:
        poses3d, poses2d = [], []
        for i in range(len(imgs)):
            datafile = glob.glob(f'../ROMP/demo/MuPoTS_TS{demo_no}/{i:08d}__2_0.08.npz')
            if len(datafile) == 0:
                poses3d.append(np.zeros((0,0,0)))
                poses2d.append(np.zeros((0,0,0)))
            else:
                data = np.load(datafile[0], allow_pickle=True)['results'].item()
                poses3d.append(data['joints'][:,-17:] + data['cam_trans'][:,None])
                poses2d.append(data['pj2d_org'][:,-17:])        

    poses3d_new = []
    for pose3d in poses3d:
        if len(pose3d) == 0:
            poses3d_new.append(pose3d)
        else:
            pose3d_new = pose3d.copy()
            root = pose3d_new[:,config.DATASET.ROOTIDX:(config.DATASET.ROOTIDX+1)]
            pose3d_new = pose3d_new - root
            root *= f_gt/f
            pose3d_new = pose3d_new + root
            poses3d_new.append(pose3d_new)
    poses3d = poses3d_new

                
    final_out_file = f"../ROMP/demo/demo_{demo_no}_resized/results_pose_{'wo_smoothing' if not if_smoothing else 'w_smoothing_3'}.mp4"
    #################################### MUPOTS #######################################


    return poses3d, poses2d, imgs, final_out_file, mode





if __name__=="__main__":
    mode = 'all'

    ## POTR3D
    if mode=='potr3d':
        for demo_no in range(13,14):
            print(f"DEMO {demo_no}")
            data = get_data_potr3d(demo_no, mode='video')
            plot(*data)
    
    ## VirtualPose
    if mode=='virtualpose':
        for demo_no in range(1,21):
            print(f"DEMO {demo_no}")
            data = get_data_virtualpose(demo_no, mode='video')
            plot(*data)

    ## ROMP
    if mode=='romp':
        for demo_no in range(7,8):
            print(f"DEMO {demo_no}")
            for if_smoothing in (True, False):
                data = get_data_romp(demo_no, if_smoothing, mode='video')
                plot(*data)

    if mode=='all':
        # for demo_no in [3,6]: #range(1,21):
        for demo_no, frame_nos in zip([6,18], [[15,90,115,520],[30,50,60,67,90]]): #range(1,21):
            print(f"DEMO {demo_no}")

            poses_3d_gt, _, _, _, _ = get_data_gt(demo_no)
            poses_3d_virtualpose, poses_2d_virtualpose, imgs, _, _ = get_data_virtualpose(demo_no)
            poses_3d_bev_wo_s, poses_2d_bev_wo_s, _, _, _ = get_data_romp(demo_no, if_smoothing=True)
            poses_3d_bev_w_s, poses_2d_bev_w_s, _, _, _ = get_data_romp(demo_no, if_smoothing=False)
            poses_3d_potr3d, poses_2d_potr3d, _, _, _ = get_data_potr3d(demo_no)

            poses_3d = [poses_3d_gt, poses_3d_virtualpose, poses_3d_bev_wo_s, poses_3d_bev_w_s, poses_3d_potr3d]
            poses_2d = [None, poses_2d_virtualpose, poses_2d_bev_wo_s, poses_2d_bev_w_s, poses_2d_potr3d]

            final_out_file = f'../VirtualPose/demo_results/mupots_{demo_no}/results_all_{demo_no}.jpg'

            # Front View
            for frame_no in frame_nos:
                plot_all2(poses_3d, poses_2d, imgs, final_out_file, frame_no, mode='image')