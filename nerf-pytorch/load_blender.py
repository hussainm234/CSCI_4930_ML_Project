import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, f'transforms_{s}.json'), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        skip = 1 if s == 'train' or testskip == 0 else testskip
        
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'])
            img = imageio.imread(fname)

            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            elif img.shape[2] == 2:
                grayscale = img[:, :, 0]
                img = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 3:
                pass
            else:
                raise ValueError(f"Unexpected number of channels: {img.shape}")

            img = cv2.resize(img, (400, 400))
            assert img.shape == (400, 400, 3), f"Unexpected shape after all conversions: {img.shape}"
            img = img.astype(np.float32) / 255.

            print(f"Loaded {fname}, shape: {img.shape}, min: {np.min(img)}, max: {np.max(img)}")

            imgs.append(img)
            poses.append(np.array(frame['transform_matrix']))

        imgs = np.array(imgs).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    render_poses = torch.stack([
        pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]
    ], 0)

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, imgs.shape[3]))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    print(f"Final imgs array shape: {imgs.shape}, dtype: {imgs.dtype}, min: {np.min(imgs)}, max: {np.max(imgs)}")

    return imgs, poses, render_poses, [H, W, focal], i_split
