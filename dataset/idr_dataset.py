import glob
import json
import os
from os.path import join
import numpy as np
from natsort import natsorted
import cv2
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from easyvolcap.utils.base_utils import *
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import load_dotdict, to_tensor, to_cuda
from easyvolcap.utils.camera_utils import read_cameras

from smplx.lbs import batch_rodrigues


class IDRDataset(Dataset):
    def __init__(self,
                 data_dir,
                 ratio=1.0,
                 frame_range=[0, 1, 1],
                 camera_range=[0, None, 1],
                 images_dir='images',
                 masks_dir='masks',
                 pos_map_dir='smpl_pos_map_1024',
                 smplx_dir='output-output-smpl-3d/smplxfull',
                 ):
        super(IDRDataset, self).__init__()

        self.data_dir = data_dir
        self.ratio = ratio
        self.frame_range = frame_range
        self.camera_range = camera_range
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.pos_map_dir = pos_map_dir
        self.smplx_dir = smplx_dir

        self.load_cam_data()
        self.load_pos_map()

        self.data_list = []
        for i in range(len(self.pos_maps)):
            for j in range(len(self.cam_names)):
                self.data_list.append((i, j))
        self.data_list = np.asarray(self.data_list)
        log('dataset length:', len(self.data_list))

    def load_pos_map(self):
        pos_maps = natsorted(glob.glob(join(self.data_dir, self.pos_map_dir, '*.exr')))
        pos_maps = np.array([p for p in pos_maps if 'cano' not in p])
        if self.frame_range[1] == None:
            self.frame_range[1] = len(pos_maps)
        pos_maps = pos_maps[self.frame_range[0]:self.frame_range[1]:self.frame_range[2]]
        self.pos_maps = pos_maps

    def load_cam_data(self):
        cam_data = read_cameras(self.data_dir)
        cam_names = natsorted(list(cam_data.keys()))
        if self.camera_range[1] == None:
            self.camera_range[1] = len(cam_names)
        self.cam_names = cam_names[self.camera_range[0]:self.camera_range[1]:self.camera_range[2]]
        log(f'Use cameras: {self.cam_names}')
        self.Ks = []
        self.Rs = []
        self.Ts = []
        for i in range(len(self.cam_names)):
            cam = cam_data[self.cam_names[i]]
            self.Ks.append(cam['K'])
            self.Rs.append(cam['R'])
            self.Ts.append(cam['T'])
        self.Ks = torch.from_numpy(np.stack(self.Ks)).float()
        self.Rs = torch.from_numpy(np.stack(self.Rs)).float()
        self.Ts = torch.from_numpy(np.stack(self.Ts)).float()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        latent_index, view_index = self.data_list[index]
        frame_index = int(os.path.splitext(os.path.basename(self.pos_maps[latent_index]))[0])
        camera_index = self.cam_names[view_index]

        K = self.Ks[view_index]
        if self.ratio != 1.0:
            K = K.clone()
            K[:2, :] *= self.ratio
        R = self.Rs[view_index]
        T = self.Ts[view_index]
        
        smpl_pos_map = cv2.imread(self.pos_maps[latent_index], cv2.IMREAD_UNCHANGED)
        pos_map_size = smpl_pos_map.shape[1] // 2
        smpl_pos_map = np.concatenate([smpl_pos_map[:, :pos_map_size], smpl_pos_map[:, pos_map_size:]], axis=2)
        smpl_pos_map = smpl_pos_map.transpose((2, 0, 1))
        smpl_pos_map = to_tensor(smpl_pos_map).float()

        # with open(join(self.data_dir, self.smplh_dir, f'{frame_index:06d}.json')) as f:
        #     smplh_data = json.load(f)['annots'][0]
        # Rh = batch_rodrigues(torch.from_numpy(np.array(smplh_data['Rh'])).float())[0]
        # Th = torch.from_numpy(np.array(smplh_data['Th'])[0]).float()
        
        smplx_data = np.load(join(self.data_dir, self.smplx_dir, f'{frame_index:06d}.npz'))
        Rh = torch.from_numpy(smplx_data['Rh'][0]).float()
        Th = torch.from_numpy(smplx_data['Th'][0]).float()
        betas = torch.from_numpy(smplx_data['betas'][0]).float()
        global_orient = torch.from_numpy(smplx_data['global_orient'][0]).float()
        transl = torch.from_numpy(smplx_data['transl'][0]).float()
        body_pose = torch.from_numpy(smplx_data['body_pose'][0]).float()
        left_hand_pose = torch.from_numpy(smplx_data['left_hand_pose'][0]).float()
        right_hand_pose = torch.from_numpy(smplx_data['right_hand_pose'][0]).float()
        jaw_pose = torch.from_numpy(smplx_data['jaw_pose'][0]).float()
        expression = torch.from_numpy(smplx_data['expression'][0]).float()

        # jaw_pose = self.flame_data['flame_pose_params'][frame_index][:3]
        # jaw_pose = batch_rodrigues(jaw_pose.reshape(1, 3)).reshape(1, 3, 3).float()
        # expression = self.flame_data['exp_code'][frame_index].flatten().float()

        color_img, mask_img = self.load_color_mask_images(frame_index, camera_index)
        color_img = torch.from_numpy(color_img).float() / 255.0
        color_img = color_img.permute(2, 0, 1)
        boundary_mask_img, mask_img = self.get_boundary_mask(mask_img)
        boundary_mask_img = torch.from_numpy(boundary_mask_img).float()
        boundary_mask_img = boundary_mask_img.permute(2, 0, 1)
        mask_img = torch.from_numpy(mask_img).float()
        mask_img = mask_img.permute(2, 0, 1)

        output = dotdict({
            'latent_index': latent_index,
            'view_index': view_index,
            'frame_index': frame_index,
            'camera_index': camera_index,
            'smpl_pos_map': smpl_pos_map,
            'K': K,
            'R': R,
            'T': T,
            'H': color_img.shape[1],
            'W': color_img.shape[2],
            'n': 0.1,
            'f': 10.0,
            'Rh': Rh,
            'Th': Th,
            'rgb': color_img,
            'msk': mask_img,
            'bd_msk': boundary_mask_img,
            'betas': betas,
            'global_orient': global_orient,
            'transl': transl,
            'body_pose': body_pose,
            'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,
            'jaw_pose': jaw_pose,
            'expression': expression,
        })
        return output
    
    def load_color_mask_images(self, frame_index, camera_index):
        color_img = cv2.imread(join(self.data_dir, self.images_dir, camera_index, f'{frame_index:06d}.jpg'))
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        mask_img = cv2.imread(join(self.data_dir, self.masks_dir, camera_index, f'{frame_index:06d}.jpg'))
        if self.ratio != 1.0:
            color_img = cv2.resize(color_img, (0, 0), fx=self.ratio, fy=self.ratio, interpolation=cv2.INTER_LINEAR)
            mask_img = cv2.resize(mask_img, (0, 0), fx=self.ratio, fy=self.ratio, interpolation=cv2.INTER_LINEAR)
        return color_img, mask_img
    
    @staticmethod
    # NOTE default kernel_size=5
    def get_boundary_mask(mask, kernel_size=3):
        mask_bk = mask.copy()
        thres = 128
        mask[mask < thres] = 0
        mask[mask > thres] = 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_erode = cv2.erode(mask.copy(), kernel)
        mask_dilate = cv2.dilate(mask.copy(), kernel)
        boundary_mask = (mask_dilate - mask_erode) == 1
        boundary_mask = np.logical_or(boundary_mask,
                                      np.logical_and(mask_bk > 5, mask_bk < 250))
        
        # boundary_mask_resized = cv.resize(boundary_mask.astype(np.uint8), (0, 0), fx = 0.5, fy = 0.5)
        # cv.imshow('boundary_mask', boundary_mask_resized.astype(np.uint8) * 255)
        # cv.waitKey(0)

        return boundary_mask, mask == 1
    

if __name__ == "__main__":
    dataset = IDRDataset(
        data_dir='./data/30min_data_0',
        frame_range=[0, 1, 1],
        camera_range=[0, 1, 1],
        images_dir='images',
        masks_dir='masks',
        pos_map_dir='smpl_pos_map_1024',
        smplx_dir='output-output-smpl-3d/smplxfull',
    )
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for data in dataloader:
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
            else:
                print(k, v)