import os
import glob
import argparse
import numpy as np
import torch
import json
import cv2
import trimesh
from natsort import natsorted

from easyvolcap.utils.base_utils import *
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import *
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.camera_utils import read_cameras

import smplx
from smplx.lbs import batch_rodrigues, batch_rot2aa


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check SMPLH mesh')
    parser.add_argument('--data_dir', type=str, default='./data/30min_data_0')
    parser.add_argument('--smplh', type=str, default='output-output-smpl-3d/mesh')
    parser.add_argument('--images_dir', type=str, default='images')
    args = parser.parse_args()
    
    cameras = read_cameras(args.data_dir)
    smplxs = glob.glob(os.path.join(args.data_dir, args.smplh, '*.ply'))
    smplxs = sorted(smplxs)[:300]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(join(args.data_dir, 'output-output-smpl-3d/mesh-smplh-vis.avi'), fourcc, 30, (1440, 2560), True)

    for i in tqdm(range(len(smplxs))):
        if 'cano' in smplxs[i]:
            continue

        frame = int(os.path.basename(smplxs[i]).split('.')[0])
        smplh_v = trimesh.load(os.path.join(args.data_dir, args.smplh, f'{frame:06d}.ply'))
        smplh_v = smplh_v.vertices

        for cam in ['14']:
            K = cameras[cam]['K']
            R = cameras[cam]['R']
            T = cameras[cam]['T']

            smplh_v = (K @ (R @ smplh_v.T + T)).T
            smplh_v = smplh_v[:, :2] / smplh_v[:, 2:]

            image = cv2.imread(os.path.join(args.data_dir, args.images_dir, cam, f'{frame:06d}.jpg'))
            for j in range(len(smplh_v)):
                image = cv2.circle(image, (int(smplh_v[j, 0]), int(smplh_v[j, 1])), radius=1, color=(0, 0, 255), thickness=2)
            out.write(image)
    out.release()