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
    parser = argparse.ArgumentParser(description='Check SMPLX mesh and face landmarks')
    parser.add_argument('--data_dir', type=str, default='./data/30min_data_0')
    parser.add_argument('--smplh', type=str, default='output-output-smpl-3d/smplfull')
    parser.add_argument('--smplx', type=str, default='output-output-smpl-3d/mesh-smplx')
    parser.add_argument('--flame', type=str, default='output-output-smpl-3d/30min_0_track_params.pt')
    parser.add_argument('--output_dir', type=str, default='output-output-smpl-3d/smplxfull')
    args = parser.parse_args()

    cameras = read_cameras(args.data_dir)
    flames = torch.load(os.path.join(args.data_dir, args.flame)) 
    smplxs = glob.glob(os.path.join(args.data_dir, args.smplx, '*.npz'))
    smplxs = sorted(smplxs)

    def write_smplx_full(smplx):
        if 'cano' in smplx:
            return

        frame = int(os.path.basename(smplx).split('.')[0])
        
        smplx_data = load_dotdict(smplx)
        smplx_data.pop('full_pose')
        smplx_data.pop('joints')
        smplx_data.pop('vertices')
        smplx_data.pop('faces')
        smplx_data.pop('leye_pose')
        smplx_data.pop('reye_pose')
        smplx_data.pop('jaw_pose')
        smplx_data.pop('expression')
        loss = smplx_data.pop('loss')
        smplx_data = to_tensor(smplx_data)

        with open(os.path.join(args.data_dir, args.smplh, f'{frame:06d}.json'), 'r') as f:
            smplh = json.load(f)['annots'][0]
        Rh = batch_rodrigues(torch.from_numpy(np.array(smplh['Rh'])).float())
        Th = torch.from_numpy(np.array(smplh['Th'])).float()
        smplx_data['Rh'] = Rh.reshape(1, 3, 3)
        smplx_data['Th'] = Th.reshape(1, 3)

        jaw_pose = flames['flame_pose_params'][frame][:3]
        jaw_pose = batch_rodrigues(jaw_pose.reshape(1, 3)).reshape(1, 1, 3, 3)
        smplx_data['jaw_pose'] = jaw_pose
        expression = flames['exp_code'][frame].reshape(1, 100)
        smplx_data['expression'] = expression

        save_npz(f'{frame:06d}', os.path.join(args.data_dir, args.output_dir), smplx_data)

    os.makedirs(os.path.join(args.data_dir, args.output_dir), exist_ok=True)
    parallel_execution(smplxs, action=write_smplx_full, print_progress=True, sequential=False)