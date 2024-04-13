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
    parser.add_argument('--images_dir', type=str, default='images')
    args = parser.parse_args()

    cameras = read_cameras(args.data_dir)
    flames = torch.load(os.path.join(args.data_dir, args.flame)) 
    smplxs = glob.glob(os.path.join(args.data_dir, args.smplx, '*.npz'))
    smplxs = sorted(smplxs)[:300]

    smpl_model = smplx.SMPLXLayer(model_path='./data/bodymodels/smplx/smplx',
                                  gender='neutral',
                                  use_compressed=False,
                                  use_face_contour=True,
                                  num_betas=16,
                                  num_expression_coeffs=100)
    print(smpl_model)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out1 = cv2.VideoWriter(join(args.data_dir, 'output-output-smpl-3d/mesh-flame-vis.avi'), fourcc, 30, (1440, 2560), True)
    out2 = cv2.VideoWriter(join(args.data_dir, 'output-output-smpl-3d/mesh-smplx-vis.avi'), fourcc, 30, (1440, 2560), True)

    for i in tqdm(range(len(smplxs))):
        if 'cano' in smplxs[i]:
            continue

        frame = int(os.path.basename(smplxs[i]).split('.')[0])
        
        smplx_data = load_dotdict(smplxs[i])
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

        jaw_pose = flames['flame_pose_params'][frame][:3]
        jaw_pose = batch_rodrigues(jaw_pose.reshape(1, 3)).reshape(1, 1, 3, 3)
        smplx_data['jaw_pose'] = jaw_pose
        expression = flames['exp_code'][frame].reshape(1, 100)
        smplx_data['expression'] = expression

        smplx_out = smpl_model(**smplx_data, return_full_pose=False, return_verts=True)
        vertices = smplx_out.vertices
        landmarks = smplx_out.landmarks
        vertices = (Rh @ vertices.mT + Th[:, :, None]).mT
        vertices = vertices[0].numpy()
        landmarks = (Rh @ landmarks.mT + Th[:, :, None]).mT
        landmarks = landmarks[0].numpy()

        for cam in ['10']:
            K = cameras[cam]['K']
            R = cameras[cam]['R']
            T = cameras[cam]['T']

            landmarks = (K @ (R @ landmarks.T + T)).T
            vertices = (K @ (R @ vertices.T + T)).T
            landmarks = landmarks[:, :2] / landmarks[:, 2:]
            vertices = vertices[:, :2] / vertices[:, 2:]

            image1 = cv2.imread(os.path.join(args.data_dir, args.images_dir, cam, f'{frame:06d}.jpg'))
            image2 = cv2.imread(os.path.join(args.data_dir, args.images_dir, cam, f'{frame:06d}.jpg'))
            for j in range(len(landmarks)):
                image = cv2.circle(image1, (int(landmarks[j, 0]), int(landmarks[j, 1])), radius=1, color=(0, 0, 255), thickness=2)
            for j in range(len(vertices)):
                image = cv2.circle(image2, (int(vertices[j, 0]), int(vertices[j, 1])), radius=1, color=(0, 0, 255), thickness=2)
            out1.write(image1)
            out2.write(image2)
    out1.release()
    out2.release()
