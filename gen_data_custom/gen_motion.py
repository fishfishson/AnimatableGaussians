import os
from os.path import join
import argparse
import numpy as np
import json
from easyvolcap.utils.parallel_utils import parallel_execution


def load_one_easymocap(input: str, args):
    ext = os.path.splitext(input)[-1]
    if ext == '.json':
        params = json.load(open(input))['annots'][0]
    elif ext == '.npy':
        params = np.load(input, allow_pickle=True).item()
    poses = np.array(params['poses'])[0]  # remove first dim
    shapes = np.array(params['shapes'])[0]  # remove first dim
    Rh = np.array(params['Rh'])[0]
    Th = np.array(params['Th'])[0]

    full_poses = np.zeros((args.n_bones * 3))
    full_poses[:poses.shape[-1]] = poses
    # full_poses[:3] = Rh

    # the params of neural body
    return full_poses.astype(np.float32), Rh.astype(np.float32), Th.astype(np.float32), shapes.astype(np.float32)


def load_easymocap_params(param_dir: str, args):
    param_files = sorted(os.listdir(param_dir), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    param_files = [join(param_dir, p) for p in param_files]
    poses, Rh, Th, shapes = zip(*parallel_execution(param_files, args, action=load_one_easymocap, print_progress=True))
    poses = np.stack(poses)
    Rh = np.stack(Rh)
    Th = np.stack(Th)
    shapes = np.stack(shapes)
    return poses, Rh, Th, shapes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/0325data')
    parser.add_argument('--motion_in', type=str, default='output-output-smpl-3d/smplfull')
    parser.add_argument('--motion_out', type=str, default='motion.npz')
    parser.add_argument('--n_bones', type=int, default=52) 
    args = parser.parse_args()

    motion_in = os.path.join(args.data_root, args.motion_in)
    poses, Rh, Th, shapes = load_easymocap_params(motion_in, args)
    np.savez_compressed(os.path.join(args.data_root, args.motion_out), poses=poses, Rh=Rh, Th=Th, shapes=shapes)


if __name__ == '__main__':
    main()