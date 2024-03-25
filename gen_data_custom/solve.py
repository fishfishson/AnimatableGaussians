import os
import argparse

def solve(num_joints, point_interpolant_exe, depth=7, tmp_dir=None):
    for jid in range(num_joints):
        print('Solving joint %d' % jid)
        cmd = f'{point_interpolant_exe} ' + \
            f'--inValues {os.path.join(tmp_dir, f"cano_data_lbs_val_{jid:02d}.xyz")} ' + \
            f'--inGradients {os.path.join(tmp_dir, f"cano_data_lbs_grad_{jid:02d}.xyz")} ' + \
            f'--gradientWeight 0.05 --dim 3 --verbose ' + \
            f'--grid {os.path.join(tmp_dir, f"grid_{jid:02d}.grd")} ' + \
            f'--depth {depth} '

        os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Solve point interpolant')
    parser.add_argument('--data_dir', type = str, default = './data/0116data')
    parser.add_argument('--type', type = str, choices=['smpl', 'smplh', 'smplx'], default='smplh')
    args = parser.parse_args()

    if args.type == 'smplh':
        num_joints = 52
        depth = 7
        tmp_dir = os.path.join(args.data_dir, 'tmp_dir')
        point_interpolant_exe = './point_interpolant'
        solve(num_joints, ".\\bins\\PointInterpolant.exe", depth, tmp_dir=tmp_dir)
    else:
        raise ValueError('Not implemented yet')