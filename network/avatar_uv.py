import os
from os.path import join
import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch3d.ops
import pytorch3d.transforms
import cv2 as cv
import trimesh
import config
from network.styleunet.dual_styleunet import DualStyleUNet
from gaussians.gaussian_model import GaussianModel
from gaussians.gaussian_renderer import render3


class AvatarNet(nn.Module):
    def __init__(self, opt):
        super(AvatarNet, self).__init__()
        self.opt = opt

        self.size = opt.get('size', 1024)
        self.inp_size = self.size // 2
        self.out_size = self.size
        if self.size == 1024:
            smpl_pos_map_dir = 'smpl_pos_map'
        else:
            smpl_pos_map_dir = f'smpl_pos_map_{self.size}'
        self.random_style = opt.get('random_style', False)
        self.with_viewdirs = opt.get('with_viewdirs', True)
        self.mask_out_flame = opt.get('mask_out_flame', False)
        self.use_extra_flame = opt.get('use_extra_flame', False)
        self.mask_out_mano = opt.get('mask_out_mano', False)
        self.use_extra_mano = opt.get('use_extra_mano', False)
        self.replace_flame = opt.get('replace_flame', False)

        # init canonical gausssian model
        self.max_sh_degree = 0
        self.cano_gaussian_model = GaussianModel(sh_degree = self.max_sh_degree)
        if os.path.exists(join(config.opt['train']['data']['data_dir'], 'cano_smpl_pos_map.exr')):
            cano_smpl_map = cv.imread(config.opt['train']['data']['data_dir'] + '/cano_smpl_pos_map.exr', cv.IMREAD_UNCHANGED)
        else:
            cano_smpl_map = cv.imread(join(config.opt['train']['data']['data_dir'], f'{smpl_pos_map_dir}/cano_smpl_pos_map.exr'), cv.IMREAD_UNCHANGED)
        self.cano_smpl_map = torch.from_numpy(cano_smpl_map).to(torch.float32).to(config.device)
        self.cano_smpl_mask = torch.linalg.norm(self.cano_smpl_map, dim = -1) > 0.
        self.init_points = self.cano_smpl_map[self.cano_smpl_mask]
        if os.path.exists(join(config.opt['train']['data']['data_dir'], 'init_pts_lbs.npy')):
            self.lbs = torch.from_numpy(np.load(config.opt['train']['data']['data_dir'] + '/init_pts_lbs.npy')).to(torch.float32).to(config.device)
        else:
            self.lbs = torch.from_numpy(np.load(join(config.opt['train']['data']['data_dir'], f'{smpl_pos_map_dir}/init_pts_lbs.npy'))).to(torch.float32).to(config.device)
        self.cano_gaussian_model.create_from_pcd(self.init_points, torch.rand_like(self.init_points), spatial_lr_scale = 2.5)

        self.color_net = DualStyleUNet(inp_size = self.inp_size, inp_ch = 3, out_ch = 3, out_size = self.out_size, style_dim = 512, n_mlp = 2)
        self.position_net = DualStyleUNet(inp_size = self.inp_size, inp_ch = 3, out_ch = 3, out_size = self.out_size, style_dim = 512, n_mlp = 2)
        self.other_net = DualStyleUNet(inp_size = self.inp_size, inp_ch = 3, out_ch = 8, out_size = self.out_size, style_dim = 512, n_mlp = 2)

        self.color_style = torch.ones([1, self.color_net.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.color_net.style_dim)
        self.position_style = torch.ones([1, self.position_net.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.position_net.style_dim)
        self.other_style = torch.ones([1, self.other_net.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.other_net.style_dim)

        if self.with_viewdirs:
            if os.path.exists(join(config.opt['train']['data']['data_dir'], 'cano_smpl_nml_map.exr')):
                cano_nml_map = cv.imread(config.opt['train']['data']['data_dir'] + '/cano_smpl_nml_map.exr', cv.IMREAD_UNCHANGED)
            else:
                cano_nml_map = cv.imread(join(config.opt['train']['data']['data_dir'], f'{smpl_pos_map_dir}/cano_smpl_nml_map.exr'), cv.IMREAD_UNCHANGED)
            self.cano_nml_map = torch.from_numpy(cano_nml_map).to(torch.float32).to(config.device)
            self.cano_nmls = self.cano_nml_map[self.cano_smpl_mask]
            self.viewdir_net = nn.Sequential(
                nn.Conv2d(1, 64, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Conv2d(64, 128, 4, 2, 1)
            )
        
        if self.mask_out_flame:
            flame_mask = cv.imread(join(config.opt['train']['data']['data_dir'], f'{smpl_pos_map_dir}/cano_smpl_flame_map.exr'), cv.IMREAD_UNCHANGED)
            flame_mask = torch.from_numpy(flame_mask).to(torch.float32).to(config.device)
            self.flame_mask = flame_mask[self.cano_smpl_mask][..., 0]

        if self.mask_out_mano:
            mano_mask = cv.imread(join(config.opt['train']['data']['data_dir'], f'{smpl_pos_map_dir}/cano_smpl_mano_map.exr'), cv.IMREAD_UNCHANGED)
            mano_mask = torch.from_numpy(mano_mask).to(torch.float32).to(config.device)
            self.mano_mask = flame_mask[self.cano_smpl_mask][..., 0]

    def generate_mean_hands(self):
        # print('# Generating mean hands ...')
        import glob
        # get hand mask
        lbs_argmax = self.lbs.argmax(1)
        self.hand_mask = lbs_argmax == 20
        self.hand_mask = torch.logical_or(self.hand_mask, lbs_argmax == 21)
        self.hand_mask = torch.logical_or(self.hand_mask, lbs_argmax >= 25)

        pose_map_paths = sorted(glob.glob(config.opt['train']['data']['data_dir'] + '/smpl_pos_map/%08d.exr' % config.opt['test']['fix_hand_id']))
        smpl_pos_map = cv.imread(pose_map_paths[0], cv.IMREAD_UNCHANGED)
        pos_map_size = smpl_pos_map.shape[1] // 2
        smpl_pos_map = np.concatenate([smpl_pos_map[:, :pos_map_size], smpl_pos_map[:, pos_map_size:]], 2)
        smpl_pos_map = smpl_pos_map.transpose((2, 0, 1))
        pose_map = torch.from_numpy(smpl_pos_map).to(torch.float32).to(config.device)
        pose_map = pose_map[:3]

        cano_pts = self.get_positions(pose_map)
        opacity, scales, rotations = self.get_others(pose_map)
        colors, color_map = self.get_colors(pose_map)

        self.hand_positions = cano_pts#[self.hand_mask]
        self.hand_opacity = opacity#[self.hand_mask]
        self.hand_scales = scales#[self.hand_mask]
        self.hand_rotations = rotations#[self.hand_mask]
        self.hand_colors = colors#[self.hand_mask]

        # # debug
        # hand_pts = trimesh.PointCloud(self.hand_positions.detach().cpu().numpy())
        # hand_pts.export('./debug/hand_template.obj')
        # exit(1)

    def transform_cano2live(self, lbs, gaussian_vals, items):
        if 'cano2live_jnt_mats' in items:
            pt_mats = torch.einsum('nj,jxy->nxy', lbs, items['cano2live_jnt_mats'])
            positions = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], gaussian_vals['positions']) + pt_mats[..., :3, 3]
            rot_mats = pytorch3d.transforms.quaternion_to_matrix(gaussian_vals['rotations'])
            rot_mats = torch.einsum('nxy,nyz->nxz', pt_mats[..., :3, :3], rot_mats)
        elif 'cano2live_jnt_mats_woRoot' in items:
            pt_mats = torch.einsum('nj,jxy->nxy', lbs, items['cano2live_jnt_mats_woRoot'])
            positions = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], gaussian_vals['positions']) + pt_mats[..., :3, 3]
            positions = (items['global_orient'] @ positions.mT + items['transl'][:, None]).mT
            rot_mats = pytorch3d.transforms.quaternion_to_matrix(gaussian_vals['rotations'])
            rot_mats = torch.einsum('nxy,nyz->nxz', pt_mats[..., :3, :3], rot_mats)
            rot_mats = items['global_orient'] @ rot_mats
        gaussian_vals['positions'] = positions
        gaussian_vals['rotations'] = pytorch3d.transforms.matrix_to_quaternion(rot_mats)

        return gaussian_vals
    
    # def transform_cano2live_flame(self, lbs, gaussian_vals, items):
    #     if 'cano2live_jnt_mats' in items:
    #         pt_mats = torch.einsum('nj,jxy->nxy', lbs, items['cano2live_jnt_mats'])
    #         positions = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], gaussian_vals['positions']) + pt_mats[..., :3, 3]
    #         rot_mats = pytorch3d.transforms.quaternion_to_matrix(gaussian_vals['rotations'])
    #         rot_mats = torch.einsum('nxy,nyz->nxz', pt_mats[..., :3, :3], rot_mats)
    #     elif 'cano2live_jnt_mats_woRoot' in items:
    #         pt_mats = torch.einsum('nj,jxy->nxy', lbs, items['cano2live_jnt_mats_woRoot'])
    #         positions = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], gaussian_vals['positions']) + pt_mats[..., :3, 3]
    #         positions = (items['global_orient'] @ positions.mT + items['transl'][:, None]).mT
    #         rot_mats = pytorch3d.transforms.quaternion_to_matrix(gaussian_vals['rotations'])
    #         rot_mats = torch.einsum('nxy,nyz->nxz', pt_mats[..., :3, :3], rot_mats)
    #         rot_mats = items['global_orient'] @ rot_mats
    #     gaussian_vals['positions'] = positions
    #     # gaussian_vals['rotations'] = pytorch3d.transforms.matrix_to_quaternion(rot_mats)

    #     return gaussian_vals

    def get_positions(self, pose_map, return_map = False):
        position_map, _ = self.position_net([self.position_style], pose_map[None], randomize_noise = False)
        front_position_map, back_position_map = torch.split(position_map, [3, 3], 1)
        position_map = torch.cat([front_position_map, back_position_map], 3)[0].permute(1, 2, 0)
        delta_position = 0.05 * position_map[self.cano_smpl_mask]
        # delta_position = position_map[self.cano_smpl_mask]

        positions = delta_position + self.cano_gaussian_model.get_xyz
        if return_map:
            return positions, position_map
        else:
            return positions

    def get_others(self, pose_map):
        other_map, _ = self.other_net([self.other_style], pose_map[None], randomize_noise = False)
        front_map, back_map = torch.split(other_map, [8, 8], 1)
        other_map = torch.cat([front_map, back_map], 3)[0].permute(1, 2, 0)
        others = other_map[self.cano_smpl_mask]  # (N, 8)
        opacity, scales, rotations = torch.split(others, [1, 3, 4], 1)
        opacity = self.cano_gaussian_model.opacity_activation(opacity + self.cano_gaussian_model.get_opacity_raw)
        scales = self.cano_gaussian_model.scaling_activation(scales + self.cano_gaussian_model.get_scaling_raw)
        rotations = self.cano_gaussian_model.rotation_activation(rotations + self.cano_gaussian_model.get_rotation_raw)

        return opacity, scales, rotations

    def get_colors(self, pose_map, front_viewdirs = None, back_viewdirs = None):
        color_style = torch.rand_like(self.color_style) if self.random_style and self.training else self.color_style
        color_map, _ = self.color_net([color_style], pose_map[None], randomize_noise = False, view_feature1 = front_viewdirs, view_feature2 = back_viewdirs)
        front_color_map, back_color_map = torch.split(color_map, [3, 3], 1)
        color_map = torch.cat([front_color_map, back_color_map], 3)[0].permute(1, 2, 0)
        colors = color_map[self.cano_smpl_mask]
        return colors, color_map

    def get_viewdir_feat(self, items):
        with torch.no_grad():
            if 'cano2live_jnt_mats' in items:
                pt_mats = torch.einsum('nj,jxy->nxy', self.lbs, items['cano2live_jnt_mats'])
                live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.init_points) + pt_mats[..., :3, 3]
                live_nmls = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.cano_nmls)
            elif 'cano2live_jnt_mats_woRoot' in items:
                pt_mats = torch.einsum('nj,jxy->nxy', self.lbs, items['cano2live_jnt_mats_woRoot'])
                live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.init_points) + pt_mats[..., :3, 3]
                live_pts = (items['global_orient'] @ live_pts.mT + items['transl'][:, None]).mT
                live_nmls = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.cano_nmls)
                live_nmls = (items['global_orient'] @ live_nmls.mT).mT
            cam_pos = -torch.matmul(torch.linalg.inv(items['extr'][:3, :3]), items['extr'][:3, 3])
            viewdirs = F.normalize(cam_pos[None] - live_pts, dim = -1, eps = 1e-3)
            if self.training:
                viewdirs += torch.randn(*viewdirs.shape).to(viewdirs) * 0.1
            viewdirs = F.normalize(viewdirs, dim = -1, eps = 1e-3)
            viewdirs = (live_nmls * viewdirs).sum(-1)

            viewdirs_map = torch.zeros(*self.cano_nml_map.shape[:2]).to(viewdirs)
            viewdirs_map[self.cano_smpl_mask] = viewdirs

            viewdirs_map = viewdirs_map[None, None]
            viewdirs_map = F.interpolate(viewdirs_map, None, 0.5, 'nearest')
            front_viewdirs, back_viewdirs = torch.split(viewdirs_map, [self.inp_size, self.inp_size], -1)

        front_viewdirs = self.opt.get('weight_viewdirs', 1.) * self.viewdir_net(front_viewdirs)
        back_viewdirs = self.opt.get('weight_viewdirs', 1.) * self.viewdir_net(back_viewdirs)
        return front_viewdirs, back_viewdirs

    def get_pose_map(self, items):
        pt_mats = torch.einsum('nj,jxy->nxy', self.lbs, items['cano2live_jnt_mats_woRoot'])
        live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.init_points) + pt_mats[..., :3, 3]
        live_pos_map = torch.zeros_like(self.cano_smpl_map)
        live_pos_map[self.cano_smpl_mask] = live_pts
        live_pos_map = F.interpolate(live_pos_map.permute(2, 0, 1)[None], None, [0.5, 0.5], mode = 'nearest')[0]
        live_pos_map = torch.cat(torch.split(live_pos_map, [self.inp_size, self.inp_size], 2), 0)
        items.update({
            'smpl_pos_map': live_pos_map
        })
        return live_pos_map

    def render(self, items, bg_color = (0., 0., 0.), use_pca = False, use_vae = False, flame_gs = None, mano_gs = None):
        """
        Note that no batch index in items.
        """
        bg_color = torch.from_numpy(np.asarray(bg_color)).to(torch.float32).to(config.device)
        pose_map = items['smpl_pos_map'][:3]
        assert not (use_pca and use_vae), "Cannot use both PCA and VAE!"
        if use_pca:
            pose_map = items['smpl_pos_map_pca'][:3]
        if use_vae:
            pose_map = items['smpl_pos_map_vae'][:3]

        cano_pts, pos_map = self.get_positions(pose_map, return_map = True)
        opacity, scales, rotations = self.get_others(pose_map)
        if not self.training:
            scales = torch.clip(scales, 0., 0.03)
        if self.with_viewdirs:
            front_viewdirs, back_viewdirs = self.get_viewdir_feat(items)
        else:
            front_viewdirs, back_viewdirs = None, None
        colors, color_map = self.get_colors(pose_map, front_viewdirs, back_viewdirs)

        lbs = self.lbs
        init_points = self.init_points
        # if not self.training and config.opt['test'].get('fix_hand', False) and config.opt['mode'] == 'test':
        #     # print('# fuse hands ...')
        #     import utils.geo_util as geo_util
        #     cano_xyz = self.init_points
        #     wl = torch.sigmoid(2.5 * (geo_util.normalize_vert_bbox(items['left_cano_mano_v'], attris = cano_xyz, dim = 0, per_axis = True)[..., 0:1] + 2.0))
        #     wr = torch.sigmoid(-2.5 * (geo_util.normalize_vert_bbox(items['right_cano_mano_v'], attris = cano_xyz, dim = 0, per_axis = True)[..., 0:1] - 2.0))
        #     wl[cano_xyz[..., 1] < items['cano_smpl_center'][1]] = 0.
        #     wr[cano_xyz[..., 1] < items['cano_smpl_center'][1]] = 0.

        #     s = torch.maximum(wl + wr, torch.ones_like(wl))
        #     wl, wr = wl / s, wr / s

        #     w = wl + wr
        #     cano_pts = w * self.hand_positions + (1.0 - w) * cano_pts
        #     opacity = w * self.hand_opacity + (1.0 - w) * opacity
        #     scales = w * self.hand_scales + (1.0 - w) * scales
        #     rotations = w * self.hand_rotations + (1.0 - w) * rotations
        #     # colors = w * self.hand_colors + (1.0 - w) * colors

        if not self.training and self.mask_out_flame:
            valid = self.flame_mask < 0.5
            cano_pts = cano_pts[valid]
            opacity = opacity[valid]
            scales = scales[valid]
            rotations = rotations[valid]
            colors = colors[valid]
            lbs = lbs[valid]
            init_points = init_points[valid]
        
        if not self.training and self.mask_out_mano:
            valid = self.mano_mask < 0.5
            cano_pts = cano_pts[valid]
            opacity = opacity[valid]
            scales = scales[valid]
            rotations = rotations[valid]
            colors = colors[valid]
            lbs = lbs[valid]
            init_points = init_points[valid]

        gaussian_vals = {
            'positions': cano_pts,
            'opacity': opacity,
            'scales': scales,
            'rotations': rotations,
            'colors': colors,
            # 'max_sh_degree': self.max_sh_degree
            'max_sh_degree': 0,
        }

        nonrigid_offset = cano_pts - init_points

        gaussian_vals = self.transform_cano2live(lbs, gaussian_vals, items)

        if not self.training and self.mask_out_flame and self.replace_flame and flame_gs is not None:
            flame_lbs = torch.zeros((flame_gs['positions'].shape[0], 55), dtype=torch.float32, device=config.device)
            flame_lbs[..., 15] = 1.0
            flame_gs = self.transform_cano2live(flame_lbs, flame_gs, items)

            gaussian_vals = {
                'positions': torch.cat([gaussian_vals['positions'], flame_gs['positions']], 0),
                'opacity': torch.cat([gaussian_vals['opacity'], flame_gs['opacity']], 0),
                'scales': torch.cat([gaussian_vals['scales'], flame_gs['scales']], 0),
                'rotations': torch.cat([gaussian_vals['rotations'], flame_gs['rotations']], 0),
                'colors': torch.cat([gaussian_vals['colors'], flame_gs['colors']], 0),
                'max_sh_degree': 0,
            }

        render_ret = render3(
            gaussian_vals,
            bg_color,
            items['extr'],
            items['intr'],
            items['img_w'],
            items['img_h']
        )
        rgb_map = render_ret['render'].permute(1, 2, 0)
        mask_map = render_ret['mask'].permute(1, 2, 0)

        ret = {
            'rgb_map': rgb_map,
            'mask_map': mask_map,
            'offset': nonrigid_offset,
            'pos_map': pos_map,
            'cano_pts': cano_pts,
            'opacity': opacity,
            'scales': scales,
            'rotations': rotations,
        }

        if not self.training:
            ret.update({
                'cano_tex_map': color_map,
                'posed_gaussians': gaussian_vals
            })

        return ret
