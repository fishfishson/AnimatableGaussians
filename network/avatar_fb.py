import math
import os
from os.path import join
import cv2
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import smplx
from smplx.lbs import batch_rodrigues, batch_rot2aa
from network.styleunet.dual_styleunet import DualStyleUNet

from easyvolcap.utils.base_utils import *
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.net_utils import make_buffer, make_params

from pytorch3d.ops import knn_points
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion, matrix_to_axis_angle


def fov2focal(fov, pixels):
    return pixels / (2 * np.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * np.arctan(pixels / (2 * focal))


def getWorld2View(R: torch.Tensor, t: torch.Tensor):
    """
    R: ..., 3, 3
    T: ..., 3, 1
    """
    sh = R.shape[:-2]
    T = torch.eye(4, dtype=R.dtype, device=R.device)  # 4, 4
    for i in range(len(sh)):
        T = T.unsqueeze(0)
    T = T.expand(sh + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3:] = t
    return T


def getProjectionMatrix(K: torch.Tensor, H: torch.Tensor, W: torch.Tensor, znear: torch.Tensor, zfar: torch.Tensor):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    s = K[0, 1]
    one = K[2, 2]

    P = torch.zeros(4, 4, dtype=K.dtype, device=K.device)

    P[0, 0] = 2 * fx / W
    P[0, 1] = 2 * s / W
    P[0, 2] = -1 + 2 * (cx / W)

    P[1, 1] = 2 * fy / H
    P[1, 2] = -1 + 2 * (cy / H)

    P[2, 2] = -(zfar + znear) / (znear - zfar)
    P[2, 3] = 2 * zfar * znear / (znear - zfar)

    P[3, 2] = one

    return P


def convert_to_gaussian_camera(K: torch.Tensor,
                               R: torch.Tensor,
                               T: torch.Tensor,
                               H: torch.Tensor,
                               W: torch.Tensor,
                               n: torch.Tensor,
                               f: torch.Tensor,
                               cpu_K: torch.Tensor,
                               cpu_R: torch.Tensor,
                               cpu_T: torch.Tensor,
                               cpu_H: int,
                               cpu_W: int,
                               cpu_n: float = 0.01,
                               cpu_f: float = 100.,
                               ):
    output = dotdict()

    output.image_height = cpu_H
    output.image_width = cpu_W

    output.K = K
    output.R = R
    output.T = T

    output.znear = cpu_n
    output.zfar = cpu_f

    output.FoVx = focal2fov(cpu_K[0, 0].cpu(), cpu_W.cpu())  # MARK: MIGHT SYNC IN DIST TRAINING, WHY?
    output.FoVy = focal2fov(cpu_K[1, 1].cpu(), cpu_H.cpu())  # MARK: MIGHT SYNC IN DIST TRAINING, WHY?

    # Use .float() to avoid AMP issues
    output.world_view_transform = getWorld2View(R, T).transpose(0, 1).float()  # this is now to be right multiplied
    output.projection_matrix = getProjectionMatrix(K, H, W, n, f).transpose(0, 1).float()  # this is now to be right multiplied
    output.full_proj_transform = torch.matmul(output.world_view_transform, output.projection_matrix).float()   # 4, 4
    output.camera_center = (-R.mT @ T)[..., 0].float()  # B, 3, 1 -> 3,

    # Set up rasterization configuration
    output.tanfovx = np.tan(output.FoVx * 0.5)
    output.tanfovy = np.tan(output.FoVy * 0.5)

    return output


class Embedder(nn.Module):
    def __init__(self, N_freqs, input_dims=3, include_input=True) -> None:
        super().__init__()
        self.log_sampling = True
        self.periodic_fns = [torch.sin, torch.cos]
        self.max_freq = N_freqs - 1
        self.N_freqs = N_freqs
        self.include_input = include_input
        self.input_dims = input_dims
        embed_fns = []
        if self.include_input:
            embed_fns.append(lambda x: x)

        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0.,
                                            self.max_freq, steps=self.N_freqs)
        else:
            freq_bands = torch.linspace(
                2.**0., 2.**self.max_freq, steps=self.N_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
        self.embed_fns = embed_fns
        self.dim_embeded = self.input_dims*len(self.embed_fns)

    def forward(self, inputs):
        output = torch.cat([fn(inputs) for fn in self.embed_fns], 2)
        return output
    

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, hidden_layers=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fcs = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(hidden_layers-1)]
        )
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        # input: B,V,d
        batch_size, N_v, input_dim = input.shape
        input_ori = input.reshape(batch_size*N_v, -1)
        h = input_ori
        for i, l in enumerate(self.fcs):
            h = self.fcs[i](h)
            h = F.relu(h)
        output = self.output_linear(h)
        output = output.reshape(batch_size, N_v, -1)

        return output


class AvatarNet(nn.Module):
    def __init__(self,
                 inp_size: int = 512,
                 out_size: int = 1024,
                 random_style: bool = False,
                 with_viewdirs: bool = True,
                 weight_viewdirs: float = 1.0,
                 smpl_pos_map_dir: str = None,
                 cano_smplx_path: str = None,
                 use_flame_mask: bool = False,
                 use_mano_mask: bool = False,
                 model_path: str = './data/bodymodels/smplx/smplx',
                 gender: str = 'neutral',
                 use_compressed: bool = False,
                 use_face_contour: bool = True,
                 num_betas: int = 16,
                 num_expression_coeffs: int = 100,
                 ):
        super(AvatarNet, self).__init__()

        self.inp_size = inp_size
        self.out_size = out_size
        self.random_style = random_style
        self.with_viewdirs = with_viewdirs
        self.weight_viewdirs = weight_viewdirs
        self.use_flame_mask = use_flame_mask
        self.use_mano_mask = use_mano_mask

        # load canonical smplx
        log(f'Loading canonical smplx from {smpl_pos_map_dir} ...')
        cano_smpl_map = cv2.imread(join(smpl_pos_map_dir, 'cano_smpl_pos_map.exr'), cv2.IMREAD_UNCHANGED)
        self.cano_smpl_map = make_buffer(torch.from_numpy(cano_smpl_map).to(torch.float32))
        cano_smpl_mask = np.stack(np.where(np.linalg.norm(cano_smpl_map, axis=-1) > 0.), axis=-1)
        self.cano_smpl_mask = make_buffer(torch.from_numpy(cano_smpl_mask).to(torch.int))
        self.init_points = make_buffer(self.cano_smpl_map[self.cano_smpl_mask[:, 0], self.cano_smpl_mask[:, 1]])
        self.lbs = make_buffer(torch.from_numpy(np.load(join(smpl_pos_map_dir, 'init_pts_lbs.npy'))).to(torch.float32))
        if use_flame_mask:
            # How to process flame points?
            # 1. cnn predicts flame base points + mlp refine
            # 2. cnn does not predict flame base points, only mlp 
            cano_flame_mask = cv2.imread(join(smpl_pos_map_dir, 'cano_smpl_flame_mask.exr'), cv2.IMREAD_UNCHANGED)
            cano_flame_mask = cano_flame_mask[cano_smpl_mask[:, 0], cano_smpl_mask[:, 1]]
            cano_flame_mask = np.where(np.linalg.norm(cano_flame_mask, axis=-1) > 0.)[0]
            self.cano_flame_mask = make_buffer(torch.from_numpy(cano_flame_mask).to(torch.int))
        if use_mano_mask:
            cano_mano_mask = cv2.imread(join(smpl_pos_map_dir, 'cano_smpl_mano_mask.exr'), cv2.IMREAD_UNCHANGED)
            cano_mano_mask = cano_mano_mask[cano_smpl_mask[:, 0], cano_smpl_mask[:, 1]]
            cano_mano_mask = np.where(np.linalg.norm(cano_mano_mask, axis=-1) > 0.)[0]
            self.cano_mano_mask = make_buffer(torch.from_numpy(cano_mano_mask).to(torch.int))
        
        # create body model
        log('Creating body model ...')
        self.body_model = smplx.SMPLXLayer(model_path=model_path,
                                           gender=gender,
                                           use_compressed=use_compressed,
                                           use_face_contour=use_face_contour,
                                           num_betas=num_betas,
                                           num_expression_coeffs=num_expression_coeffs)
        for p in self.body_model.parameters():
            p.requires_grad = False
        cano_smplx = np.load(cano_smplx_path)
        betas = torch.from_numpy(cano_smplx['betas']).to(torch.float32)
        body_pose = np.zeros(63, dtype=np.float32)
        body_pose[2] = math.radians(25)
        body_pose[5] = math.radians(-25)
        body_pose = torch.from_numpy(body_pose).to(torch.float32)
        body_pose = batch_rodrigues(body_pose.view(-1, 3)).view(1, -1, 3, 3)
        cano_smplx = self.body_model(betas=betas, body_pose=body_pose)
        self.cano_smplx_A = make_buffer(cano_smplx.A)
        self.inverse_cano_smplx_A = make_buffer(torch.inverse(cano_smplx.A))

        # create canonical gaussian model
        log('Creating canonical gaussian model ...')
        dist2 = torch.clamp_min(knn_points(self.init_points[None].cuda(), self.init_points[None].cuda(), K=4)[0][0, :, 1:].mean(-1), 0.0000001).cpu()
        _scaling = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        self._scaling = make_buffer(_scaling)
        _rotation = torch.zeros(self.init_points.shape[0], 4, dtype=torch.float32)
        _rotation[:, 0] = 1.
        self._rotation = make_buffer(_rotation)
        _opacity = self.inverse_opacity_activation(0.1 * torch.ones((self.init_points.shape[0], 1), dtype=torch.float32))
        self._opacity = make_buffer(_opacity)

        log('Creating networks ...')
        self.color_net = DualStyleUNet(inp_size=self.inp_size, inp_ch=3, out_ch=3, out_size=self.out_size, style_dim=512, n_mlp=2)
        self.position_net = DualStyleUNet(inp_size=self.inp_size, inp_ch=3, out_ch=3, out_size=self.out_size, style_dim=512, n_mlp=2)
        self.other_net = DualStyleUNet(inp_size=self.inp_size, inp_ch=3, out_ch=8, out_size=self.out_size, style_dim=512, n_mlp=2)

        self.color_style = make_buffer(torch.ones([1, self.color_net.style_dim], dtype=torch.float32) / np.sqrt(self.color_net.style_dim))
        self.position_style = make_buffer(torch.ones([1, self.position_net.style_dim], dtype=torch.float32) / np.sqrt(self.position_net.style_dim))
        self.other_style = make_buffer(torch.ones([1, self.other_net.style_dim], dtype=torch.float32) / np.sqrt(self.other_net.style_dim))

        if self.with_viewdirs:
            cano_nml_map = cv2.imread(join(smpl_pos_map_dir, 'cano_smpl_nml_map.exr'), cv2.IMREAD_UNCHANGED)
            self.cano_nml_map = make_buffer(torch.from_numpy(cano_nml_map).to(torch.float32))
            self.cano_nmls = make_buffer(self.cano_nml_map[self.cano_smpl_mask[:, 0], self.cano_smpl_mask[:, 1]])
            self.viewdir_net = nn.Sequential(
                nn.Conv2d(1, 64, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Conv2d(64, 128, 4, 2, 1)
            )
        
        if use_flame_mask:
            self.flame_embedder = Embedder(N_freqs=6)
            self.flame_position_mlp = MLP(
                input_dim=self.flame_embedder.dim_embeded + 100 + 3,
                output_dim=3,
                hidden_dim=64,
                hidden_layers=4,
            )
            self.flame_others_mlp = MLP(
                input_dim=self.flame_embedder.dim_embeded + 100 + 3,
                output_dim=8,
                hidden_dim=64,
                hidden_layers=4,
            )
            self.flame_colors_mlp = MLP(
                input_dim=self.flame_embedder.dim_embeded + 100 + 3,
                output_dim=3,
                hidden_dim=64,
                hidden_layers=4,
            )

        if use_mano_mask:
            pass

    @staticmethod
    def scaling_activation(x):
        return torch.exp(x)

    @staticmethod
    def inverse_scaling_activation(x):
        return torch.log(x)

    @staticmethod
    def rotation_activation(x):
        return F.normalize(x, dim=-1)

    @staticmethod
    def opacity_activation(x):
        return torch.sigmoid(x)
    
    @staticmethod
    def inverse_opacity_activation(x):
        return torch.logit(x)

    # def generate_mean_hands(self):
    #     # print('# Generating mean hands ...')
    #     import glob
    #     # get hand mask
    #     lbs_argmax = self.lbs.argmax(1)
    #     self.hand_mask = lbs_argmax == 20
    #     self.hand_mask = torch.logical_or(self.hand_mask, lbs_argmax == 21)
    #     self.hand_mask = torch.logical_or(self.hand_mask, lbs_argmax >= 25)

    #     pose_map_paths = sorted(glob.glob(config.opt['train']['data']['data_dir'] + '/smpl_pos_map/%08d.exr' % config.opt['test']['fix_hand_id']))
    #     smpl_pos_map = cv.imread(pose_map_paths[0], cv.IMREAD_UNCHANGED)
    #     pos_map_size = smpl_pos_map.shape[1] // 2
    #     smpl_pos_map = np.concatenate([smpl_pos_map[:, :pos_map_size], smpl_pos_map[:, pos_map_size:]], 2)
    #     smpl_pos_map = smpl_pos_map.transpose((2, 0, 1))
    #     pose_map = torch.from_numpy(smpl_pos_map).to(torch.float32).to(config.device)
    #     pose_map = pose_map[:3]

    #     cano_pts = self.get_positions(pose_map)
    #     opacity, scales, rotations = self.get_others(pose_map)
    #     colors, color_map = self.get_colors(pose_map)

    #     self.hand_positions = cano_pts#[self.hand_mask]
    #     self.hand_opacity = opacity#[self.hand_mask]
    #     self.hand_scales = scales#[self.hand_mask]
    #     self.hand_rotations = rotations#[self.hand_mask]
    #     self.hand_colors = colors#[self.hand_mask]

    #     # # debug
    #     # hand_pts = trimesh.PointCloud(self.hand_positions.detach().cpu().numpy())
    #     # hand_pts.export('./debug/hand_template.obj')
    #     # exit(1)

    # def transform_cano2live(self, lbs, gaussian_vals, items):
    #     if 'cano2live_jnt_mats' in items:
    #         pt_mats = torch.einsum('nj,jxy->nxy', lbs, items['cano2live_jnt_mats'])
    #         positions = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], gaussian_vals['positions']) + pt_mats[..., :3, 3]
    #         rot_mats = quaternion_to_matrix(gaussian_vals['rotations'])
    #         rot_mats = torch.einsum('nxy,nyz->nxz', pt_mats[..., :3, :3], rot_mats)
    #     elif 'cano2live_jnt_mats_woRoot' in items:
    #         pt_mats = torch.einsum('nj,jxy->nxy', lbs, items['cano2live_jnt_mats_woRoot'])
    #         positions = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], gaussian_vals['positions']) + pt_mats[..., :3, 3]
    #         positions = (items['global_orient'] @ positions.mT + items['transl'][:, None]).mT
    #         rot_mats = quaternion_to_matrix(gaussian_vals['rotations'])
    #         rot_mats = torch.einsum('nxy,nyz->nxz', pt_mats[..., :3, :3], rot_mats)
    #         rot_mats = items['global_orient'] @ rot_mats
    #     gaussian_vals['positions'] = positions
    #     gaussian_vals['rotations'] = matrix_to_quaternion(rot_mats)

    #     return gaussian_vals
    
    def transform_cano2live(self, position, rotation, batch):
        batch_size = batch['betas'].shape[0]
        lbs = self.lbs.repeat(batch_size, 1, 1)
        if 'cano2live_jnt_mats' in batch:
            pt_mats = torch.einsum('bnj,bjxy->bnxy', lbs, batch['cano2live_jnt_mats'])
            position = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], position) + pt_mats[..., :3, 3]
            rot_mats = quaternion_to_matrix(rotation)
            rot_mats = torch.einsum('bnxy,bnyz->bnxz', pt_mats[..., :3, :3], rot_mats)
            rotation = matrix_to_quaternion(rot_mats)
        elif 'cano2live_woRT_jnt_mats' in batch:
            pt_mats = torch.einsum('bnj,bjxy->bnxy', lbs, batch['cano2live_woRT_jnt_mats'])
            position = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], position) + pt_mats[..., :3, 3]
            position = (batch['Rh'] @ position.mT + batch['Th'][..., None]).mT
            rot_mats = quaternion_to_matrix(rotation)
            rot_mats = torch.einsum('bnxy,bnyz->bnxz', pt_mats[..., :3, :3], rot_mats)
            rot_mats = batch['Rh'] @ rot_mats
            rotation = matrix_to_quaternion(rot_mats)
        
        return position, rotation

    def get_positions(self, pos_map: torch.Tensor, batch: dotdict = {}, return_delta: bool = False):
        position_map, _ = self.position_net([self.position_style], pos_map, randomize_noise=False)
        front_position_map, back_position_map = torch.split(position_map, [3, 3], dim=1)
        position_map = torch.cat([front_position_map, back_position_map], dim=3).permute(0, 2, 3, 1)
        delta_position = 0.05 * position_map[:, self.cano_smpl_mask[:, 0], self.cano_smpl_mask[:, 1]]
        if self.use_flame_mask and 'jaw_pose' in batch and 'expression' in batch:
            batch_size = batch['jaw_pose'].shape[0]
            points = self.init_points[self.cano_flame_mask][None].expand(batch_size, -1, -1)
            jaw_pose = matrix_to_axis_angle(batch['jaw_pose']).view(-1, 1, 3).expand(-1, points.shape[1], -1)
            expression = batch['expression'].view(-1, 1, 100).expand(-1, points.shape[1], -1)
            feat = torch.cat([self.flame_embedder(points), jaw_pose, expression], dim=-1)
            flame_position = self.flame_position_mlp(feat) + delta_position[:, self.cano_flame_mask]
            delta_position[:, self.cano_flame_mask] = flame_position
        if self.use_mano_mask and 'left_hand_pose' in batch and 'right_hand_pose' in batch:
            pass
        if return_delta:
            return delta_position
        else: 
            positions = delta_position + self.init_points[None]
            return positions

    def get_others(self, pos_map: torch.Tensor, batch: dotdict = {}, return_delta: bool = False):
        other_map, _ = self.other_net([self.other_style], pos_map, randomize_noise=False)
        front_map, back_map = torch.split(other_map, [8, 8], 1)
        other_map = torch.cat([front_map, back_map], dim=3).permute(0, 2, 3, 1)
        others = other_map[:, self.cano_smpl_mask[:, 0], self.cano_smpl_mask[:, 1]]  # (B, N, 8)
        # delta_opacity, delta_scaling, delta_rotation = torch.split(others, [1, 3, 4], dim=-1)
        delta_opacity = others[..., 0:1]
        delta_scaling = others[..., 1:4]
        delta_rotation = others[..., 4:8]
        if self.use_flame_mask and 'jaw_pose' in batch and 'expression' in batch:
            batch_size = batch['jaw_pose'].shape[0]
            points = self.init_points[self.cano_flame_mask][None].expand(batch_size, -1, -1)
            jaw_pose = matrix_to_axis_angle(batch['jaw_pose']).view(-1, 1, 3).expand(-1, points.shape[1], -1)
            expression = batch['expression'].view(-1, 1, 100).expand(-1, points.shape[1], -1)
            feat = torch.cat([self.flame_embedder(points), jaw_pose, expression], dim=-1)
            flame_opacity, flame_scaling, flame_rotation = torch.split(self.flame_others_mlp(feat), [1, 3, 4], dim=-1)
            flame_opacity = flame_opacity + delta_opacity[:, self.cano_flame_mask]
            flame_scaling = flame_scaling + delta_scaling[:, self.cano_flame_mask]
            flame_rotation = flame_rotation + delta_rotation[:, self.cano_flame_mask]
            delta_opacity[:, self.cano_flame_mask] = flame_opacity
            delta_scaling[:, self.cano_flame_mask] = flame_scaling
            delta_rotation[:, self.cano_flame_mask] = flame_rotation
        if self.use_mano_mask and 'left_hand_pose' in batch and 'right_hand_pose' in batch:
            pass
        if return_delta: 
            return delta_opacity, delta_scaling, delta_rotation
        else:
            opacity = self.opacity_activation(delta_opacity + self._opacity[None])
            scaling = self.scaling_activation(delta_scaling + self._scaling[None])
            rotation = self.rotation_activation(delta_rotation + self._rotation[None])
            return opacity, scaling, rotation

    def get_colors(self, pos_map: torch.Tensor, front_viewdirs: torch.Tensor = None, back_viewdirs: torch.Tensor = None, batch: dotdict = {}, return_map: bool = False):
        color_style = torch.rand_like(self.color_style) if self.random_style and self.training else self.color_style
        color_map, _ = self.color_net([color_style], pos_map, 
                                      randomize_noise=False, 
                                      view_feature1=front_viewdirs, 
                                      view_feature2=back_viewdirs)
        front_color_map, back_color_map = torch.split(color_map, [3, 3], dim=1)
        color_map = torch.cat([front_color_map, back_color_map], dim=3).permute(0, 2, 3, 1)
        colors = color_map[:, self.cano_smpl_mask[:, 0], self.cano_smpl_mask[:, 1]]
        if self.use_flame_mask and 'jaw_pose' in batch and 'expression' in batch:
            batch_size = batch['jaw_pose'].shape[0]
            points = self.init_points[self.cano_flame_mask][None].expand(batch_size, -1, -1)
            jaw_pose = matrix_to_axis_angle(batch['jaw_pose']).view(-1, 1, 3).expand(-1, points.shape[1], -1)
            expression = batch['expression'].view(-1, 1, 100).expand(-1, points.shape[1], -1)
            feat = torch.cat([self.flame_embedder(points), jaw_pose, expression], dim=-1)
            flame_colors = self.flame_colors_mlp(feat) + colors[:, self.cano_flame_mask]
            colors[:, self.cano_flame_mask] = flame_colors
        if self.use_mano_mask and 'left_hand_pose' in batch and 'right_hand_pose' in batch:
            pass
        if return_map:
            return colors, color_map
        else:
            return colors

    def get_viewdir_feat(self, batch: dotdict):
        batch_size = batch['betas'].shape[0]
        lbs = self.lbs.repeat(batch_size, 1, 1)
        init_points = self.init_points.repeat(batch_size, 1, 1)
        cano_nmls = self.cano_nmls.repeat(batch_size, 1, 1)
        with torch.no_grad():
            if 'cano2live_jnt_mats' in batch:
                pt_mats = torch.einsum('bnj,bjxy->bnxy', lbs, batch['cano2live_jnt_mats'])
                live_pts = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], init_points) + pt_mats[..., :3, 3]
                live_nmls = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], cano_nmls)
            elif 'cano2live_woRT_jnt_mats' in batch:
                pt_mats = torch.einsum('bnj,bjxy->bnxy', lbs, batch['cano2live_woRT_jnt_mats'])
                live_pts = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], init_points) + pt_mats[..., :3, 3]
                live_pts = (batch['Rh'] @ live_pts.mT + batch['Th'][..., None]).mT
                live_nmls = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], cano_nmls)
                live_nmls = (batch['Rh'] @ live_nmls.mT).mT
            cam_pos = -torch.matmul(torch.inverse(batch['R']), batch['T']).mT
            viewdirs = F.normalize(cam_pos - live_pts, dim=-1, eps=1e-3)
            if self.training:
                viewdirs += torch.randn_like(viewdirs) * 0.1
            viewdirs = F.normalize(viewdirs, dim=-1)
            viewdirs = (live_nmls * viewdirs).sum(dim=-1)

            viewdirs_map = torch.zeros((batch_size, *self.cano_nml_map.shape[:2]), dtype=viewdirs.dtype, device=viewdirs.device)
            viewdirs_map[:, self.cano_smpl_mask[:, 0], self.cano_smpl_mask[:, 1]] = viewdirs

            viewdirs_map = viewdirs_map[:, None]
            viewdirs_map = F.interpolate(viewdirs_map, None, 0.5, 'nearest')
            front_viewdirs, back_viewdirs = torch.split(viewdirs_map, [self.inp_size, self.inp_size], -1)

        front_viewdirs = self.weight_viewdirs * self.viewdir_net(front_viewdirs)
        back_viewdirs = self.weight_viewdirs * self.viewdir_net(back_viewdirs)
        return front_viewdirs, back_viewdirs

    # def get_pose_map(self, items):
    #     pt_mats = torch.einsum('nj,jxy->nxy', self.lbs, items['cano2live_jnt_mats_woRoot'])
    #     live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.init_points) + pt_mats[..., :3, 3]
    #     live_pos_map = torch.zeros_like(self.cano_smpl_map)
    #     live_pos_map[self.cano_smpl_mask] = live_pts
    #     live_pos_map = F.interpolate(live_pos_map.permute(2, 0, 1)[None], None, [0.5, 0.5], mode = 'nearest')[0]
    #     live_pos_map = torch.cat(torch.split(live_pos_map, [self.inp_size, self.inp_size], 2), 0)
    #     items.update({
    #         'smpl_pos_map': live_pos_map
    #     })
    #     return live_pos_map

    def render_diff_gauss(self, xyz3: torch.Tensor, occ1: torch.Tensor, scale3: torch.Tensor, rot4: torch.Tensor, rgb3: torch.Tensor, camera: dotdict, bg_color: torch.Tensor):
        from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
        raster_settings = GaussianRasterizationSettings(
            image_height=camera.image_height,
            image_width=camera.image_width,
            tanfovx=camera.tanfovx,
            tanfovy=camera.tanfovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=0,
            campos=camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        scr = torch.zeros_like(xyz3, requires_grad=False) + 0
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        rendered_image, rendered_depth, rendered_alpha, _ = rasterizer(
            means3D=xyz3, 
            means2D=scr,
            shs=None,
            colors_precomp=rgb3,
            opacities=occ1,
            scales=scale3,
            rotations=rot4,
            cov3D_precomp=None,
        )

        rgb = rendered_image[None]
        acc = rendered_alpha[None]
        dpt = rendered_depth[None]
        return rgb, acc, dpt

    def render(self, batch):
        with torch.no_grad():
            live_smplx_woRT = self.body_model(
                transl=batch['transl'],
                global_orient=batch['global_orient'],
                betas=batch['betas'], 
                body_pose=batch['body_pose'], 
                left_hand_pose=batch['left_hand_pose'], 
                right_hand_pose=batch['right_hand_pose'],
                jaw_pose=batch['jaw_pose'],
                expression=batch['expression'],
            )
            cano2live_woRT_jnt_mats = torch.matmul(live_smplx_woRT.A, self.inverse_cano_smplx_A)
            batch['cano2live_woRT_jnt_mats'] = cano2live_woRT_jnt_mats

        if 'smpl_pos_map_pca' in batch:
            pos_map = batch['smpl_pos_map_pca'][:, :3]
        elif 'smpl_pos_map_vae' in batch:
            pos_map = batch['smpl_pos_map_vae'][:, :3]
        else:
            pos_map = batch['smpl_pos_map'][:, :3]

        cano_pts = self.get_positions(pos_map, batch=batch)
        opacity, scaling, cano_rot = self.get_others(pos_map, batch=batch)
        # if not self.training:
        #     scales = torch.clip(scales, 0., 0.03)
        if self.with_viewdirs:
            front_viewdirs, back_viewdirs = self.get_viewdir_feat(batch)
        else:
            front_viewdirs, back_viewdirs = None, None
        colors = self.get_colors(pos_map, front_viewdirs, back_viewdirs, batch=batch)
        live_pts, cano_rot = self.transform_cano2live(cano_pts, cano_rot, batch)

        cameras = []
        for i in range(len(batch.H)):
            camera = convert_to_gaussian_camera(batch.K[i], batch.R[i], batch.T[i],
                                                batch.H[i], batch.W[i], batch.n[i], batch.f[i],
                                                batch.K[i].cpu(), batch.R[i].cpu(), batch.T[i].cpu(),
                                                batch.H[i].cpu(), batch.W[i].cpu(), batch.n[i].item(), batch.f[i].item())
            cameras.append(camera)

        rgbs = []
        accs = []
        dpts = []
        for i in range(len(batch.H)):
            xyz3 = live_pts[i]
            rgb3 = colors[i]
            occ1 = opacity[i]
            scale3 = scaling[i]
            rot4 = cano_rot[i]
            camera = cameras[i]
            if 'bg_color' in batch:
                bg_color = batch.bg_color
            else:
                bg_color = torch.full([3], 0.0, device=xyz3.device, dtype=xyz3.dtype)
            rgb, acc, dpt = self.render_diff_gauss(xyz3, occ1, scale3, rot4, rgb3, camera, bg_color)
            
            rgbs.append(rgb)
            accs.append(acc)
            dpts.append(dpt)
        
        rgbs = torch.clamp(torch.cat(rgbs), 0.0, 1.0)
        accs = torch.cat(accs)
        dpts = torch.cat(dpts)
        offset = cano_pts - self.init_points[None]

        output = dotdict({
            'rgb': rgbs,
            'acc': accs,
            'dpt': dpts,
            'offset': offset,
        })
        return output


if __name__ == "__main__":
    net = AvatarNet(
        size=1024,
        inp_size=512,
        out_size=1024,
        random_style=False,
        with_viewdirs=True,
        data_dir='./data/30min_data_0'
    )