import os
from os.path import join
import numpy as np
import torch
import torch.nn.functional as F
import cv2 as cv
import trimesh
import math
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

from easyvolcap.utils.base_utils import *
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import load_dotdict, to_tensor, to_cuda

from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    OrthographicCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesUV,
    TexturesVertex
)
from pytorch3d.renderer.mesh.shader import ShaderBase, HardDepthShader, HardPhongShader
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.ops import knn_gather


class VertexAtrriShader(ShaderBase):
    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        texels = meshes.sample_textures(fragments)
        blend_params = kwargs.get("blend_params", self.blend_params)

        images = hard_rgb_blend(texels, fragments, blend_params)
        return images


class Renderer:
    def __init__(self, img_w: int, img_h: int, mvp = None, shader_name = 'vertex_attribute', bg_color = (0, 0, 0), win_name = None, device = 'cuda'):
        self.img_w = img_w
        self.img_h = img_h
        self.device = device
        raster_settings = RasterizationSettings(
            image_size = (img_h, img_w),
            blur_radius = 0.0,
            faces_per_pixel = 1,
            bin_size = None,
            max_faces_per_bin = 50000
        )

        self.shader_name = shader_name
        blend_params = BlendParams(background_color = bg_color)
        if shader_name == 'vertex_attribute':
            shader = VertexAtrriShader(device = device, blend_params = blend_params)
        elif shader_name == 'position':
            shader = VertexAtrriShader(device = device, blend_params = blend_params)
        elif shader_name == 'phong_geometry':
            shader = HardPhongShader(device = device, blend_params = blend_params)
        else:
            raise ValueError('Invalid shader_name')
        self.renderer = MeshRenderer(
            rasterizer = MeshRasterizer(
                cameras = None,
                raster_settings = raster_settings
            ),
            shader = shader
        )

        self.mesh = None

    def set_camera(self, extr, intr = None):
        affine_mat = np.identity(4, np.float32)
        affine_mat[0, 0] = -1
        affine_mat[1, 1] = -1
        extr = affine_mat @ extr
        extr[:3, :3] = np.linalg.inv(extr[:3, :3])
        extr = torch.from_numpy(extr).to(torch.float32).to(self.device)
        if intr is None:  # assume orthographic projection
            cameras = OrthographicCameras(
                focal_length = ((self.img_w / 2., self.img_h / 2.),),
                principal_point = ((self.img_w / 2., self.img_h / 2.),),
                R = extr[:3, :3].unsqueeze(0),
                T = extr[:3, 3].unsqueeze(0),
                in_ndc = False,
                device = self.device,
                image_size = ((self.img_h, self.img_w),)
            )
        else:
            intr = torch.from_numpy(intr).to(torch.float32).to(self.device)
            cameras = PerspectiveCameras(((intr[0, 0], intr[1, 1]),),
                                         ((intr[0, 2], intr[1, 2]),),
                                         extr[:3, :3].unsqueeze(0),
                                         extr[:3, 3].unsqueeze(0),
                                         in_ndc = False,
                                         device = self.device,
                                         image_size = ((self.img_h, self.img_w),))
        self.renderer.rasterizer.cameras = cameras

    def set_model(self, vertices, vertex_attributes = None):
        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices)
        if vertex_attributes is not None:
            if isinstance(vertex_attributes, np.ndarray):
                vertex_attributes = torch.from_numpy(vertex_attributes)
            vertex_attributes = vertex_attributes.to(torch.float32).to(self.device)
        vertices = vertices.to(torch.float32).to(self.device)
        faces = torch.arange(0, vertices.shape[0], dtype = torch.int64).to(self.device).reshape(-1, 3)

        if self.shader_name == 'vertex_attribute':
            textures = TexturesVertex([vertex_attributes])
        elif self.shader_name == 'position':
            textures = TexturesVertex([vertices])
        else:
            textures = TexturesVertex([torch.ones_like(vertices)])

        self.mesh = Meshes([vertices], [faces], textures = textures)

    def render(self):
        img = self.renderer(self.mesh, cameras = self.renderer.rasterizer.cameras)
        return img[0].cpu().numpy()


class CanoBlendWeightVolume:
    def __init__(self, data_path):
        if not os.path.exists(data_path):
            raise FileNotFoundError('# CanoBlendWeightVolume is not found from %s' % data_path)
        data = np.load(data_path)

        diff_weight_volume = data['diff_weight_volume']
        diff_weight_volume = diff_weight_volume.transpose((3, 0, 1, 2))[None]
        # base_weight_volume = base_weight_volume.transpose((3, 2, 1, 0))[None]
        self.diff_weight_volume = torch.from_numpy(diff_weight_volume).to(torch.float32).to(config.device)
        self.res_x, self.res_y, self.res_z = self.diff_weight_volume.shape[2:]
        self.joint_num = self.diff_weight_volume.shape[1]

        self.ori_weight_volume = torch.from_numpy(data['ori_weight_volume'].transpose((3, 0, 1, 2))[None]).to(torch.float32).to(config.device)

        if 'sdf_volume' in data:
            smpl_sdf_volume = data['sdf_volume']
            if len(smpl_sdf_volume.shape) == 3:
                smpl_sdf_volume = smpl_sdf_volume[..., None]
            smpl_sdf_volume = smpl_sdf_volume.transpose((3, 0, 1, 2))[None]
            self.smpl_sdf_volume = torch.from_numpy(smpl_sdf_volume).to(torch.float32).to(config.device)

        self.volume_bounds = torch.from_numpy(data['volume_bounds']).to(torch.float32).to(config.device)
        self.center = torch.from_numpy(data['center']).to(torch.float32).to(config.device)
        self.smpl_bounds = torch.from_numpy(data['smpl_bounds']).to(torch.float32).to(config.device)

        volume_len = self.volume_bounds[1] - self.volume_bounds[0]
        self.voxel_size = volume_len / torch.tensor([self.res_x-1, self.res_y-1, self.res_z-1]).to(volume_len)
        # self.base_gradient_volume = compute_gradient_volume(self.diff_weight_volume[0], self.voxel_size)  # [joint_num, 3, X, Y, Z]

    def forward_weight(self, pts, requires_scale = True, volume_type = 'diff'):
        """
        :param pts: (B, N, 3)
        :param requires_scale: bool, scale pts to [0, 1]
        :return: (B, N, 24)
        """
        if requires_scale:
            pts = (pts - self.volume_bounds[None, None, 0]) / (self.volume_bounds[1] - self.volume_bounds[0])[None, None]
        B, N, _ = pts.shape
        grid = 2 * pts - 1
        grid = grid[..., [2, 1, 0]]
        grid = grid[:, :, None, None]

        weight_volume = self.diff_weight_volume if volume_type == 'diff' else self.ori_weight_volume

        base_w = F.grid_sample(weight_volume.expand(B, -1, -1, -1, -1),
                               grid,
                               mode = 'bilinear',
                               padding_mode = 'border',
                               align_corners = True)
        base_w = base_w[:, :, :, 0, 0].permute(0, 2, 1)
        return base_w

    def forward_weight_grad(self, pts, requires_scale = True):
        """
        :param pts: (B, N, 3)
        :param requires_scale: bool, scale pts to [0, 1]
        :return: (B, N, 24)
        """
        if requires_scale:
            pts = (pts - self.volume_bounds[None, None, 0]) / (self.volume_bounds[1] - self.volume_bounds[0])[None, None]
        B, N, _ = pts.shape
        grid = 2 * pts - 1
        grid = grid.reshape(-1, 3)[:, [2, 1, 0]]
        grid = grid[None, :, None, None]

        base_g = F.grid_sample(self.base_gradient_volume.view(self.joint_num * 3, self.res_x, self.res_y, self.res_z)[None].expand(B, -1, -1, -1, -1),
                               grid,
                               mode = 'nearest',
                               padding_mode = 'border',
                               align_corners = True)
        base_g = base_g[:, :, :, 0, 0].permute(0, 2, 1).reshape(B, N, -1, 3)
        return base_g

    def forward_sdf(self, pts, requires_scale = True):
        if requires_scale:
            pts = (pts - self.volume_bounds[None, None, 0]) / (self.volume_bounds[1] - self.volume_bounds[0])[None, None]
        B, N, _ = pts.shape
        grid = 2 * pts - 1
        grid = grid.reshape(-1, 3)[:, [2, 1, 0]]
        grid = grid[None, :, None, None]

        sdf = F.grid_sample(self.smpl_sdf_volume.expand(B, -1, -1, -1, -1),
                            grid,
                            padding_mode = 'border',
                            align_corners = True)
        sdf = sdf[:, :, :, 0, 0].permute(0, 2, 1)

        return sdf


def save_pos_map(pos_map, path):
    mask = np.linalg.norm(pos_map, axis = -1) > 0.
    positions = pos_map[mask]
    print('Point nums %d' % positions.shape[0])
    pc = trimesh.PointCloud(positions)
    pc.export(path)


def nearest_face_pytorch3d(points, vertices, faces):
    """
    :param points: (B, N, 3)
    :param vertices: (B, M, 3)
    :param faces: (F, 3)
    :return dists (B, N), indices (B, N), bc_coords (B, N, 3)
    """
    import posevocab_custom_ops
    B, N = points.shape[:2]
    F = faces.shape[0]
    dists, indices, bc_coords = [], [], []
    points = points.contiguous()
    for b in range(B):
        triangles = vertices[b, faces.reshape(-1).to(torch.long)].reshape(F, 3, 3)
        triangles = triangles.contiguous()

        l_idx = torch.tensor([0, ]).to(torch.long).to(points.device)
        dist, index, w0, w1, w2 = posevocab_custom_ops.nearest_face_pytorch3d(
            points[b],
            l_idx,
            triangles,
            l_idx,
            N
        )
        dists.append(torch.sqrt(dist))
        indices.append(index)
        bc_coords.append(torch.stack([w0, w1, w2], 1))

    dists = torch.stack(dists, 0)
    indices = torch.stack(indices, 0)
    bc_coords = torch.stack(bc_coords, 0)

    return dists, indices, bc_coords


def barycentric_interpolate(vert_attris, faces, face_ids, bc_coords):
    """
    :param vert_attris: (B, V, C)
    :param faces: (B, F, 3)
    :param face_ids: (B, N)
    :param bc_coords: (B, N, 3)
    :return inter_attris: (B, N, C)
    """
    selected_faces = torch.gather(faces, 1, face_ids.unsqueeze(-1).expand(-1, -1, 3))  # (B, N, 3)
    face_attris = knn_gather(vert_attris, selected_faces)  # (B, N, 3, C)
    inter_attris = (face_attris * bc_coords.unsqueeze(-1)).sum(-2)  # (B, N, C)
    return inter_attris


def interpolate_lbs(pts, vertices, faces, vertex_lbs):
    dists, indices, bc_coords = nearest_face_pytorch3d(
        torch.from_numpy(pts).to(torch.float32).cuda()[None],
        torch.from_numpy(vertices).to(torch.float32).cuda()[None],
        torch.from_numpy(faces).to(torch.int64).cuda()
    )
    # print(dists.mean())
    lbs = barycentric_interpolate(
        vert_attris = vertex_lbs[None].to(torch.float32).cuda(),
        faces = torch.from_numpy(faces).to(torch.int64).cuda()[None],
        face_ids = indices,
        bc_coords = bc_coords
    )
    return lbs[0].cpu().numpy()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--data_dir', type = str, help = 'Data directory.', default='./data/0116data')
    parser.add_argument('--type', type = str, choices=['smpl', 'smplh', 'smplx'], default = 'smplh')
    parser.add_argument('--size', type = int, default = 1024, help = 'Size of the positional map.')
    # parser.add_argument('-c', '--config_path', type = str, help = 'Configuration file path.')
    args = parser.parse_args()

    # opt = yaml.load(open(args.config_path, encoding = 'UTF-8'), Loader = yaml.FullLoader)
    # dataset_module = opt['train'].get('dataset', 'MvRgbDatasetAvatarReX')
    # MvRgbDataset = importlib.import_module('dataset.dataset_mv_rgb').__getattribute__(dataset_module)
    # dataset = MvRgbDataset(**opt['train']['data'])
    # data_dir, frame_list = dataset.data_dir, dataset.pose_list

    os.makedirs(join(args.data_dir, 'smpl_pos_map'), exist_ok = True)

    cano_renderer = Renderer(args.size, args.size, shader_name = 'vertex_attribute')

    # smpl_model = smplx.SMPLX(config.PROJ_DIR + '/smpl_files/smplx', gender = 'neutral', use_pca = False, num_pca_comps = 45, flat_hand_mean = True, batch_size = 1)
    if args.type == 'smplh':
        # smpl_model = smplx.SMPLX(model_path = config.PROJ_DIR + '/smpl_files/smplx', gender = gender, use_pca = False, num_pca_comps = 45, flat_hand_mean = True, batch_size = 1)
        from easymocap.bodymodel.smplx import SMPLHModel
        bodymodel_cfg = dotdict()
        bodymodel_cfg.model_path = 'data/bodymodels/smplhv1.2/neutral/model.npz'
        bodymodel_cfg.regressor_path = 'data/smplx/J_regressor_body25_smplh.txt'
        bodymodel_cfg.mano_path = 'data/bodymodels/manov1.2'
        bodymodel_cfg.cfg_hand = dotdict()
        bodymodel_cfg.cfg_hand.use_pca = True
        bodymodel_cfg.cfg_hand.use_flat_mean = False
        bodymodel_cfg.cfg_hand.num_pca_comps = 12
        smpl_model = SMPLHModel(**bodymodel_cfg, device='cpu')
    else:
        raise NotImplementedError
    
    # smpl_data = np.load(data_dir + '/smpl_params.npz')
    # smpl_data = {k: torch.from_numpy(v.astype(np.float32)) for k, v in smpl_data.items()}
    smpl_data = to_tensor(load_dotdict(join(args.data_dir, 'motion.npz')))

    with torch.no_grad():
        if args.type == 'smplh':
            cano_pose = torch.zeros(156, dtype=torch.float32)
            cano_pose[5] = math.radians(25)
            cano_pose[8] = math.radians(-25)
            shapes = smpl_data.shapes[0]
            params = dotdict()
            params.poses = cano_pose[None]
            params.shapes = shapes[None]
            # cano_smpl_v = cano_smpl.vertices[0].cpu().numpy()
            cano_smpl_v = smpl_model(**params)[0].cpu().numpy()
            cano_center = 0.5 * (cano_smpl_v.min(0) + cano_smpl_v.max(0))
            cano_smpl_v_min = cano_smpl_v.min()
            smpl_faces = smpl_model.faces_tensor.numpy().astype(np.int64)
        else:
            raise NotImplementedError

    if os.path.exists(join(args.data_dir, 'template.ply')):
        print('# Loading template from %s' % join(args.data_dir, 'template.ply'))
        template = trimesh.load(join(args.data_dir, 'template.ply'), process = False)
        using_template = True
    else:
        print(f'# Cannot find template.ply from {args.data_dir}, using {args.type} as template')
        template = trimesh.Trimesh(cano_smpl_v, smpl_faces, process = False)
        using_template = False

    cano_smpl_v = template.vertices.astype(np.float32)
    smpl_faces = template.faces.astype(np.int64)
    cano_smpl_v_dup = cano_smpl_v[smpl_faces.reshape(-1)]
    cano_smpl_n_dup = template.vertex_normals.astype(np.float32)[smpl_faces.reshape(-1)]

    os.makedirs(join(args.data_dir, 'smpl_pos_map'), exist_ok = True)

    # define front & back view matrices
    front_mv = np.identity(4, np.float32)
    front_mv[:3, 3] = -cano_center + np.array([0, 0, -10], np.float32)
    front_mv[1:3] *= -1

    back_mv = np.identity(4, np.float32)
    rot_y = cv.Rodrigues(np.array([0, np.pi, 0], np.float32))[0]
    back_mv[:3, :3] = rot_y
    back_mv[:3, 3] = -rot_y @ cano_center + np.array([0, 0, -10], np.float32)
    back_mv[1:3] *= -1

    # render canonical smpl position maps
    cano_renderer.set_model(cano_smpl_v_dup, cano_smpl_v_dup)
    cano_renderer.set_camera(front_mv)
    front_cano_pos_map = cano_renderer.render()[:, :, :3]

    cano_renderer.set_camera(back_mv)
    back_cano_pos_map = cano_renderer.render()[:, :, :3]
    back_cano_pos_map = cv.flip(back_cano_pos_map, 1)
    cano_pos_map = np.concatenate([front_cano_pos_map, back_cano_pos_map], 1)
    cv.imwrite(join(args.data_dir, 'smpl_pos_map', 'cano_smpl_pos_map.exr'), cano_pos_map)

    # render canonical smpl normal maps
    cano_renderer.set_model(cano_smpl_v_dup, cano_smpl_n_dup)
    cano_renderer.set_camera(front_mv)
    front_cano_nml_map = cano_renderer.render()[:, :, :3]

    cano_renderer.set_camera(back_mv)
    back_cano_nml_map = cano_renderer.render()[:, :, :3]
    back_cano_nml_map = cv.flip(back_cano_nml_map, 1)
    cano_nml_map = np.concatenate([front_cano_nml_map, back_cano_nml_map], 1)
    cv.imwrite(join(args.data_dir, 'smpl_pos_map', 'cano_smpl_nml_map.exr'), cano_nml_map)

    body_mask = np.linalg.norm(cano_pos_map, axis = -1) > 0.
    cano_pts = cano_pos_map[body_mask]
    if using_template:
        weight_volume = CanoBlendWeightVolume(join(args.data_dir, 'cano_weight_volume.npz'))
        pts_lbs = weight_volume.forward_weight(torch.from_numpy(cano_pts)[None].cuda())[0]
    else:
        pts_lbs = interpolate_lbs(cano_pts, cano_smpl_v, smpl_faces, smpl_model.weights)
        pts_lbs = torch.from_numpy(pts_lbs).cuda()
    np.save(join(args.data_dir, 'smpl_pos_map', 'init_pts_lbs.npy'), pts_lbs.cpu().numpy())

    smpl_model = smpl_model.cuda()
    params = to_cuda(params)
    smpl_data = to_cuda(smpl_data)
    N = smpl_data.poses.shape[0]

    # inv_cano_smpl_A = torch.linalg.inv(cano_smpl.A).cuda()
    A, _ = smpl_model.transform(params)
    inv_cano_smpl_A = torch.linalg.inv(A).cuda()
    body_mask = torch.from_numpy(body_mask).cuda()
    cano_pts = torch.from_numpy(cano_pts).cuda()
    pts_lbs = pts_lbs.cuda()
    
    for pose_idx in tqdm(range(N), desc = 'Generating positional maps...'):
        with torch.no_grad():
            # live_smpl_woRoot = smpl_model.forward(
            #     betas = smpl_data['betas'],
            #     # global_orient = smpl_data['global_orient'][pose_idx][None],
            #     # transl = smpl_data['transl'][pose_idx][None],
            #     body_pose = smpl_data['body_pose'][pose_idx][None],
            #     jaw_pose = smpl_data['jaw_pose'][pose_idx][None],
            #     expression = smpl_data['expression'][pose_idx][None],
            #     # left_hand_pose = smpl_data['left_hand_pose'][pose_idx][None],
            #     # right_hand_pose = smpl_data['right_hand_pose'][pose_idx][None]
            # )
            params = dotdict()
            params.poses = smpl_data.poses[pose_idx][None]
            params.shapes = smpl_data.shapes[pose_idx][None]
            live_smpl_woRoot = smpl_model(**params)
            live_A_woRoot, _ = smpl_model.transform(params)

        cano2live_jnt_mats_woRoot = torch.matmul(live_A_woRoot, inv_cano_smpl_A)[0]
        pt_mats = torch.einsum('nj,jxy->nxy', pts_lbs, cano2live_jnt_mats_woRoot)
        live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], cano_pts) + pt_mats[..., :3, 3]
        live_pos_map = torch.zeros((args.size, 2 * args.size, 3)).to(live_pts)
        live_pos_map[body_mask] = live_pts
        live_pos_map = F.interpolate(live_pos_map.permute(2, 0, 1)[None], None, [0.5, 0.5], mode = 'nearest')[0]
        live_pos_map = live_pos_map.permute(1, 2, 0).cpu().numpy()
        pcd = trimesh.PointCloud(live_pos_map.reshape(-1, 3)).export(join(args.data_dir, 'smpl_pos_map', '%06d.ply' % pose_idx))
        cv.imwrite(join(args.data_dir, 'smpl_pos_map', '%06d.exr' % pose_idx), live_pos_map)
