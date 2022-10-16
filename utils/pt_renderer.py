import os
import time
import math
import warnings
from pathlib import Path

import cv2
import gin
import imageio
import numpy as np
import torch
from fastcore.all import store_attr
from joblib import Memory
from plyfile import PlyData
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (AmbientLights, BlendParams, DirectionalLights,
                                MeshRasterizer, PerspectiveCameras,
                                PointLights, RasterizationSettings,
                                SoftPhongShader, HardFlatShader, TexturesUV, TexturesVertex)
# Data structures and functions for rendering
from pytorch3d.structures import Meshes, join_meshes_as_batch
from torch import nn
from tqdm import tqdm

temp_cache_dir = Path('temporary_cache')
temp_cache_dir.mkdir(exist_ok=True)
MEMORY = Memory(temp_cache_dir)
BOP_DS_DIR = Path("local_data/bop_datasets")

det = lambda t: np.linalg.det(t.detach().cpu().numpy())

@MEMORY.cache(verbose=0)
def read_ply(ply_path: Path):
    assert ply_path.exists()
    if ply_path.suffix == '.obj':
        return load_objs_as_meshes([ply_path])
    assert ply_path.suffix == '.ply'
    f = lambda a: torch.from_numpy(a).unsqueeze(0)
    plydata = PlyData.read(ply_path)
    xyz = f(np.stack([plydata['vertex'][e] for e in 'xyz'], axis=-1))
    vert_ind = f(np.stack(plydata['face']['vertex_indices'])).long()
    if ply_path.with_suffix('.png').exists():
        uv = f(np.stack([plydata['vertex']['texture_' + e] for e in 'uv'], axis=-1))
        img = imageio.imread(ply_path.with_suffix('.png'))
        lH, lW, _ = img.shape
        if max(lH, lW) > 2048:
            output_size = (2048, int(lW * (2048/lH))) if lH == max(lH, lW) else (int(lH * (2048/lW)), 2048)
            img = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)
        img = f(img).float() / 255
        texture = TexturesUV(maps=img, faces_uvs=vert_ind, verts_uvs=uv)
    elif 'red' in plydata['vertex']:
        xyz_color = f(np.stack([plydata['vertex'][e] for e in ('red', 'green', 'blue')], axis=-1)).float() / 255
        texture = TexturesVertex(verts_features=xyz_color)
    else:
        texture = TexturesVertex(verts_features=torch.ones_like(xyz))

    if 'nx' in plydata['vertex']:
        nxyz = f(np.stack([plydata['vertex']['n'+ e] for e in 'xyz'], axis=-1))
        return Meshes(verts=xyz, verts_normals=nxyz, faces=vert_ind, textures=texture)
    return Meshes(verts=xyz, faces=vert_ind, textures=texture)

class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf, fragments.pix_to_face


def resize_pointcloud(pt_cloud, size, seed=0):
    num_pts = pt_cloud.shape[0]
    indices = np.arange(num_pts)
    np.random.default_rng(seed).shuffle(indices)
    indices = np.tile(indices, math.ceil(size/num_pts))[:size]
    indices = torch.as_tensor(indices, device=pt_cloud.device)
    assert indices.numel() == size
    return pt_cloud[indices]


@gin.configurable
class Pytorch3DRenderer(nn.Module):

    def load_meshes(self):
        mesh_list = []
        model_dir = BOP_DS_DIR / self.dataset_name / "models"
        assert model_dir.exists(), f"{model_dir} does not exist"
        ply_filenames = sorted(list(model_dir.rglob("obj_*.ply")))
        for ply_filename in tqdm(ply_filenames, desc=f"Loading {self.dataset_name} meshes"):
            mesh_list.append(read_ply(ply_filename))
        Pytorch3DRenderer.ply_filenames = [p.stem for p in ply_filenames]
        return join_meshes_as_batch(mesh_list).to(self.device)

    def __init__(self, dataset_name: str, device):
        super().__init__()
        store_attr()
        if hasattr(Pytorch3DRenderer, 'mesh_lookup'):
            return

        initial_memory_usage = torch.cuda.memory_allocated()
        initial_time = time.time()

        Pytorch3DRenderer.mesh_lookup = self.load_meshes()
        Pytorch3DRenderer.mesh_lookup.scale_verts_(1e-3)

        p_clouds = []
        max_point_cloud_size = min(Pytorch3DRenderer.mesh_lookup.verts_padded().shape[1], 2600)
        for pts in Pytorch3DRenderer.mesh_lookup.verts_list():
            resized_pt_cloud = resize_pointcloud(pts, max_point_cloud_size)
            p_clouds.append(resized_pt_cloud)
        Pytorch3DRenderer.point_cloud_lookup = torch.stack(p_clouds)

        end_time = time.time()
        total_memory_used = (torch.cuda.memory_allocated() - initial_memory_usage)/1e6
        time.sleep(0.5)
        print(f"Pytorch3DRenderer uses {format(total_memory_used, '.2f')}MB. Loaded {len(Pytorch3DRenderer.mesh_lookup)} meshes. Init took {format(end_time - initial_time, '.2f')}s.")

    def get_pointclouds(self, label_strs):
        label_idxs = self._labels_2_idxs(label_strs)
        return Pytorch3DRenderer.point_cloud_lookup[label_idxs]

    def _labels_2_idxs(self, label_strs):
        for label_str in label_strs:
            if label_str not in Pytorch3DRenderer.ply_filenames:
                raise Exception(f"{label_str} is not in the {self.dataset_name} dataset. The valid labels are {', '.join(Pytorch3DRenderer.ply_filenames)}")
        return [(int(s.split('_')[1])-1) for s in label_strs]

    @gin.configurable(module='pt_renderer')
    def forward(self, label_strs: list, poses, K: torch.Tensor, resolution=(240, 320), re_render_iters=10, scale_res=1.0, shader_type='soft'):
        assert shader_type in {'soft', 'flat'}

        if type(label_strs) is torch.Tensor:
            label_strs = [f"obj_{str(n.item()).zfill(6)}" for n in label_strs]
        bsz = len(label_strs)
        assert poses.shape[0] == K.shape[0] == bsz, [poses.shape, K.shape, bsz]
        label_idxs = self._labels_2_idxs(label_strs)

        assert max(label_idxs) < len(Pytorch3DRenderer.mesh_lookup), (label_idxs, len(Pytorch3DRenderer.mesh_lookup))
        meshes = Pytorch3DRenderer.mesh_lookup[label_idxs]
    
        R_orig, T = poses[..., :3, :3].contiguous(), poses[..., :3, 3].contiguous()
        assert np.abs(det(R_orig)-1).max() < 1e-1, det(R_orig)
        R = R_orig.transpose(1,2)

        principal_point = K[:,:2,2] * scale_res
        focal_length = torch.stack((K[:,0,0], K[:,1,1]), dim=-1) * scale_res
        resolution = resolution.mul(scale_res).long()

        K_kwargs = dict(focal_length=-focal_length, in_ndc=False, image_size=resolution, principal_point=principal_point)
        assert np.abs(det(R)-1).max() < 1e-1, det(R)
        cameras = PerspectiveCameras(device=self.device, R=R, T=T, **K_kwargs)
        assert len(cameras) == bsz

        # We leave bin_size and max_faces_per_bin to their default values of None, which 
        # sets their values using heuristics and ensures that the faster coarse-to-fine 
        # rasterization method is used.  Refer to docs/notes/renderer.md for an 
        # explanation of the difference between naive and coarse-to-fine rasterization. 
        resolution = (resolution[0,0].item(), resolution[0,1].item())
        raster_settings = RasterizationSettings(
            image_size=resolution, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
            perspective_correct=False
        )

        # Slightly slower but necessary
        raster_settings.max_faces_per_bin=int(1e5)

        # Create a Phong renderer by composing a rasterizer and a shader. The textured 
        # Phong shader will interpolate the texture uv coordinates for each vertex, 
        # sample from a texture image and apply the Phong lighting model

        # Optional directional lights
        # direc = torch.tensor((-1.0,-1.0,-1.0)).to(T).view(1,-1,1)
        # direc = R@direc
        # lights = DirectionalLights(device=self.device, direction=direc.view(-1,3), specular_color=((0.0,)*3,), diffuse_color=((0.5,)*3,), ambient_color=((0.5,)*3,))     

        loc = torch.tensor((1.0,-1.0,-1.0)).to(T).view(1,-1,1) # flipped Z
        loc = (R@loc).view(-1,3)
        lights = PointLights(device=self.device, location=loc, specular_color=((0.0,)*3,), diffuse_color=((0.5,)*3,), ambient_color=((0.5,)*3,))     

        shader = HardFlatShader if (shader_type == 'flat') else SoftPhongShader

        renderer = MeshRendererWithDepth(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=shader(
                device=self.device, 
                cameras=cameras,
                lights=lights,
                blend_params=BlendParams(background_color=(0.0,)*3)
            )
        )
        for itr in range(re_render_iters):
            rgb, depth, pix_to_face = renderer(meshes, lights=lights)

            if rgb.max() > 1.1:
                warnings.warn(f"The rendered image is too bright! ({itr}) {rgb.mean().item()} {rgb.max().item()}")
                rgb.clamp_(min=0.0, max=1.0)

            if not torch.isnan(rgb).any() and (depth.amax(dim=[1,2,3]) > 0).all() and rgb.max() < 1.1:
                break

        pix_to_face = pix_to_face.squeeze(-1) - meshes.mesh_to_faces_packed_first_idx().view(bsz, 1, 1)
        pix_to_face[pix_to_face < 0] = -1

        assert not torch.isnan(depth).any()
        assert not torch.isnan(rgb[...,:3]).any(), [torch.isnan(rgb[...,i]).sum().item() for i in range(4)]
        assert -1e2 < rgb.min() and rgb.max() < 1.5 and rgb.dtype == depth.dtype == torch.float32, [rgb.min(), rgb.max()]

        # The renderer randomly adds a small value (e.g. 1e-5) uniformly to all pixels.
        # This is enough to significantly change the model output. The line below fixes the issue.
        if not rgb.requires_grad:
            rgb = rgb.mul(255).round().div(255)

        rgb, depth, pix_to_face = rgb.to(device=poses.device), depth.to(device=poses.device), pix_to_face.to(device=poses.device)

        return rgb.permute(0,3,1,2)[:,:3], depth.squeeze(-1).clamp(min=0), pix_to_face
