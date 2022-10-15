"""category-level pose data utils."""

import logging
import os
import math
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import _pickle as cPickle
from tqdm import tqdm
from lib.vis_utils.colormap import colormap


def occlude_obj_by_bboxes(bbox, mask):
    x1, y1, x2, y2 = bbox.type(torch.int).tolist()
    types = [0, 1, 2, 3]
    for a in types:
        # a = np.random.randint(4)
        occlude_mask = mask.clone()
        if a == 0:
            # mask right down area
            top_x = int(x1 * 0.75 + x2 * 0.25)
            top_y = int(y1 * 0.75 + y2 * 0.25)
            occlude_mask[top_x:x2, top_y:y2] = 0
        elif a == 1:
            # mask left down area
            end_x = int(x1 * 0.25 + x2 * 0.75)
            top_y = int(y1 * 0.75 + y2 * 0.25)
            occlude_mask[x1:end_x, top_y:y2] = 0
        elif a == 2:
            # mask left up area
            end_x = int(x1 * 0.25 + x2 * 0.75)
            end_y = int(y1 * 0.25 + y2 * 0.75)
            occlude_mask[x1:end_x, y1:end_y] = 0
        elif a == 3:
            # mask right up area
            top_x = int(x1 * 0.75 + x2 * 0.25)
            end_y = int(y1 * 0.25 + y2 * 0.75)
            occlude_mask[top_x:x2, y1:end_y] = 0
        else:
            raise NotImplementedError
        occlude_mask = occlude_mask.contiguous()
        occlude_ratio = occlude_mask.sum().item() / mask.sum().item()
        if occlude_ratio < 1.0:
            break

    return occlude_mask, occlude_ratio


def plot_xyz_axis(scales, Rs, ts, K, img, color_id=(1, 25, 50)):
    center_2d = [project(t, K) for t in ts]
    x_axis = [R[:, 0:1] * s[0] / 2 + t for s, R, t in zip(scales, Rs, ts)]
    y_axis = [R[:, 1:2] * s[1] / 2 + t for s, R, t in zip(scales, Rs, ts)]
    z_axis = [R[:, 2:3] * s[2] / 2 + t for s, R, t in zip(scales, Rs, ts)]
    x_2d = [project(t, K) for t in x_axis]
    y_2d = [project(t, K) for t in y_axis]
    z_2d = [project(t, K) for t in z_axis]

    colors = colormap(rgb=False, maximum=255)
    x_color = tuple(int(_c) for _c in colors[color_id[0]])
    y_color = tuple(int(_c) for _c in colors[color_id[1]])
    z_color = tuple(int(_c) for _c in colors[color_id[2]])
    for _i in range(len(scales)):
        # img_vis = misc.draw_projected_box3d(img_vis, bboxes_2d[_i])
        cv2.line(
            img,
            (center_2d[_i][0], center_2d[_i][1]),
            (x_2d[_i][0], x_2d[_i][1]),
            x_color,
            4,
        )
        cv2.line(
            img,
            (center_2d[_i][0], center_2d[_i][1]),
            (y_2d[_i][0], y_2d[_i][1]),
            y_color,
            4,
        )
        cv2.line(
            img,
            (center_2d[_i][0], center_2d[_i][1]),
            (z_2d[_i][0], z_2d[_i][1]),
            z_color,
            4,
        )
    return img


def load_obj(path_to_file):
    """Load obj file.

    Args:
        path_to_file: path

    Returns:
        vertices: ndarray
        faces: ndarray, index of triangle vertices
    """
    vertices = []
    faces = []
    with open(path_to_file, "r") as f:
        for line in f:
            if line[:2] == "v ":
                vertex = line[2:].strip().split(" ")
                vertex = [float(xyz) for xyz in vertex]
                vertices.append(vertex)
            elif line[0] == "f":
                face = line[1:].replace("//", "/").strip().split(" ")
                face = [int(idx.split("/")[0]) - 1 for idx in face]
                faces.append(face)
            else:
                continue
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    return vertices, faces


def create_sphere():
    # 642 verts, 1280 faces,
    verts, faces = load_obj("assets/sphere_mesh_template.obj")
    return verts, faces


def random_point(face_vertices):
    """Sampling point using Barycentric coordiante."""
    r1, r2 = np.random.random(2)
    sqrt_r1 = np.sqrt(r1)
    point = (
        (1 - sqrt_r1) * face_vertices[0, :]
        + sqrt_r1 * (1 - r2) * face_vertices[1, :]
        + sqrt_r1 * r2 * face_vertices[2, :]
    )

    return point


def pairwise_distance(A, B):
    """Compute pairwise distance of two point clouds.point.

    Args:
        A: n x 3 numpy array
        B: m x 3 numpy array

    Return:
        C: n x m numpy array
    """
    diff = A[:, :, None] - B[:, :, None].T
    C = np.sqrt(np.sum(diff**2, axis=1))

    return C


def uniform_sample(vertices, faces, n_samples, with_normal=False):
    """Sampling points according to the area of mesh surface."""
    sampled_points = np.zeros((n_samples, 3), dtype=float)
    normals = np.zeros((n_samples, 3), dtype=float)
    faces = vertices[faces]
    vec_cross = np.cross(faces[:, 1, :] - faces[:, 0, :], faces[:, 2, :] - faces[:, 0, :])
    face_area = 0.5 * np.linalg.norm(vec_cross, axis=1)
    cum_area = np.cumsum(face_area)
    for i in range(n_samples):
        face_id = np.searchsorted(cum_area, np.random.random() * cum_area[-1])
        sampled_points[i] = random_point(faces[face_id, :, :])
        normals[i] = vec_cross[face_id]
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    if with_normal:
        sampled_points = np.concatenate((sampled_points, normals), axis=1)
    return sampled_points


def farthest_point_sampling(points, n_samples):
    """Farthest point sampling."""
    selected_pts = np.zeros((n_samples,), dtype=int)
    dist_mat = pairwise_distance(points, points)
    # start from first point
    pt_idx = 0
    dist_to_set = dist_mat[:, pt_idx]
    for i in range(n_samples):
        selected_pts[i] = pt_idx
        dist_to_set = np.minimum(dist_to_set, dist_mat[:, pt_idx])
        pt_idx = np.argmax(dist_to_set)
    return selected_pts


def sample_points_from_mesh(path, n_pts, with_normal=False, fps=False, ratio=2):
    """Uniformly sampling points from mesh model.

    Args:
        path: path to OBJ file.
        n_pts: int, number of points being sampled.
        with_normal: return points with normal, approximated by mesh triangle normal
        fps: whether to use fps for post-processing, default False.
        ratio: int, if use fps, sample ratio*n_pts first, then use fps to sample final output.

    Returns:
        points: n_pts x 3, n_pts x 6 if with_normal = True
    """
    vertices, faces = load_obj(path)
    if fps:
        points = uniform_sample(vertices, faces, ratio * n_pts, with_normal)
        pts_idx = farthest_point_sampling(points[:, :3], n_pts)
        points = points[pts_idx]
    else:
        points = uniform_sample(vertices, faces, n_pts, with_normal)
    return points


def sample_bp_depth(image, depth, coord, mask=None):
    non_zero_mask = depth[:, :, -1] > 0
    if mask is not None:
        final_instance_mask = torch.logical_and(mask, non_zero_mask)
    else:
        final_instance_mask = non_zero_mask

    mask_flatten = final_instance_mask.flatten().nonzero()
    pts = depth.reshape(-1, 3)[mask_flatten].squeeze()
    rgb = image.reshape(-1, 3)[mask_flatten].squeeze()

    if coord is not None:
        assert coord.shape[-1] == 3
        nocs = coord.reshape(-1, 3)[mask_flatten].squeeze()
    else:
        nocs = None

    return rgb, pts, nocs


def backproject(depth, intrinsics, mask=None):
    """Backproject a depth map to a cloud map.

    :param depth: Input depth map [H, W]
    :param K: Intrinsics of the camera
    """
    assert depth.ndim == 2, depth.ndim
    H, W = depth.shape[:2]

    Y, X = torch.meshgrid(
        torch.arange(H, device=depth.device, dtype=depth.dtype) - intrinsics[1, 2],
        torch.arange(W, device=depth.device, dtype=depth.dtype) - intrinsics[0, 2],
        indexing="ij",
    )

    depth_bp = torch.stack((X * depth / intrinsics[0, 0], Y * depth / intrinsics[1, 1], depth), dim=2)

    non_zero_mask = depth > 0
    if mask is not None:
        final_instance_mask = torch.logical_and(mask, non_zero_mask)
    else:
        final_instance_mask = non_zero_mask

    mask_flatten = final_instance_mask.flatten().nonzero()
    pts = depth_bp.reshape(-1, 3)[mask_flatten]
    return pts.squeeze()


# NOTE: wrong implementation
# def backproject(depth, intrinsics, mask=None):
#     intrinsics_inv = torch.linalg.inv(intrinsics)
#     image_shape = depth.shape
#     width = image_shape[1]
#     height = image_shape[0]

#     non_zero_mask = (depth > 0)
#     if mask is not None:
#         final_instance_mask = torch.logical_and(mask, non_zero_mask)
#     else:
#         final_instance_mask = non_zero_mask

#     idxs = torch.where(final_instance_mask)
#     grid = torch.stack((idxs[1], height - idxs[0]), dim=0)

#     length = grid.shape[1]
#     ones = torch.ones([1, length])
#     uv_grid = torch.cat((grid, ones), dim=0)  # [3, num_pixel]

#     xyz = intrinsics_inv @ uv_grid  # [3, num_pixel]
#     xyz = xyz.permute(1, 0)  # [num_pixel, 3]

#     z = depth[idxs[0], idxs[1]]

#     pts = xyz * z[:, None] / xyz[:, -1:]

#     pts[:, 1] = -pts[:, 1]   # x, y is divided by |z| during projection --> here depth > 0 = |z| = -z

#     return pts


def crop_ball_from_pts(pts, center, radius, num_points=None, device=None, fps_sample=False):
    distance = torch.sqrt(((pts - center) ** 2).sum(-1))  # [N]
    radius = max(radius, 0.05)
    for i in range(10):
        idx = torch.where(distance <= radius)[0]
        if len(idx) >= 10 or num_points is None:
            break
        radius *= 1.10
    if num_points is not None:
        if len(idx) == 0:
            idx = torch.where(distance <= 1e9)[0]
        if len(idx) == 0:
            return idx
        while len(idx) < num_points:
            idx = torch.cat([idx, idx], dim=0)

        if fps_sample:
            sample_idx = farthest_point_sample(pts[idx], num_points, device)
        else:
            sample_idx = random_sample(pts[idx], num_points)
        idx = idx[sample_idx]

    return idx


def random_sample(xyz, npoint):
    # random sample
    idx = torch.randperm(len(xyz))[:npoint]
    while len(idx) < npoint:
        idx_ap = random_sample(xyz, npoint - len(idx))
        idx = torch.cat((idx, idx_ap), dim=0)
    return idx


def farthest_point_sample(xyz, npoint, device):
    """
    Input:
        xyz: pointcloud data, [N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint]
    """
    if device == "cpu":
        from core.utils.farthest_points_torch import farthest_points
        from torch.nn import functional as F

        _, idx = farthest_points(
            xyz,
            n_clusters=npoint,
            dist_func=F.pairwise_distance,
            return_center_indexes=True,
            init_center=True,
        )
        return idx
    else:
        from core.utils.pointnet_utils import farthest_point_sample as farthest_point_sample_cuda

        if len(xyz) > 5 * npoint:
            idx = torch.randperm(len(xyz))[:npoint]
            torch_xyz = torch.tensor(xyz[idx]).float().to(device).reshape(1, -1, 3)
            torch_idx = farthest_point_sample_cuda(torch_xyz, npoint)
            idx = idx[torch_idx.view(-1)]
            return idx
        else:
            torch_xyz = torch.tensor(xyz).float().to(device).reshape(1, -1, 3)
            torch_idx = farthest_point_sample_cuda(torch_xyz, npoint)
            idx = torch_idx.view(-1)
            return idx


def crop_mask_depth_image(
    image,
    depth,
    mask,
    coord=None,
    num_points=None,
):
    assert depth.shape[-1] == 3
    raw_rgb, raw_pts, raw_nocs = sample_bp_depth(image, depth, coord, mask)
    sample_idx = random_sample(raw_pts, num_points)
    pts = raw_pts[sample_idx]
    rgb = raw_rgb[sample_idx]

    if coord is not None:
        nocs = raw_nocs[sample_idx]
    else:
        nocs = None

    return rgb, pts, nocs


def crop_ball_from_depth_image(
    image, depth, mask, pose, scale, ratio, cam_intrinsics, coord=None, num_points=None, device=None, fps_sample=False
):
    assert depth.shape[-1] == 3

    center = pose[:, 3]
    radius = ratio * torch.norm(pose[:, :3] @ scale)

    raw_rgb, raw_pts, raw_nocs = sample_bp_depth(image, depth, coord, mask)

    idx = crop_ball_from_pts(raw_pts, center, radius, num_points, device=device, fps_sample=fps_sample)
    if len(idx) == 0:
        return crop_ball_from_depth_image(
            image, depth, mask, pose, scale, ratio * 1.2, cam_intrinsics, coord, num_points, device, fps_sample
        )

    rgb = raw_rgb[idx]
    pts = raw_pts[idx]
    nocs = raw_nocs[idx] if raw_nocs is not None else None

    return rgb, pts, nocs


def get_proj_corners(depth, center, radius, cam_intrinsics):
    radius = max(radius, 0.05)
    aa_corner = get_corners([center - np.ones(3) * radius * 1.0, center + np.ones(3) * radius * 1.0])
    aabb = bbox_from_corners(aa_corner)
    height, width = depth.shape
    projected_corners = project(aabb, cam_intrinsics).astype(np.int32)[:, [1, 0]]
    projected_corners[:, 0] = height - projected_corners[:, 0]
    corner_2d = np.stack([np.min(projected_corners, axis=0), np.max(projected_corners, axis=0)], axis=0)
    corner_2d[0, :] = np.maximum(corner_2d[0, :], 0)
    corner_2d[1, :] = np.minimum(corner_2d[1, :], np.array([height - 1, width - 1]))
    return corner_2d


def project(pts, intrinsics, scale=1000):  # not flipping y axis
    pts = pts * scale
    pts = -pts / pts[:, -1:]
    pts[:, -1] = -pts[:, -1]
    pts = np.transpose(intrinsics @ np.transpose(pts))[:, :2]
    return pts


def get_corners(points):  # [Bs, N, 3] -> [Bs, 2, 3]
    if isinstance(points, torch.Tensor):
        points = np.array(points.detach().cpu())
    pmin = np.min(points, axis=-2)  # [Bs, N, 3] -> [Bs, 3]
    pmax = np.max(points, axis=-2)  # [Bs, N, 3] -> [Bs, 3]
    return np.stack([pmin, pmax], axis=-2)


def bbox_from_corners(corners):  # corners [[3], [3]] or [Bs, 2, 3]
    if not isinstance(corners, np.ndarray):
        corners = np.array(corners)

    # bbox = np.zeros((8, 3))
    bbox_shape = corners.shape[:-2] + (8, 3)  # [Bs, 8, 3]
    bbox = np.zeros(bbox_shape)
    for i in range(8):
        x, y, z = (i % 4) // 2, i // 4, i % 2
        bbox[..., i, 0] = corners[..., x, 0]
        bbox[..., i, 1] = corners[..., y, 1]
        bbox[..., i, 2] = corners[..., z, 2]
    return bbox


def load_depth(depth_path):
    """Load depth image from img_path."""
    depth = cv2.imread(depth_path, -1)
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        # NOTE: RGB is actually BGR in opencv
        depth16 = depth[:, :, 1] * 256 + depth[:, :, 2]
        depth16 = np.where(depth16 == 32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == "uint16":
        depth16 = depth
    else:
        assert False, "[ Error ]: Unsupported depth type."
    return depth16


def get_bbox_from_scale(scale):
    """scale shape (3, )"""
    minx, maxx = -scale[0] / 2, scale[0] / 2
    miny, maxy = -scale[1] / 2, scale[1] / 2
    minz, maxz = -scale[2] / 2, scale[2] / 2

    bbox = np.array(
        [
            [maxx, maxy, maxz],
            [minx, maxy, maxz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, maxy, minz],
            [minx, maxy, minz],
            [minx, miny, minz],
            [maxx, miny, minz],
        ],
        dtype=np.float32,
    )
    return bbox


def get_bbox(bbox):
    """Compute square image crop window."""
    y1, x1, y2, x2 = bbox
    img_width = 480
    img_length = 640
    window_size = (max(y2 - y1, x2 - x1) // 40 + 1) * 40
    window_size = min(window_size, 440)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax
