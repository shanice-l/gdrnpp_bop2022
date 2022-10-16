import argparse
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import colored_traceback
import gin
import imageio
import numpy as np
import pandas as pd
import torch
from lietorch import SE3

from crops import crop_inputs
from detector import PandasTensorCollection, concatenate
from train import (create_dataloader, gin_globals, load_raft_model,
                   make_datasets, format_gin_override)
from utils import Pytorch3DRenderer, get_perturbations, transform_pts

from joblib import Memory
MEMORY = Memory('joblib_cache', verbose=2)
# from utils.transform_cosypose import Transform
from utils.transforms import mat2SE3
import detector.tensor_collection as tc
import json
import pickle
import pycocotools.mask as cocomask
F = torch.nn.functional

LOCAL_DATA_DIR = Path('local_data')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def align_pointclouds_to_boxes(boxes_2d, model_points_3d, K):
    assert boxes_2d.shape[-1] == 4
    assert boxes_2d.dim() == 2
    bsz = boxes_2d.shape[0]
    z_guess = 1.0
    fxfy = K[:, [0, 1], [0, 1]]
    cxcy = K[:, [0, 1], [2, 2]]
    TCO = torch.tensor([
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [-1, 0, 0, z_guess],
        [0, 0, 0, 1]
    ]).to(torch.float).to(boxes_2d.device).repeat(bsz, 1, 1)
    bb_xy_centers = (boxes_2d[:, [0, 1]] + boxes_2d[:, [2, 3]]) / 2
    xy_init = ((bb_xy_centers - cxcy) * z_guess) / fxfy
    TCO[:, :2, 3] = xy_init

    C_pts_3d = transform_pts(TCO, model_points_3d)
    deltax_3d = C_pts_3d[:, :, 0].max(dim=1).values - C_pts_3d[:, :, 0].min(dim=1).values
    deltay_3d = C_pts_3d[:, :, 1].max(dim=1).values - C_pts_3d[:, :, 1].min(dim=1).values

    bb_deltax = (boxes_2d[:, 2] - boxes_2d[:, 0]) + 1
    bb_deltay = (boxes_2d[:, 3] - boxes_2d[:, 1]) + 1

    z_from_dx = fxfy[:, 0] * deltax_3d / bb_deltax
    z_from_dy = fxfy[:, 1] * deltay_3d / bb_deltay
    z = (z_from_dy.unsqueeze(1) + z_from_dx.unsqueeze(1)) / 2

    xy_init = ((bb_xy_centers - cxcy) * z) / fxfy
    TCO[:, :2, 3] = xy_init
    TCO[:, 2, 3] = z.flatten()
    return TCO


@gin.configurable
def generate_pose_from_detections(renderer, detections, K):
    K = K[detections.infos['batch_im_id'].values]
    boxes = detections.bboxes
    points_3d = renderer.get_pointclouds(detections.infos['label'])
    TCO_init = align_pointclouds_to_boxes(boxes, points_3d, K)
    return PandasTensorCollection(infos=detections.infos, poses=TCO_init)


def format_results(predictions):
    df = defaultdict(list)
    df = pd.DataFrame(df)
    results = dict(summary=dict(), summary_txt='',
        predictions=predictions, metrics=dict(),
        summary_df=df, dfs=dict())
    return results

def rle2mask(rle, height, width):
    if "counts" in rle and isinstance(rle["counts"], list):
        # if compact RLE, ignore this conversion
        # Magic RLE format handling painfully discovered by looking at the
        # COCO API showAnns function.
        rle = cocomask.frPyObjects(rle, height, width)
    mask = cocomask.decode(rle)
    return mask

@MEMORY.cache
def load_init_results(init_pose_path):
    infos, poses, bboxes, masks = [], [], [], []
    with open(init_pose_path, 'rb') as file:
        results = pickle.load(file)

    for key, im_results in results.items():
        scene_id, view_id = key.split('/')
        int_scene_id, int_view_id = int(scene_id), int(view_id)
        for inst_results in im_results:
            score = inst_results['score']

            if score < 0.03:
                continue

            R = inst_results['R']
            t = inst_results['t']
            pose = np.identity(4)
            pose[:3, :3] = R
            pose[:3, 3] = t
            poses.append(pose)
            bbox = inst_results['bbox_est']
            bboxes.append(bbox)
            obj_id = inst_results['obj_id']
            mask = inst_results['mask']
            mask = rle2mask(mask, mask['size'][0], mask['size'][1])
            masks.append(mask)
            label = f'obj_{obj_id:06d}'
            infos.append(dict(
                            scene_id=int_scene_id,
                            view_id=int_view_id,
                            score=score,
                            label=label,
                            time=0.0,
                        ))

    data = tc.PandasTensorCollection(
        infos=pd.DataFrame(infos),
        poses=torch.as_tensor(np.stack(poses)).float(),
        bboxes=torch.as_tensor(np.stack(bboxes)).float(),
        masks=torch.as_tensor(np.stack(masks)).bool(),
    ).cpu()
    return data



def filter_predictions_csv(preds, scene_id, im_id, obj_id=None, th=None):
    mask = preds['scene_id'] == scene_id
    mask = np.logical_and(mask, preds['im_id'] == im_id)
    if obj_id is not None:
        mask = np.logical_and(mask, preds['obj_id'] == obj_id)
    if th is not None:
        mask = np.logical_and(mask, preds['score'] >= th)
    if obj_id is not None:
        keep_ids = np.where(mask)[0]
    else:
        keep_ids = np.where(mask)
    preds = preds.loc[keep_ids]
    return preds

def filter_predictions_pd(preds, scene_id, view_id, th=None):
    mask = preds.infos['scene_id'] == scene_id
    mask = np.logical_and(mask, preds.infos['view_id'] == view_id)
    if th is not None:
        mask = np.logical_and(mask, preds.infos['score'] >= th)
    keep_ids = np.where(mask)
    infos = preds.infos.loc[keep_ids]
    poses = preds.poses[keep_ids]
    bboxes = preds.bboxes[keep_ids]
    masks = preds.masks[keep_ids]
    if len(keep_ids[0]) == 0:
        return None
    masks = torch.as_tensor(np.stack(masks)).bool()
    resize = (480, 640)
    masks = F.interpolate(masks.unsqueeze(0).float(), size=resize, mode='nearest')
    masks = masks.squeeze(0).bool()

    data = tc.PandasTensorCollection(
        infos=pd.DataFrame(infos),
        poses=torch.as_tensor(np.stack(poses)).float(),
        bboxes=torch.as_tensor(np.stack(bboxes)).float(),
        masks=masks,
    ).cuda()
    return data

def parse_Rt_in_csv(_item):
    return np.array([float(i) for i in _item.strip(" ").split(" ")])


def aug_poses_normal_np(poses, std_rot=15, std_trans=[0.01, 0.01, 0.05], max_rot=45, min_z=0.1):
    """
    Args:
        poses (ndarray): [n,3,4]
        std_rot: deg, randomly chosen from cfg.INPUT.NOISE_ROT_STD_{TRAIN|TEST}
        std_trans: [dx, dy, dz], cfg.INPUT.NOISE_TRANS_STD_{TRAIN|TEST}
        max_rot: deg, cfg.INPUT.NOISE_ROT_MAX_{TRAIN|TEST}
    Returns:
        poses_aug (ndarray): [n,3,4]
    """
    assert poses.ndim == 3, poses.shape
    poses_aug = poses.copy()
    bs = poses.shape[0]

    if isinstance(std_rot, (tuple, list, Sequence)):
        std_rot = np.random.choice(std_rot)

    euler_noises_deg = np.random.normal(loc=0, scale=std_rot, size=(bs, 3))
    if max_rot is not None:
        euler_noises_deg = np.clip(euler_noises_deg, -max_rot, max_rot)
    euler_noises_rad = euler_noises_deg * math.pi / 180.0
    rot_noises = np.array([euler2mat(*xyz) for xyz in euler_noises_rad])

    if mmcv.is_seq_of(std_trans, (tuple, list, Sequence)) and isinstance(std_trans[0], (tuple, list, Sequence)):
        sel_idx = np.random.choice(len(std_trans))
        sel_std_trans = std_trans[sel_idx]
    else:
        sel_std_trans = std_trans

    trans_noises = np.concatenate(
        [np.random.normal(loc=0, scale=std_trans_i, size=(bs, 1)) for std_trans_i in sel_std_trans],
        axis=1,
    )

    poses_aug[:, :3, :3] = rot_noises @ poses[:, :3, :3]
    poses_aug[:, :3, 3] += trans_noises
    # z should be >= min_z or > 0
    min_z = max(min_z, 1e-4)
    poses_aug[:, 2, 3] = np.clip(poses_aug[:, 2, 3], a_min=min_z, a_max=None)
    return poses_aug

@torch.no_grad()
def main():
    colored_traceback.add_hook()
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--override', nargs='+', type=str, default=[], help="gin-config settings to override")
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--num_images', type=int, default=None)
    parser.add_argument('--load_weights', type=str, default=None, help='path to the model weights to load')
    parser.add_argument('--init_pose_file', type=str, default=None, help='initial pose file path')
    parser.add_argument('--num_outer_loops', type=int, default=4, help="number of outer-loops in each forward pass")
    parser.add_argument('--num_inner_loops', type=int, default=40, help="number of inner-loops in each forward pass")
    parser.add_argument('--num_solver_steps', type=int, default=10, help="number of BD-PnP solver steps per inner-loop (doesn't affect Modified BD-PnP)")
    parser.add_argument('--save_dir', type=Path, default="test_evaluation")
    parser.add_argument('--dataset', required=True, choices=['ycbv', 'tless', 'tless_alignk', 'lmo', 'hb', 'tudl', 'icbin', 'itodd'], help="dataset for training (and evaluation)")
    parser.add_argument('--rgb_only', action='store_true', help="use the RGB-only model")
    args = parser.parse_args()
    init_data = load_init_results(args.init_pose_file)

    args.override = format_gin_override(args.override)
    gin.parse_config_files_and_bindings(["configs/base.gin", f"configs/test_{args.dataset}_{'rgb' if args.rgb_only else 'rgbd'}.gin"], args.override)
    test_dataset = make_datasets(gin_globals().test_splits)
    print(f"The entire dataset is of length {len(test_dataset)}")

    if 'SLURM_ARRAY_TASK_MIN' in os.environ: # array job
        assert int(os.environ['SLURM_ARRAY_TASK_MIN']) == 0
        num_jobs = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
        assert len(test_dataset)%num_jobs == 0
        num_images = len(test_dataset) // num_jobs
        start_index = int(os.environ['SLURM_ARRAY_TASK_ID']) * num_images
    else:
        num_images = args.num_images if (args.num_images is not None) else len(test_dataset)
        start_index = args.start_index
    print(f"Processing images in range [{start_index}, {start_index+num_images})")

    test_dataset = torch.utils.data.Subset(test_dataset, list(range(start_index, start_index+num_images)))
    assert len(test_dataset) == num_images, len(test_dataset)

    args.save_dir.mkdir(exist_ok=True)
    qual_output = args.save_dir / "qual_output"
    qual_output.mkdir(exist_ok=True)

    test_loader = create_dataloader(test_dataset, 1, 1, 0, num_workers=0, training=False)

    model = load_raft_model(args.load_weights)
    model.eval()

    all_preds = []
    all_preds_dict = {i:[] for i in range(args.num_outer_loops)}

    for image_index, (images, _, obs) in enumerate(test_loader):
        print(f"Processing image {image_index+1}/{num_images}")
        images = images.to('cuda', torch.float).permute(0,3,1,2) / 255
        obs['camera'] = {k:(v.to('cuda') if torch.is_tensor(v) else v) for k,v in obs['camera'].items()}

        scene_id, view_id = obs['frame_info']['scene_id'][0], obs['frame_info']['view_id'][0]

        detections = filter_predictions_pd(init_data, scene_id, view_id)

        if detections is None:
            print(f'scene_id view_id {scene_id} {view_id}')
            continue

        data_TCO_init = detections.clone()
        data_TCO_init.infos.loc[:,"scene_id"] = scene_id
        data_TCO_init.infos.loc[:,"view_id"] = view_id
        data_TCO_init.infos.loc[:,"time"] = 0.0

        for obj_idx, info in detections.infos.iterrows():
            obj_label = info.loc['label']
            should_save_img = (random.random() < gin_globals().save_img_prob)
            mrcnn_mask = detections.masks[[obj_idx]]
            mrcnn_pose = data_TCO_init.poses[[obj_idx]]


            basename = f"{scene_id}_{view_id}_{obj_label}_{obj_idx+1}"
            if 'SLURM_ARRAY_JOB_ID' in os.environ:
                with Path(f"slurm_outputs/{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}_log.txt").open("a") as f:
                    f.write(f"{basename}/{len(detections.infos)} || {time.strftime('%l:%M:%S %p on %b %d, %Y')}\n")
            print(f"{basename}/{len(detections.infos)} || {time.strftime('%l:%M:%S %p on %b %d, %Y')}\n")

            images_cropped, K_cropped, _, _, masks_cropped, depths_cropped = crop_inputs(images=images, K=obs['camera']['K'], TCO=mrcnn_pose, \
                labels=[obj_label], masks=mrcnn_mask, sce_depth=obs['camera']['interpolated_depth'], render_size=(240,320))

            initial_images_cropped_masked = images_cropped.clone()
            initial_images_cropped_masked = initial_images_cropped_masked.transpose(0, 1)
            initial_images_cropped_masked[:, ~masks_cropped] = 0
            initial_images_cropped_masked = initial_images_cropped_masked.transpose(0, 1)

            mrcnn_rendered_rgb, _, _ = Pytorch3DRenderer()([obj_label], mrcnn_pose, K_cropped, obs['camera']['resolution'].div(2))
            assert (mrcnn_rendered_rgb.shape == images_cropped.shape)
            current_pose_est = mrcnn_pose

            start_time = time.time()

            for outer_loop_idx in range(args.num_outer_loops):
                images_cropped, K_cropped, _, _, masks_cropped, depths_cropped = crop_inputs(images=images, K=obs['camera']['K'], TCO=current_pose_est, \
                    labels=[obj_label], masks=mrcnn_mask, sce_depth=obs['camera']['interpolated_depth'], render_size=(240,320))

                input_pose_multiview = get_perturbations(current_pose_est).flatten(0,1)
                Nr = input_pose_multiview.shape[0]

                label_rep = np.repeat([obj_label], Nr)
                K_rep = K_cropped.repeat_interleave(Nr, dim=0)
                res_rep = obs['camera']['resolution'].div(2).repeat_interleave(Nr, dim=0)
                rendered_rgb, rendered_depth, _ = Pytorch3DRenderer()(label_rep, input_pose_multiview, K_rep, res_rep)
                if should_save_img:
                    imageio.imwrite(qual_output / f"{basename}_B{outer_loop_idx}.png", rendered_rgb[0].permute(1,2,0).mul(255).byte().cpu())
                
                # Forward pass
                combine = lambda a, b: torch.cat((a.unflatten(0, (1, Nr)), b.unsqueeze(1)), dim=1)
                images_input = combine(rendered_rgb, images_cropped)
                depths_input = combine(rendered_depth, depths_cropped)
                masks_input = combine(rendered_depth > 1e-3, masks_cropped)
                pose_input = combine(input_pose_multiview, current_pose_est)
                K_input = combine(K_rep, K_cropped)

                outputs = model(Gs=pose_input, images=images_input, depths_fullres=depths_input, \
                    masks_fullres=masks_input, intrinsics_mat=K_input, labels=[obj_label], \
                        num_solver_steps=args.num_solver_steps, num_inner_loops=args.num_inner_loops)
                current_pose_est = SE3(outputs['Gs'][-1].contiguous()[:, -1]).matrix()
                batch_preds = PandasTensorCollection(data_TCO_init.infos[obj_idx:obj_idx + 1],
                                                     poses=current_pose_est.cpu())
                all_preds_dict[outer_loop_idx].append(batch_preds)

            process_time = time.time() - start_time
            data_TCO = data_TCO_init.infos[obj_idx:obj_idx+1]
            data_TCO.loc[:, 'time'] += process_time
            batch_preds = PandasTensorCollection(data_TCO, poses=current_pose_est.cpu())
            all_preds.append(batch_preds)

            # Saving qualitative output
            if should_save_img:
                final_rendered_rgb, _, _ = Pytorch3DRenderer()([obj_label], current_pose_est, K_cropped, obs['camera']['resolution'].div(2))
                imageio.imwrite(qual_output / f"{basename}_A.png", mrcnn_rendered_rgb[0].permute(1,2,0).mul(255).byte().cpu())
                imageio.imwrite(qual_output / f"{basename}_C.png", final_rendered_rgb[0].permute(1,2,0).mul(255).byte().cpu())
                imageio.imwrite(qual_output / f"{basename}_D.png", images_cropped[0].permute(1,2,0).mul(255).byte().cpu())
                images_cropped_masked = images_cropped.clone()
                images_cropped_masked = images_cropped_masked.transpose(0, 1)
                images_cropped_masked[:, ~masks_cropped] = 0
                images_cropped_masked = images_cropped_masked.transpose(0, 1)
                imageio.imwrite(qual_output / f"{basename}_E.png",
                                images_cropped_masked[0].permute(1, 2, 0).mul(255).byte().cpu())
                imageio.imwrite(qual_output / f"{basename}_F.png",
                                initial_images_cropped_masked[0].permute(1, 2, 0).mul(255).byte().cpu())


    all_preds = {f'maskrcnn_detections/refiner': concatenate(all_preds)}
    results = format_results(all_preds)
    output_filepath = args.save_dir / f'{gin_globals().dataset_name}_{start_index}_{start_index+num_images}_results.pth.tar'
    torch.save(results, output_filepath)
    for i in range(args.num_outer_loops):
        all_preds = all_preds_dict[i]
        all_preds = {f'maskrcnn_detections/refiner': concatenate(all_preds)}
        results = format_results(all_preds)
        output_filepath = args.save_dir / f'iter{i}'/ f'{gin_globals().dataset_name}_{start_index}_{start_index + num_images}_results.pth.tar'
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        torch.save(results, output_filepath)

    print("Done.")



if __name__ == '__main__':
    main()
