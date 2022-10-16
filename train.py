import argparse
import ast
import os
import warnings
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import colored_traceback
import gin
import imageio
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from gin.torch import external_configurables
from lietorch import SE3
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from crops import crop_inputs
from datasets import LD2DL, BOPDataset, collate_fn
from detector import load_detector
from pose_models import RaftSe3
from utils import (Pytorch3DRenderer, add_noise, depth_to_jet, gen,
                   geodesic_and_flow_loss, get_perturbations, metrics)


class AlwaysWarn(UserWarning):
    pass

warnings.simplefilter('always', AlwaysWarn)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--override', nargs='+', type=str, default=[], help="gin-config settings to override")
    parser.add_argument('--batch_size', type=int, default=2, help="batch size")
    parser.add_argument('--ds_workers', type=int, default=12) # This command doesn't work on Mac or Windows
    parser.add_argument('--no_spawn', action='store_true', help="don't spawn separate processes for training, even if # GPUs > 1")
    parser.add_argument('--num_inner_loops', type=int, default=10, help="number of inner-loops in each forward pass")
    parser.add_argument('--num_solver_steps', type=int, default=3, help="number of BD-PnP solver steps per inner-loop (doesn't affect Modified BD-PnP)")
    parser.add_argument('--load_weights', type=str, default=None, help='path to the model weights to load')
    parser.add_argument('--resume_step', type=str, default=None, help='path to the model weights to load')
    parser.add_argument('--evaluate', action='store_true', help="just evaluate the model, don't train")
    parser.add_argument('--dataset', required=True, choices=['ycbv', 'tless', 'lmo', 'hb', 'tudl', 'icbin', 'itodd'], help="dataset for training (and evaluation)")
    parser.add_argument('--rgb_only', action='store_true', help="use the RGB-only model")
    parser.add_argument('--pbr_only', action='store_true', help="use the PBR data")

    args = parser.parse_args()
    print('num worker', args.ds_workers)
    args.override = format_gin_override(args.override)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(12356 + np.random.randint(100))
    world_size = torch.cuda.device_count()
    assert world_size > 0, "You need a GPU!"
    colored_traceback.add_hook()
    smp = mp.get_context('spawn')
    train_metrics = smp.Queue()
    val_metrics = smp.Queue()
    if world_size == 1 or args.no_spawn:
        train(0, 1, args, train_metrics, val_metrics, args.batch_size)
    else:
        spwn_ctx = mp.spawn(train, nprocs=world_size, args=(world_size, args, train_metrics, val_metrics, args.batch_size//world_size), join=False)
        spwn_ctx.join()
    print("Done!")

def setup_ddp(rank, world_size):
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',     
    	world_size=world_size,                              
    	rank=rank)

    torch.manual_seed(0)
    torch.cuda.set_device(rank)

def format_gin_override(overrides):
    if len(overrides) > 0:
        print("Overriden parameters:", overrides)
    output = deepcopy(overrides)
    for i,o in enumerate(overrides):
        k,v = o.split('=')
        try:
            ast.literal_eval(v)
        except:
            output[i] = f'{k}="{v}"'
    return output

@gin.configurable
def gin_globals(**kwargs):
    return SimpleNamespace(**kwargs)

def make_datasets(dataset_splits):
    datasets = []
    for (kwargs, n_repeat) in dataset_splits:
        ds = BOPDataset(**kwargs)
        print(f'Loaded BOPDataset({kwargs}) with {len(ds)}x{n_repeat}={len(ds)*n_repeat} images.')
        datasets.extend([ds]*n_repeat)
    return torch.utils.data.ConcatDataset(datasets)

def create_dataloader(dataset, batch_size, world_size, rank, num_workers, training):
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=training, num_replicas=world_size, rank=rank, drop_last=training)
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, persistent_workers=(num_workers > 0),
        num_workers=num_workers, sampler=sampler, pin_memory=True, collate_fn=collate_fn, drop_last=training)

@gin.configurable
def get_optim(model, optimizer_cls, scheduler_cls, warmup_length):
    optimizer = optimizer_cls(model.parameters())

    lambd = lambda batch: (batch + 1) / warmup_length
    lr_scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambd)
    lr_scheduler_warmup.last_epoch = 0

    lr_scheduler = scheduler_cls(optimizer)
    lr_scheduler.last_epoch = -1

    combined_scheduler = lambda idx: lr_scheduler_warmup if idx < (warmup_length-1) else lr_scheduler

    return optimizer, combined_scheduler

def sample_obj(objects, generator):
    idx = torch.randint(len(objects),(1,), generator=generator).item()
    return objects[idx]

def check_param_usage(model):
    for name, param in model.named_parameters():
        if param.grad is None:
            warnings.warn(f"{name} is not used")

def load_raft_model(file_path):
    model = RaftSe3()
    if file_path:
        state_dict = torch.load(file_path)
        state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
        model.load_state_dict(state_dict)
        del state_dict
        print(f"Loaded RAFT model: {file_path}")
    model.cuda()
    return model

def train(rank, world_size, args, train_metrics, val_metrics, batch_size):
    colored_traceback.add_hook()
    print(f'world_size={world_size}')
    print(f"configs/train_{args.dataset}_{'rgb' if args.rgb_only else 'rgbd'}"
          f"{'_pbr' if args.pbr_only else ''}.gin")
    gin.parse_config_files_and_bindings(["configs/base.gin", f"configs/train_{args.dataset}_{'rgb' if args.rgb_only else 'rgbd'}"
                                                             f"{'_pbr' if args.pbr_only else ''}.gin"], args.override)
    # coordinate multiple GPUs
    setup_ddp(rank, world_size)

    if args.resume_step:
        global_step = int(args.resume_step)
        # model = load_raft_model(None)
        model = load_raft_model(f'checkpoints/{args.resume_step}.pth')
        optimizer, scheduler = get_optim(model)
        optimizer.load_state_dict(torch.load(f'checkpoints/{args.resume_step}_optimizer.pth'))
        scheduler(global_step).load_state_dict(torch.load(f'checkpoints/{args.resume_step}_scheduler.pth'))
    else:
        model = load_raft_model(args.load_weights)
        optimizer, scheduler = get_optim(model)
        global_step = 0
    model.train()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)
    train_dataset = make_datasets(gin_globals().train_splits)
    val_dataset = make_datasets(gin_globals().val_splits)
    val_indices = torch.randperm(len(val_dataset), generator=gen(0)).tolist()[:gin_globals().val_size]
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    train_loader = create_dataloader(train_dataset, batch_size, world_size, rank, num_workers=args.ds_workers//world_size, training=True)
    val_loader = create_dataloader(val_dataset, batch_size, world_size, rank, num_workers=0, training=False)

    detector = load_detector()

    if rank == 0:
        writer = SummaryWriter()


    for epoch in range(100000000):
        for epoch_step, data in enumerate(tqdm(train_loader, disable=(rank != 0 or args.evaluate))):
            if not args.evaluate:
                optimizer.zero_grad()
                assert model.training
                loss, forward_pass_metrics = forward_pass(model, data, generator=gen(global_step), detector=detector, \
                    num_inner_loops=args.num_inner_loops, num_solver_steps=args.num_solver_steps)
                assert model.training
                loss.backward()
                if global_step == 0:
                    check_param_usage(model)
                forward_pass_metrics["gradient_norm"] = torch.cat([v.grad.flatten() for v in model.parameters()]).pow(2).sum().sqrt().cpu().unsqueeze(0)

                assert all([((t.device == torch.device('cpu')) and (t.ndim == 1)) for t in forward_pass_metrics.values()])
                train_metrics.put(forward_pass_metrics)

                if (((global_step + 1) % gin_globals().print_freq) == 0) and (rank == 0):
                    queue_size = world_size * gin_globals().print_freq
                    writer.add_scalar("Learning_rate", optimizer.param_groups[0]['lr'], global_step)
                    print(f"Training Metrics at {epoch_step} ({global_step}), ep. {epoch}\n{'-'*20}")
                    log_metrics(train_metrics, queue_size, global_step, writer, "Training")
                    writer.add_text(f"Operative_Gin_Config", gin.operative_config_str(), global_step)
                    Path("operative_gin_config.txt").write_text(gin.operative_config_str())

                if (global_step >= gin_globals().print_freq) and (rank == 0):
                    assert Path("operative_gin_config.txt").exists()

                torch.nn.utils.clip_grad_norm_(model.parameters(), gin_globals().clip)
                optimizer.step()
                scheduler(global_step).step()

            if (global_step > 0 and ((global_step % gin_globals().val_freq) == 0)) or args.evaluate:
                with torch.no_grad():
                    model.eval()
                    print("Validating...")
                    for val_idx, data in enumerate(val_loader):
                        _, forward_pass_metrics = forward_pass(model, data, val_mode=True, generator=gen(val_idx), detector=detector)
                        if args.evaluate and (rank == 0):
                            for key in ['Accuracy/mspd_recall', 'Accuracy/mssd_recall']:
                                print(f"({val_idx}) {key}: {np.around(forward_pass_metrics[key].cpu().numpy(), 1)}")
                        assert all([((t.device == torch.device('cpu')) and (t.ndim == 1)) for t in forward_pass_metrics.values()])
                        val_metrics.put(forward_pass_metrics)

                    if rank == 0:
                        queue_size = len(val_loader) * world_size
                        print(f"Validation Metrics at {epoch_step} ({global_step}), ep. {epoch}\n{'-'*20}")
                        log_metrics(val_metrics, queue_size, global_step, writer, "Validation")
                        if not args.evaluate:
                            os.makedirs(f'checkpoints_{args.dataset}', exist_ok=True)
                            torch.save(model.state_dict(), f'checkpoints_{args.dataset}/{global_step}.pth')
                            torch.save(optimizer.state_dict(), f'checkpoints_{args.dataset}/{global_step}_optimizer.pth')
                            torch.save(scheduler(global_step).state_dict(), f'checkpoints_{args.dataset}/{global_step}_scheduler.pth')

                    if args.evaluate:
                        exit(0)
                    model.train()

            global_step += 1

            if global_step >= 200000:
                print(f"Reached 200000 steps. Finished training.")
                exit(0)

def log_metrics(queue, queue_size, global_step, writer, prefix):
    assert queue_size > 0
    assert queue.qsize() <= queue_size, (queue.qsize(), queue_size)
    if queue.qsize() != queue_size:
        warnings.warn(f"WARNING: qsize() is {queue.qsize()} != {queue_size}", AlwaysWarn)
    metrics_list = [queue.get(timeout=600) for _ in range(queue_size)]
    metrics_agg = {k:torch.cat(v).mean().item() for k,v in LD2DL(metrics_list).items()}
    max_key_len = max(map(len, metrics_agg.keys()))
    for key, val in metrics_agg.items():
        key = f"{prefix}/{key}"
        writer.add_scalar(key, val, global_step)
        print(f"{key.ljust(max_key_len)} : {val:.5}")

"""
objects 'label', 'name', 'TWO', 'T0O', 'visib_fract', 'id_in_segm', 'bbox', 'diameter_m', 'symmetries'
camera 'T0C', 'K', 'TWC', 'resolution', 'depth', 'interpolated_depth
frame_info 'scene_id', 'cam_id', 'view_id', 'cam_name'
"""
@gin.configurable
def forward_pass(model, data, generator, detector, use_detector, val_mode=False, num_inner_loops=40, num_solver_steps=10, debug=False):
    (images, masks, obs) = data
    batch_size = images.shape[0]
    images = images.to('cuda', torch.float).permute(0,3,1,2) / 255
    obs['camera'] = {k:(v.to('cuda') if torch.is_tensor(v) else v) for k,v in obs['camera'].items()}
    obs['objects'] = LD2DL([sample_obj(o,generator) for o in obs['objects']])
    for key in {'TWO', 'T0O', 'bbox', 'id_in_segm', 'diameter_m', 'symmetries'}:
        obs['objects'][key] = torch.as_tensor(np.array(obs['objects'][key]), device='cuda')
    assert obs['objects']['bbox'].min() >= 0, obs['objects']['bbox'].cpu().numpy()
    masks = (masks.cuda() == obs['objects']['id_in_segm'].view(-1, 1, 1))

    if val_mode or use_detector:
        """
        Replace the ground truth segmentation mask with one generated by a Mask-RCNN
        This is optional during training, but can discourage overfitting to perfect masks
        """
        detections = detector.get_detections(images=images, mask_th=0.5, detection_th=0.0)
        det_infos = detections.infos
        for b_idx, (gt_mask, obj_label) in enumerate(zip(masks, obs['objects']['label'])):
            relevant_detections = det_infos.loc[(det_infos['label'] == obj_label) & (det_infos["batch_im_id"] == b_idx)]
            if len(relevant_detections) > 0:
                relevant_masks = detections.masks[relevant_detections.index.tolist()]
                iou = (gt_mask & relevant_masks).sum(dim=[1,2]).double()/(gt_mask | relevant_masks).sum(dim=[1,2]).double()
                if iou.max() > 0.3: # Associate the predicted mask to the correct object by checking its IOU w/ the GT mask
                    masks[b_idx] = relevant_masks[iou.argmax()]
        del detections

    gt_pose = obs['camera']['TWC'].inverse() @ obs['objects']['TWO']
    input_pose = add_noise(gt_pose, generator=generator)
    images_cropped, K_cropped, _, _, masks_cropped, depths_cropped = crop_inputs(images=images, K=obs['camera']['K'], TCO=input_pose, \
        labels=obs['objects']['label'], masks=masks, sce_depth=obs['camera']['interpolated_depth'], render_size=obs['camera']['resolution'].div(2)[0])
    # Note: Sometimes the perturbation can push the input pose outside the bounds of the image, causing the input image to be blank.
    # This is rare, but occasionally happens on itodd.

    input_pose_multiview = get_perturbations(input_pose).flatten(0,1) if val_mode else input_pose
    Nr = input_pose_multiview.shape[0] // batch_size
    label_rep = np.repeat(obs['objects']['label'], Nr)
    K_rep = K_cropped.repeat_interleave(Nr, dim=0)
    res_rep = obs['camera']['resolution'].div(2).repeat_interleave(Nr, dim=0)
    rendered_rgb, rendered_depth, _ = Pytorch3DRenderer()(label_rep, input_pose_multiview, K_rep, res_rep)
    
    # Forward pass
    combine = lambda a, b: torch.cat((a.unflatten(0, (batch_size, Nr)), b.unsqueeze(1)), dim=1)
    images_input = combine(rendered_rgb, images_cropped)
    depths_input = combine(rendered_depth, depths_cropped)
    masks_input = combine(rendered_depth > 1e-3, masks_cropped)
    pose_input = combine(input_pose_multiview, input_pose)
    K_input = combine(K_rep, K_cropped)

    outputs = model(Gs=pose_input, images=images_input, \
        depths_fullres=depths_input, masks_fullres=masks_input, \
        intrinsics_mat=K_input, labels=obs['objects']['label'],
        num_inner_loops=num_inner_loops, num_solver_steps=num_solver_steps)

    if debug: # Activate using `-o forward_pass.debug=True`
        current_pose_est = SE3(outputs['Gs'][-1].contiguous()[:, -1]).matrix()
        final_rendered_rgb, _, _ = Pytorch3DRenderer()(obs['objects']['label'], current_pose_est, K_cropped, obs['camera']['resolution'].div(2))
        gt_rendered_rgb, _, _ = Pytorch3DRenderer()(obs['objects']['label'], gt_pose, K_cropped, obs['camera']['resolution'].div(2))
        Path("debug_images").mkdir(exist_ok=True)
        for n in range(batch_size):
            basename = f"b{n}_{obs['frame_info']['scene_id'][n]}_{obs['frame_info']['view_id'][n]}_{obs['objects']['label'][n]}"
            imageio.imwrite(f"debug_images/{basename}_A.png", gt_rendered_rgb[n].permute(1,2,0).mul(255).byte().cpu())
            imageio.imwrite(f"debug_images/{basename}_B.png", rendered_rgb.unflatten(0, (batch_size, Nr))[n,0].permute(1,2,0).mul(255).byte().cpu())
            imageio.imwrite(f"debug_images/{basename}_C.png", final_rendered_rgb[n].permute(1,2,0).mul(255).byte().cpu())
            imageio.imwrite(f"debug_images/{basename}_D.png", images_cropped[n].permute(1,2,0).mul(255).byte().cpu())
            imageio.imwrite(f"debug_images/{basename}_E.png", depth_to_jet(depths_cropped[n].cpu().numpy(), max_val=1.0))
            imageio.imwrite(f"debug_images/{basename}_F.png", images[n].permute(1,2,0).mul(255).byte().cpu())
            imageio.imwrite(f"debug_images/{basename}_G.png", depth_to_jet(obs['camera']['interpolated_depth'][n].cpu().numpy(), max_val=1.0))

    loss, metrics_dict = geodesic_and_flow_loss(outputs, gt_pose.unsqueeze(1) @ obs['objects']['symmetries'], intrinsics_mat=K_input, labels=obs['objects']['label'], N=Nr)

    with torch.no_grad():
        point_clouds = Pytorch3DRenderer().get_pointclouds(obs['objects']['label'])
        metrics_dict['Accuracy/mssd_recall_starting'] = metrics.calc_mssd_recall(input_pose, gt_pose, point_clouds, obs['objects']['symmetries'], obs['objects']['diameter_m'])
        metrics_dict['Accuracy/mssd_recall'] = metrics.calc_mssd_recall(outputs['Gs'][-1].matrix()[:, -1], gt_pose, point_clouds, obs['objects']['symmetries'], obs['objects']['diameter_m'])
        metrics_dict['Accuracy/mspd_recall_starting'] = metrics.calc_mspd_recall(input_pose, gt_pose, point_clouds, obs['objects']['symmetries'], obs['camera']['K'])
        metrics_dict['Accuracy/mspd_recall'] = metrics.calc_mspd_recall(outputs['Gs'][-1].matrix()[:, -1], gt_pose, point_clouds, obs['objects']['symmetries'], obs['camera']['K'])
        metrics_dict['Accuracy/rot_error_starting'] = metrics.calc_rot_error(input_pose, gt_pose, obs['objects']['symmetries'])
        metrics_dict['Accuracy/rot_error'] = metrics.calc_rot_error(outputs['Gs'][-1].matrix()[:, -1], gt_pose, obs['objects']['symmetries'])
        metrics_dict = {k:v.detach().cpu() for k,v in metrics_dict.items()}

    return loss, metrics_dict


if __name__ == '__main__':
    main()
