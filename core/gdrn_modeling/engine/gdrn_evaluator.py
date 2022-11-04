# -*- coding: utf-8 -*-
"""inference on dataset; save results; evaluate with bop_toolkit (if gt is
available)"""
import datetime
import itertools
import logging
import os.path as osp
import random
import time
from collections import OrderedDict

import cv2
import mmcv
import numpy as np
import ref
import torch
from torch.cuda.amp import autocast
from transforms3d.quaternions import quat2mat

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.layers import paste_masks_in_image
from detectron2.evaluation import DatasetEvaluator, DatasetEvaluators, inference_context
from detectron2.utils.logger import log_every_n_seconds, log_first_n

from core.utils.my_comm import all_gather, get_world_size, is_main_process, synchronize
from lib.pysixd import inout, misc
from lib.pysixd.pose_error import te
from lib.utils.mask_utils import binary_mask_to_rle
from lib.utils.utils import dprint
from lib.vis_utils.image import grid_show, vis_image_bboxes_cv2, vis_image_mask_cv2

from .engine_utils import batch_data, get_out_coor, get_out_mask, batch_data_inference_roi
from .test_utils import eval_cached_results, save_and_eval_results, to_list


logger = logging.getLogger(__name__)


class GDRN_Evaluator(DatasetEvaluator):
    """use bop toolkit to evaluate."""

    def __init__(self, cfg, dataset_name, distributed, output_dir, train_objs=None):
        self.cfg = cfg
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        # if test objs are just a subset of train objs
        self.train_objs = train_objs

        self._metadata = MetadataCatalog.get(dataset_name)
        self.data_ref = ref.__dict__[self._metadata.ref_key]
        self.obj_names = self._metadata.objs
        self.obj_ids = [self.data_ref.obj2id[obj_name] for obj_name in self.obj_names]
        # with contextlib.redirect_stdout(io.StringIO()):
        #     self._coco_api = COCO(self._metadata.json_file)
        self.model_paths = [
            osp.join(self.data_ref.model_eval_dir, "obj_{:06d}.ply".format(obj_id)) for obj_id in self.obj_ids
        ]
        self.models_3d = [
            inout.load_ply(model_path, vertex_scale=self.data_ref.vertex_scale) for model_path in self.model_paths
        ]
        if cfg.DEBUG or cfg.TEST.USE_DEPTH_REFINE:
            from lib.render_vispy.model3d import load_models
            from lib.render_vispy.renderer import Renderer

            if cfg.TEST.USE_DEPTH_REFINE:
                net_cfg = cfg.MODEL.POSE_NET
                width = net_cfg.OUTPUT_RES
                height = width
            else:
                width = self.data_ref.width
                height = self.data_ref.height

            self.ren = Renderer(size=(width, height), cam=self.data_ref.camera_matrix)
            self.ren_models = load_models(
                model_paths=self.data_ref.model_paths,
                scale_to_meter=0.001,
                cache_dir=".cache",
                texture_paths=self.data_ref.texture_paths if cfg.DEBUG else None,
                center=False,
                use_cache=True,
            )

        self.depth_refine_threshold = cfg.TEST.DEPTH_REFINE_THRESHOLD

        # eval cached
        if cfg.VAL.EVAL_CACHED or cfg.VAL.EVAL_PRINT_ONLY:
            eval_cached_results(self.cfg, self._output_dir, obj_ids=self.obj_ids)

    def reset(self):
        self._predictions = []

    def _maybe_adapt_label_cls_name(self, label):
        if self.train_objs is not None:
            cls_name = self.obj_names[label]
            if cls_name not in self.train_objs:
                return None, None  # this class was not trained
            label = self.train_objs.index(cls_name)
        else:
            cls_name = self.obj_names[label]
        return label, cls_name

    def get_fps_and_center(self, pts, num_fps=8, init_center=True):
        from core.csrc.fps.fps_utils import farthest_point_sampling

        avgx = np.average(pts[:, 0])
        avgy = np.average(pts[:, 1])
        avgz = np.average(pts[:, 2])
        fps_pts = farthest_point_sampling(pts, num_fps, init_center=init_center)
        res_pts = np.concatenate([fps_pts, np.array([[avgx, avgy, avgz]])], axis=0)
        return res_pts

    def get_img_model_points_with_coords2d(
        self, mask_pred_crop, xyz_pred_crop, coord2d_crop, im_H, im_W, extent, max_num_points=-1, mask_thr=0.5
    ):
        """
        from predicted crop_and_resized xyz, bbox top-left,
        get 2D-3D correspondences (image points, 3D model points)
        Args:
            mask_pred_crop: HW, predicted mask in roi_size
            xyz_pred_crop: HWC, predicted xyz in roi_size(eg. 64)
            coord2d_crop: HW2 coords 2d in roi size
            im_H, im_W
            extent: size of x,y,z
        """
        # [0, 1] --> [-0.5, 0.5] --> original
        xyz_pred_crop[:, :, 0] = (xyz_pred_crop[:, :, 0] - 0.5) * extent[0]
        xyz_pred_crop[:, :, 1] = (xyz_pred_crop[:, :, 1] - 0.5) * extent[1]
        xyz_pred_crop[:, :, 2] = (xyz_pred_crop[:, :, 2] - 0.5) * extent[2]

        coord2d_crop = coord2d_crop.copy()
        coord2d_crop[:, :, 0] = coord2d_crop[:, :, 0] * im_W
        coord2d_crop[:, :, 1] = coord2d_crop[:, :, 1] * im_H

        sel_mask = (
            (mask_pred_crop > mask_thr)
            & (abs(xyz_pred_crop[:, :, 0]) > 0.0001 * extent[0])
            & (abs(xyz_pred_crop[:, :, 1]) > 0.0001 * extent[1])
            & (abs(xyz_pred_crop[:, :, 2]) > 0.0001 * extent[2])
        )
        model_points = xyz_pred_crop[sel_mask].reshape(-1, 3)
        image_points = coord2d_crop[sel_mask].reshape(-1, 2)

        if max_num_points >= 4:
            num_points = len(image_points)
            max_keep = min(max_num_points, num_points)
            indices = [i for i in range(num_points)]
            random.shuffle(indices)
            model_points = model_points[indices[:max_keep]]
            image_points = image_points[indices[:max_keep]]
        return image_points, model_points

    def process(self, inputs, outputs, out_dict):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id", "scene_id".
            outputs: stores time
        """
        cfg = self.cfg
        if cfg.TEST.USE_PNP:
            if cfg.TEST.PNP_TYPE.lower() == "ransac_pnp":
                return self.process_pnp_ransac(inputs, outputs, out_dict)
            elif cfg.TEST.PNP_TYPE.lower() == "net_iter_pnp":
                return self.process_net_and_pnp(inputs, outputs, out_dict, pnp_type="iter")
            elif cfg.TEST.PNP_TYPE.lower() == "net_ransac_pnp":
                return self.process_net_and_pnp(inputs, outputs, out_dict, pnp_type="ransac")
            elif cfg.TEST.PNP_TYPE.lower() == "net_ransac_pnp_rot":
                # use rot from PnP/RANSAC and translation from Net
                return self.process_net_and_pnp(inputs, outputs, out_dict, pnp_type="ransac_rot")
            else:
                raise NotImplementedError

        if cfg.TEST.USE_DEPTH_REFINE:
            return self.process_depth_refine(inputs, outputs, out_dict)

        out_rots = out_dict["rot"].detach().to(self._cpu_device).numpy()
        out_transes = out_dict["trans"].detach().to(self._cpu_device).numpy()

        out_i = -1
        for i, (_input, output) in enumerate(zip(inputs, outputs)):
            json_results = []
            start_process_time = time.perf_counter()
            for inst_i in range(len(_input["roi_img"])):
                out_i += 1  # the index in the flattened output
                scene_im_id_split = _input["scene_im_id"][inst_i].split("/")
                K = _input["cam"][inst_i].cpu().numpy().copy()

                roi_label = _input["roi_cls"][inst_i]  # 0-based label
                score = _input["score"][inst_i]
                roi_label, cls_name = self._maybe_adapt_label_cls_name(roi_label)
                if cls_name is None:
                    continue

                # scene_id = int(scene_im_id_split[0])
                scene_id = scene_im_id_split[0]
                im_id = int(scene_im_id_split[1])
                obj_id = self.data_ref.obj2id[cls_name]

                # get pose
                rot_est = out_rots[out_i]
                trans_est = out_transes[out_i]
                pose_est = np.hstack([rot_est, trans_est.reshape(3, 1)])

                if cfg.DEBUG:  # visualize pose
                    file_name = _input["file_name"][inst_i]

                    # if f"{int(scene_id)}/{im_id}" != "9/47":
                    #     continue

                    im_ori = mmcv.imread(file_name, "color")
                    bbox = _input["bbox_est"][inst_i].cpu().numpy().copy()
                    im_vis = vis_image_bboxes_cv2(im_ori, [bbox], [f"{cls_name}_{score}"])

                    self.ren.clear()
                    self.ren.draw_background(mmcv.bgr2gray(im_ori, keepdim=True))
                    self.ren.draw_model(self.ren_models[self.data_ref.objects.index(cls_name)], pose_est)
                    ren_im, _ = self.ren.finish()
                    grid_show(
                        [ren_im[:, :, ::-1], im_vis[:, :, ::-1]],
                        [f"ren_im_{cls_name}", f"{scene_id}/{im_id}_{score}"],
                        row=1,
                        col=2,
                    )

                json_results.extend(
                    self.pose_prediction_to_json(
                        pose_est, scene_id, im_id, obj_id=obj_id, score=score, pose_time=output["time"], K=K
                    )
                )

            output["time"] += time.perf_counter() - start_process_time
            # process time for this image
            for item in json_results:
                item["time"] = output["time"]
            self._predictions.extend(json_results)

    def process_net_and_pnp(self, inputs, outputs, out_dict, pnp_type="iter"):
        """Initialize with network prediction (learned PnP) + iter PnP
        Args:
            inputs: the inputs to a model.
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id", "scene_id".
            pnp_type: iter | ransac (use ransac+EPnP)
            outputs:
        """
        cfg = self.cfg
        out_coor_x = out_dict["coor_x"].detach()
        out_coor_y = out_dict["coor_y"].detach()
        out_coor_z = out_dict["coor_z"].detach()
        out_xyz = get_out_coor(cfg, out_coor_x, out_coor_y, out_coor_z)
        out_xyz = out_xyz.to(self._cpu_device).numpy()

        out_mask = get_out_mask(cfg, out_dict["mask"].detach())
        out_mask = out_mask.to(self._cpu_device).numpy()

        out_rots = out_dict["rot"].detach().to(self._cpu_device).numpy()
        out_transes = out_dict["trans"].detach().to(self._cpu_device).numpy()

        out_i = -1
        for i, (_input, output) in enumerate(zip(inputs, outputs)):
            start_process_time = time.perf_counter()
            json_results = []
            for inst_i in range(len(_input["roi_img"])):
                out_i += 1
                bbox_center_i = _input["bbox_center"][inst_i]
                cx_i, cy_i = bbox_center_i
                scale_i = _input["scale"][inst_i]

                coord_2d_i = _input["roi_coord_2d"][inst_i].cpu().numpy().transpose(1, 2, 0)  # CHW->HWC
                im_H = _input["im_H"][inst_i].item()
                im_W = _input["im_W"][inst_i].item()

                scene_im_id_split = _input["scene_im_id"][inst_i].split("/")
                K = _input["cam"][inst_i].cpu().numpy().copy()

                roi_label = _input["roi_cls"][inst_i]  # 0-based label
                score = _input["score"][inst_i]
                roi_label, cls_name = self._maybe_adapt_label_cls_name(roi_label)
                if cls_name is None:
                    continue

                # scene_id = int(scene_im_id_split[0])
                scene_id = scene_im_id_split[0]
                im_id = int(scene_im_id_split[1])
                obj_id = self.data_ref.obj2id[cls_name]

                # get pose
                xyz_i = out_xyz[out_i].transpose(1, 2, 0)
                mask_i = np.squeeze(out_mask[out_i])

                img_points, model_points = self.get_img_model_points_with_coords2d(
                    mask_i,
                    xyz_i,
                    coord_2d_i,
                    im_H=im_H,
                    im_W=im_W,
                    extent=_input["roi_extent"][inst_i].cpu().numpy(),
                    mask_thr=cfg.MODEL.POSE_NET.GEO_HEAD.MASK_THR_TEST,
                )

                rot_est_net = out_rots[out_i]
                trans_est_net = out_transes[out_i]

                num_points = len(img_points)
                if num_points >= 4:
                    dist_coeffs = np.zeros(shape=[8, 1], dtype="float64")
                    points_2d = np.ascontiguousarray(img_points.astype(np.float64))
                    points_3d = np.ascontiguousarray(model_points.astype(np.float64))
                    camera_matrix = K.astype(np.float64)

                    rvec0, _ = cv2.Rodrigues(rot_est_net)

                    if pnp_type == "ransac":
                        points_3d = np.expand_dims(points_3d, 0)
                        points_2d = np.expand_dims(points_2d, 0)
                        _, rvec, t_est, _ = cv2.solvePnPRansac(
                            objectPoints=points_3d,
                            imagePoints=points_2d,
                            cameraMatrix=camera_matrix,
                            distCoeffs=dist_coeffs,
                            flags=cv2.SOLVEPNP_EPNP,
                            useExtrinsicGuess=True,
                            rvec=rvec0,
                            tvec=trans_est_net,
                            reprojectionError=3.0,  # default 8.0
                            iterationsCount=20,
                        )
                    else:  # iter PnP
                        # points_3d = np.expand_dims(points_3d, 0)  # uncomment for EPNP
                        # points_2d = np.expand_dims(points_2d, 0)
                        _, rvec, t_est = cv2.solvePnP(
                            objectPoints=points_3d,
                            imagePoints=points_2d,
                            cameraMatrix=camera_matrix,
                            distCoeffs=dist_coeffs,
                            flags=cv2.SOLVEPNP_ITERATIVE,
                            # flags=cv2.SOLVEPNP_EPNP,
                            useExtrinsicGuess=True,
                            rvec=rvec0,
                            tvec=trans_est_net,
                        )
                    rot_est, _ = cv2.Rodrigues(rvec)
                    if pnp_type not in ["ransac_rot"]:
                        diff_t_est = te(t_est, trans_est_net)
                        if diff_t_est > 1:  # diff too large
                            logger.warning("translation error too large: {}".format(diff_t_est))
                            t_est = trans_est_net
                    else:
                        t_est = trans_est_net
                    pose_est = np.concatenate([rot_est, t_est.reshape((3, 1))], axis=-1)
                else:
                    logger.warning("num points: {}".format(len(img_points)))
                    pose_est_net = np.hstack([rot_est_net, trans_est_net.reshape(3, 1)])
                    pose_est = pose_est_net

                json_results.extend(
                    self.pose_prediction_to_json(
                        pose_est, scene_id, im_id, obj_id=obj_id, score=score, pose_time=output["time"], K=K
                    )
                )

            output["time"] += time.perf_counter() - start_process_time

            # process time for this image
            for item in json_results:
                item["time"] = output["time"]
            self._predictions.extend(json_results)

    def process_pnp_ransac(self, inputs, outputs, out_dict):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id", "scene_id".
            outputs:
        """
        cfg = self.cfg
        net_cfg = cfg.MODEL.POSE_NET
        out_coor_x = out_dict["coor_x"].detach()
        out_coor_y = out_dict["coor_y"].detach()
        out_coor_z = out_dict["coor_z"].detach()
        out_xyz = get_out_coor(cfg, out_coor_x, out_coor_y, out_coor_z)
        out_xyz = out_xyz.to(self._cpu_device).numpy()

        out_mask = get_out_mask(cfg, out_dict["mask"].detach())
        out_mask = out_mask.to(self._cpu_device).numpy()

        out_i = -1
        for i, (_input, output) in enumerate(zip(inputs, outputs)):
            start_process_time = time.perf_counter()
            json_results = []
            for inst_i in range(len(_input["roi_img"])):
                out_i += 1

                coord_2d_i = _input["roi_coord_2d"][inst_i].cpu().numpy().transpose(1, 2, 0)  # CHW->HWC
                im_H = _input["im_H"][inst_i].item()
                im_W = _input["im_W"][inst_i].item()

                K = _input["cam"][inst_i].cpu().numpy().copy()

                roi_label = _input["roi_cls"][inst_i]  # 0-based label
                score = _input["score"][inst_i]
                roi_label, cls_name = self._maybe_adapt_label_cls_name(roi_label)
                if cls_name is None:
                    continue

                scene_im_id_split = _input["scene_im_id"][inst_i].split("/")
                scene_id = scene_im_id_split[0]
                im_id = int(scene_im_id_split[1])
                obj_id = self.data_ref.obj2id[cls_name]

                # get pose
                xyz_i = out_xyz[out_i].transpose(1, 2, 0)
                mask_i = np.squeeze(out_mask[out_i])

                (img_points, model_points,) = self.get_img_model_points_with_coords2d(
                    mask_i,
                    xyz_i,
                    coord_2d_i,
                    im_H=im_H,
                    im_W=im_W,
                    extent=_input["roi_extent"][inst_i].cpu().numpy(),
                    mask_thr=net_cfg.GEO_HEAD.MASK_THR_TEST,
                )

                pnp_method = cv2.SOLVEPNP_EPNP
                # pnp_method = cv2.SOLVEPNP_ITERATIVE
                num_points = len(img_points)
                if num_points >= 4:
                    pose_est = misc.pnp_v2(
                        model_points,
                        img_points,
                        K,
                        method=pnp_method,
                        ransac=True,
                        ransac_reprojErr=3,
                        ransac_iter=100,
                        # ransac_reprojErr=1,  # more accurate but ~10ms slower
                        # ransac_iter=150,
                    )
                else:
                    logger.warning("num points: {}".format(len(img_points)))
                    pose_est = -100 * np.ones((3, 4), dtype=np.float32)

                json_results.extend(
                    self.pose_prediction_to_json(
                        pose_est, scene_id, im_id, obj_id=obj_id, score=score, pose_time=output["time"], K=K
                    )
                )
            output["time"] += time.perf_counter() - start_process_time

            # process time for this image
            for item in json_results:
                item["time"] = output["time"]
            self._predictions.extend(json_results)

    def process_depth_refine(self, inputs, outputs, out_dict):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id", "scene_id".
            outputs:
        """
        cfg = self.cfg
        net_cfg = cfg.MODEL.POSE_NET
        out_coor_x = out_dict["coor_x"].detach()
        out_coor_y = out_dict["coor_y"].detach()
        out_coor_z = out_dict["coor_z"].detach()
        out_xyz = get_out_coor(cfg, out_coor_x, out_coor_y, out_coor_z)
        out_xyz = out_xyz.to(self._cpu_device) #.numpy()

        out_mask = get_out_mask(cfg, out_dict["mask"].detach())
        out_mask = out_mask.to(self._cpu_device) #.numpy()
        out_rots = out_dict["rot"].detach().to(self._cpu_device).numpy()
        out_transes = out_dict["trans"].detach().to(self._cpu_device).numpy()

        zoom_K = batch_data_inference_roi(cfg, inputs)['roi_zoom_K']

        out_i = -1
        for i, (_input, output) in enumerate(zip(inputs, outputs)):
            start_process_time = time.perf_counter()
            json_results = []
            for inst_i in range(len(_input["roi_img"])):
                out_i += 1

                K = _input["cam"][inst_i].cpu().numpy().copy()
                # print('K', K)

                K_crop = zoom_K[inst_i].cpu().numpy().copy()
                # print('K_crop', K_crop)

                roi_label = _input["roi_cls"][inst_i]  # 0-based label
                score = _input["score"][inst_i]
                roi_label, cls_name = self._maybe_adapt_label_cls_name(roi_label)
                if cls_name is None:
                    continue

                scene_im_id_split = _input["scene_im_id"][inst_i].split("/")
                scene_id = scene_im_id_split[0]
                im_id = int(scene_im_id_split[1])
                obj_id = self.data_ref.obj2id[cls_name]

                # get pose
                xyz_i = out_xyz[out_i].permute(1, 2, 0)
                mask_i = np.squeeze(out_mask[out_i])

                rot_est = out_rots[out_i]
                trans_est = out_transes[out_i]
                pose_est = np.hstack([rot_est, trans_est.reshape(3, 1)])
                depth_sensor_crop = _input['roi_depth'][inst_i].cpu().numpy().copy().squeeze()
                depth_sensor_mask_crop = depth_sensor_crop > 0

                net_cfg = cfg.MODEL.POSE_NET
                crop_res = net_cfg.OUTPUT_RES



                for _ in range(cfg.TEST.DEPTH_REFINE_ITER):
                    self.ren.clear()
                    self.ren.set_cam(K_crop)
                    self.ren.draw_model(self.ren_models[self.data_ref.objects.index(cls_name)], pose_est)
                    ren_im, ren_dp = self.ren.finish()
                    ren_mask = ren_dp > 0

                    if self.cfg.TEST.USE_COOR_Z_REFINE:
                        coor_np = xyz_i.numpy()
                        coor_np_t = coor_np.reshape(-1, 3)
                        coor_np_t = coor_np_t.T
                        coor_np_r = rot_est @ coor_np_t
                        coor_np_r = coor_np_r.T
                        coor_np_r = coor_np_r.reshape(crop_res, crop_res, 3)
                        query_img_norm = coor_np_r[:, :, -1] * mask_i.numpy()
                        query_img_norm = query_img_norm * ren_mask * depth_sensor_mask_crop
                    else:
                        query_img = xyz_i

                        query_img_norm = torch.norm(query_img, dim=-1) * mask_i
                        query_img_norm = query_img_norm.numpy() * ren_mask * depth_sensor_mask_crop
                    norm_sum = query_img_norm.sum()
                    if norm_sum == 0:
                        continue
                    query_img_norm /= norm_sum
                    norm_mask = query_img_norm > (query_img_norm.max() * self.depth_refine_threshold)
                    yy, xx = np.argwhere(norm_mask).T  # 2 x (N,)
                    depth_diff = depth_sensor_crop[yy, xx] - ren_dp[yy, xx]
                    depth_adjustment = np.median(depth_diff)



                    yx_coords = np.meshgrid(np.arange(crop_res), np.arange(crop_res))
                    yx_coords = np.stack(yx_coords[::-1], axis=-1)  # (crop_res, crop_res, 2yx)
                    yx_ray_2d = (yx_coords * query_img_norm[..., None]).sum(axis=(0, 1))  # y, x
                    ray_3d = np.linalg.inv(K_crop) @ (*yx_ray_2d[::-1], 1)
                    ray_3d /= ray_3d[2]

                    trans_delta = ray_3d[:, None] * depth_adjustment
                    trans_est = trans_est + trans_delta.reshape(3)
                    pose_est = np.hstack([rot_est, trans_est.reshape(3, 1)])

                json_results.extend(
                    self.pose_prediction_to_json(
                        pose_est, scene_id, im_id, obj_id=obj_id, score=score, pose_time=output["time"], K=K
                    )
                )
            output["time"] += time.perf_counter() - start_process_time

            # process time for this image
            for item in json_results:
                item["time"] = output["time"]
            self._predictions.extend(json_results)

    def evaluate(self):
        # bop toolkit eval in subprocess, no return value
        if self._distributed:
            synchronize()
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

        return self._eval_predictions()
        # return copy.deepcopy(self._eval_predictions())

    def _eval_predictions(self):
        """Evaluate self._predictions on 6d pose.

        Return results with the metrics of the tasks.
        """
        logger.info("Eval results with BOP toolkit ...")
        self._process_time_of_preds(self._predictions)
        results_all = {"iter0": self._predictions}
        save_and_eval_results(self.cfg, results_all, self._output_dir, obj_ids=self.obj_ids)
        return {}

    def _process_time_of_preds(self, results):
        # process time so that each image's inference time is the largest of all different inference times
        # in-place modification, no return
        times = {}
        for item in results:
            im_key = "{}/{}".format(item["scene_id"], item["im_id"])
            if im_key not in times:
                times[im_key] = []
            times[im_key].append(item["time"])

        for item in results:
            im_key = "{}/{}".format(item["scene_id"], item["im_id"])
            item["time"] = float(np.max(times[im_key]))

    def pose_from_upnp(self, mean_pts2d, covar, points_3d, K):
        import scipy
        from core.csrc.uncertainty_pnp.un_pnp_utils import uncertainty_pnp

        cov_invs = []
        for vi in range(covar.shape[0]):
            if covar[vi, 0, 0] < 1e-6 or np.sum(np.isnan(covar)[vi]) > 0:
                cov_invs.append(np.zeros([2, 2]).astype(np.float32))
                continue

            cov_inv = np.linalg.inv(scipy.linalg.sqrtm(covar[vi]))
            cov_invs.append(cov_inv)
        cov_invs = np.asarray(cov_invs)  # pn,2,2
        weights = cov_invs.reshape([-1, 4])
        weights = weights[:, (0, 1, 3)]
        pose_pred = uncertainty_pnp(mean_pts2d, weights, points_3d, K)
        return pose_pred

    def pose_from_upnp_v2(self, mean_pts2d, covar, points_3d, K):
        from core.csrc.uncertainty_pnp.un_pnp_utils import uncertainty_pnp_v2

        pose_pred = uncertainty_pnp_v2(mean_pts2d, covar, points_3d, K)
        return pose_pred

    def pose_prediction_to_json(self, pose_est, scene_id, im_id, obj_id, score=None, pose_time=-1, K=None):
        """
        Args:
            pose_est:
            scene_id (str): the scene id
            img_id (str): the image id
            label: used to get obj_id
            score: confidence
            pose_time:

        Returns:
            list[dict]: the results in BOP evaluation format
        """
        results = []
        if score is None:  # TODO: add score key in test bbox json file
            score = 1.0
        rot = pose_est[:3, :3]
        trans = pose_est[:3, 3]
        # for standard bop datasets, scene_id and im_id can be obtained from file_name
        result = {
            "scene_id": scene_id,  # if not available, assume 0
            "im_id": im_id,
            "obj_id": obj_id,  # the obj_id in bop datasets
            "score": score,
            "R": to_list(rot),
            "t": to_list(1000 * trans),  # mm
            "time": pose_time,
        }
        results.append(result)
        return results


def gdrn_inference_on_dataset(cfg, model, data_loader, evaluator, amp_test=False):
    """Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately. The model
    will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    total_process_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
                total_process_time = 0

            start_compute_time = time.perf_counter()
            #############################
            # process input
            if not isinstance(inputs, list):  # bs=1
                inputs = [inputs]
            batch = batch_data(cfg, inputs, phase="test")
            if evaluator.train_objs is not None:
                roi_labels = batch["roi_cls"].cpu().numpy().tolist()
                obj_names = [evaluator.obj_names[_l] for _l in roi_labels]
                if all(_obj not in evaluator.train_objs for _obj in obj_names):
                    continue

            # if cfg.DEBUG:
            #     for i in range(len(batch["roi_cls"])):
            #         vis_roi_im = batch["roi_img"][i].cpu().numpy().transpose(1,2,0)[:, :, ::-1]
            #         show_ims = [vis_roi_im]
            #         show_titles = ["roi_im"]
            #
            #         vis_coor2d = batch["roi_coord_2d"][i].cpu().numpy()
            #         show_ims.extend([vis_coor2d[0], vis_coor2d[1]])
            #         show_titles.extend(["coord_2d_x", "coord_2d_y"])
            #         grid_show(show_ims, show_titles, row=1, col=3)

            if cfg.INPUT.WITH_DEPTH and "depth" in cfg.MODEL.POSE_NET.NAME.lower():
                inp = torch.cat([batch["roi_img"], batch["roi_depth"]], dim=1)
            else:
                inp = batch["roi_img"]

            with autocast(enabled=amp_test):  # gdrn amp_test seems slower
                out_dict = model(
                    inp,
                    roi_classes=batch["roi_cls"],
                    roi_cams=batch["roi_cam"],
                    roi_whs=batch["roi_wh"],
                    roi_centers=batch["roi_center"],
                    resize_ratios=batch["resize_ratio"],
                    roi_coord_2d=batch.get("roi_coord_2d", None),
                    roi_coord_2d_rel=batch.get("roi_coord_2d_rel", None),
                    roi_extents=batch.get("roi_extent", None),
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            cur_compute_time = time.perf_counter() - start_compute_time
            total_compute_time += cur_compute_time
            # NOTE: added
            outputs = [{} for _ in range(len(inputs))]
            for _i in range(len(outputs)):
                # outputs[_i]["time"] = cur_compute_time + float(inputs[_i].get("time", 0))
                det_time = 0
                if "time" in inputs[_i]:
                    det_time = inputs[_i]["time"][0]  # list
                outputs[_i]["time"] = cur_compute_time + det_time

            start_process_time = time.perf_counter()
            evaluator.process(inputs, outputs, out_dict)  # RANSAC/PnP
            cur_process_time = time.perf_counter() - start_process_time
            total_process_time += cur_process_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    f"Inference done {idx+1}/{total}. {seconds_per_img:.4f} s / img. ETA={str(eta)}",
                    n=5,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        f"Total inference time: {total_time_str} "
        f"({total_time / (total - num_warmup):.6f} s / img per device, on {num_devices} devices)"
    )
    # pure forward time
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str,
            total_compute_time / (total - num_warmup),
            num_devices,
        )
    )
    # post_process time
    total_process_time_str = str(datetime.timedelta(seconds=int(total_process_time)))
    logger.info(
        "Total inference post process time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_process_time_str,
            total_process_time / (total - num_warmup),
            num_devices,
        )
    )

    results = evaluator.evaluate()  # results is always None
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


def gdrn_save_result_of_dataset(cfg, model, data_loader, output_dir, dataset_name, train_objs=None, amp_test=False):
    """
    Run model (in eval mode) on the data_loader and save predictions
    Args:
        cfg: config
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    net_cfg = cfg.MODEL.POSE_NET

    # NOTE: dataset name should be the same as TRAIN to get the correct meta
    _metadata = MetadataCatalog.get(dataset_name)
    data_ref = ref.__dict__[_metadata.ref_key]
    obj_names = _metadata.objs
    obj_ids = [data_ref.obj2id[obj_name] for obj_name in obj_names]

    if cfg.TEST.get("COLOR_AUG", False):
        result_name = "results_color_aug.pkl"
    else:
        result_name = "results.pkl"
    mmcv.mkdir_or_exist(output_dir)  # NOTE: should be the same as the evaluation output dir
    result_path = osp.join(output_dir, result_name)
    if osp.exists(result_path):
        logger.warning("{} exists, overriding!".format(result_path))

    total = len(data_loader)  # inference data loader must have a fixed length
    result_dict = {}
    VIS = cfg.TEST.VIS  # NOTE: change this for debug/vis
    if VIS:
        import cv2
        from lib.vis_utils.image import vis_image_mask_bbox_cv2, vis_image_bboxes_cv2, grid_show
        from core.utils.my_visualizer import MyVisualizer, _GREY, _GREEN, _BLUE, _RED
        from core.utils.data_utils import crop_resize_by_warp_affine

        obj_models = {
            data_ref.obj2id[_obj_name]: inout.load_ply(m_path, vertex_scale=data_ref.vertex_scale)
            for _obj_name, m_path in zip(data_ref.objects, data_ref.model_paths)
        }
        # key is [str(obj_id)]["bbox3d_and_center"]
        kpts3d_dict = data_ref.get_keypoints_3d()
        dset_dicts = DatasetCatalog.get(dataset_name)
        scene_im_id_to_gt_index = {d["scene_im_id"]: i for i, d in enumerate(dset_dicts)}

    test_bbox_type = cfg.TEST.TEST_BBOX_TYPE
    if test_bbox_type == "gt":
        bbox_key = "bbox"
    else:
        bbox_key = f"bbox_{test_bbox_type}"

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            # process input ----------------------------------------------------------
            if not isinstance(inputs, list):
                inputs = [inputs]
            batch = batch_data(cfg, inputs, phase="test")
            if train_objs is not None:
                roi_labels = batch["roi_cls"].cpu().numpy().tolist()
                cur_obj_names = [obj_names[_l] for _l in roi_labels]  # obj names in this test batch
                if all(_obj not in train_objs for _obj in cur_obj_names):
                    continue
            # NOTE: do model inference -----------------------------
            if cfg.INPUT.WITH_DEPTH:
                inp = torch.cat([batch["roi_img"], batch["roi_depth"]], dim=1)
            else:
                inp = batch["roi_img"]
            with autocast(enabled=amp_test):  # gdrn amp_test seems slower
                out_dict = model(
                    inp,
                    roi_classes=batch["roi_cls"],
                    roi_cams=batch["roi_cam"],
                    roi_whs=batch["roi_wh"],
                    roi_centers=batch["roi_center"],
                    resize_ratios=batch["resize_ratio"],
                    roi_coord_2d=batch.get("roi_coord_2d", None),
                    roi_coord_2d_rel=batch.get("roi_coord_2d_rel", None),
                    roi_extents=batch.get("roi_extent", None),
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            cur_compute_time = time.perf_counter() - start_compute_time
            total_compute_time += cur_compute_time

            # convert raw mask (out size) to mask in image ---------------------------------------------------------------------
            raw_masks = out_dict["mask"]
            mask_probs = get_out_mask(cfg, raw_masks)

            # for crop and resize
            bs = batch["roi_cls"].shape[0]
            tensor_kwargs = {"dtype": torch.float32, "device": "cuda"}
            rois_xy0 = batch["roi_center"] - batch["scale"].view(bs, -1) / 2  # bx2
            rois_xy1 = batch["roi_center"] + batch["scale"].view(bs, -1) / 2  # bx2
            batch["inst_rois"] = torch.cat([torch.arange(bs, **tensor_kwargs).view(-1, 1), rois_xy0, rois_xy1], dim=1)

            im_H = int(batch["im_H"][0])
            im_W = int(batch["im_W"][0])
            # BHW
            masks_in_im = paste_masks_in_image(
                mask_probs[:, 0, :, :],
                batch["inst_rois"][:, 1:5],
                image_shape=(im_H, im_W),
                threshold=net_cfg.GEO_HEAD.MASK_THR_TEST,
            )
            masks_np = masks_in_im.detach().to(torch.uint8).cpu().numpy()
            masks_rle = [binary_mask_to_rle(_m, compressed=True) for _m in masks_np]

            if "full_mask" in out_dict:
                raw_full_masks = out_dict["full_mask"]
                full_mask_probs = get_out_mask(cfg, raw_full_masks)
                full_masks_in_im = paste_masks_in_image(
                    full_mask_probs[:, 0, :, :],
                    batch["inst_rois"][:, 1:5],
                    image_shape=(im_H, im_W),
                    threshold=net_cfg.GEO_HEAD.MASK_THR_TEST,
                )
                full_masks_np = full_masks_in_im.detach().to(torch.uint8).cpu().numpy()
                full_masks_rle = [binary_mask_to_rle(_m, compressed=True) for _m in full_masks_np]

            # NOTE: process results ----------------------------------------------------------------------
            i_out = -1
            for i_in, _input in enumerate(inputs):
                for i_inst in range(len(_input["roi_img"])):
                    i_out += 1

                    scene_im_id = _input["scene_im_id"][i_inst]
                    cur_obj_id = obj_ids[int(batch["roi_cls"][i_out])]
                    cur_res = {
                        "obj_id": cur_obj_id,
                        "score": float(_input["score"][i_inst]),
                        bbox_key: _input[bbox_key][i_inst].detach().cpu().numpy(),  # xyxy
                        "mask": masks_rle[i_out],  # save mask as rle
                    }
                    if cfg.TEST.USE_PNP:
                        pose_est_pnp = get_pnp_ransac_pose(cfg, _input, out_dict, i_inst, i_out)
                        cur_res["R"] = pose_est_pnp[:3, :3]
                        cur_res["t"] = pose_est_pnp[:3, 3]
                    else:
                        cur_res.update(
                            {
                                "R": out_dict["rot"][i_out].detach().cpu().numpy(),
                                "t": out_dict["trans"][i_out].detach().cpu().numpy(),
                            }
                        )
                    if "full_mask" in out_dict:
                        cur_res["full_mask"] = full_masks_rle[i_out]

                    if scene_im_id not in result_dict:
                        result_dict[scene_im_id] = []
                    result_dict[scene_im_id].append(cur_res)  # each image's results saved as a list

                    if VIS:  # vis -----------------------------------------------------------
                        vis_dict = {}
                        image_path = _input["file_name"][i_inst]
                        image = mmcv.imread(image_path, "color")
                        img_vis = vis_image_mask_bbox_cv2(
                            image,
                            [masks_np[i_out]],
                            [_input[bbox_key][i_inst].detach().cpu().numpy()],
                            labels=[data_ref.id2obj[cur_obj_id]],
                        )
                        vis_dict[f"im_{bbox_key}_mask_vis"] = img_vis[:, :, ::-1]
                        if "full_mask" in out_dict:
                            img_vis_full_mask = vis_image_mask_bbox_cv2(
                                image,
                                [full_masks_np[i_out]],
                                [_input[bbox_key][i_inst].detach().cpu().numpy()],
                                labels=[data_ref.id2obj[cur_obj_id]],
                            )
                            vis_dict[f"im_{bbox_key}_mask_full"] = img_vis_full_mask[:, :, ::-1]

                        K = _input["cam"][i_inst].detach().cpu().numpy()
                        kpt3d = kpts3d_dict[str(cur_obj_id)]["bbox3d_and_center"]
                        # gt pose
                        gt_idx = scene_im_id_to_gt_index[scene_im_id]
                        gt_dict = dset_dicts[gt_idx]
                        has_gt = False
                        if "annotations" in gt_dict:
                            has_gt = True
                            gt_annos = gt_dict["annotations"]
                            # find the gt anno ---------------
                            found_gt = False
                            for gt_anno in gt_annos:
                                gt_label = gt_anno["category_id"]
                                gt_obj = obj_names[gt_label]
                                gt_obj_id = data_ref.obj2id[gt_obj]
                                if cur_obj_id == gt_obj_id:
                                    found_gt = True
                                    gt_pose = gt_anno["pose"]
                                    break
                            if not found_gt:
                                kpt2d_gt = None
                            else:
                                kpt2d_gt = misc.project_pts(kpt3d, K, gt_pose[:3, :3], gt_pose[:3, 3])

                        pose_est = np.hstack([cur_res["R"], cur_res["t"].reshape(3, 1)])
                        kpt2d_est = misc.project_pts(kpt3d, K, pose_est[:3, :3], pose_est[:3, 3])

                        proj_pts_est = misc.project_pts(obj_models[cur_obj_id]["pts"], K, cur_res["R"], cur_res["t"])
                        mask_pose_est = misc.points2d_to_mask(proj_pts_est, im_H, im_W)

                        image_mask_pose_est = vis_image_mask_cv2(image, mask_pose_est, color="yellow")
                        image_mask_pose_est = vis_image_bboxes_cv2(
                            image_mask_pose_est,
                            [_input[bbox_key][i_inst].detach().cpu().numpy()],
                            labels=[data_ref.id2obj[cur_obj_id]],
                        )
                        vis_dict[f"im_{bbox_key}_mask_pose_est"] = image_mask_pose_est[:, :, ::-1]

                        maxx, maxy, minx, miny = 0, 0, 1000, 1000
                        for i in range(len(kpt2d_est)):
                            maxx, maxy, minx, miny = (
                                max(maxx, kpt2d_est[i][0]),
                                max(maxy, kpt2d_est[i][1]),
                                min(minx, kpt2d_est[i][0]),
                                min(miny, kpt2d_est[i][1]),
                            )
                            if has_gt and kpt2d_gt is not None:
                                maxx, maxy, minx, miny = (
                                    max(maxx, kpt2d_gt[i][0]),
                                    max(maxy, kpt2d_gt[i][1]),
                                    min(minx, kpt2d_gt[i][0]),
                                    min(miny, kpt2d_gt[i][1]),
                                )
                        center_ = np.array([(minx + maxx) / 2, (miny + maxy) / 2])
                        scale_ = max(maxx - minx, maxy - miny) * 1.5  # * 3  # + 10
                        CROP_SIZE = 256
                        im_zoom = crop_resize_by_warp_affine(image_mask_pose_est, center_, scale_, CROP_SIZE)

                        zoom_kpt2d_est = kpt2d_est.copy()
                        for i in range(len(kpt2d_est)):
                            zoom_kpt2d_est[i][0] = (kpt2d_est[i][0] - (center_[0] - scale_ / 2)) * CROP_SIZE / scale_
                            zoom_kpt2d_est[i][1] = (kpt2d_est[i][1] - (center_[1] - scale_ / 2)) * CROP_SIZE / scale_

                        if has_gt and kpt2d_gt is not None:
                            zoom_kpt2d_gt = kpt2d_gt.copy()
                            for i in range(len(kpt2d_gt)):
                                zoom_kpt2d_gt[i][0] = (kpt2d_gt[i][0] - (center_[0] - scale_ / 2)) * CROP_SIZE / scale_
                                zoom_kpt2d_gt[i][1] = (kpt2d_gt[i][1] - (center_[1] - scale_ / 2)) * CROP_SIZE / scale_
                        visualizer = MyVisualizer(im_zoom[:, :, ::-1], _metadata)
                        linewidth = 3
                        if has_gt and kpt2d_gt is not None:
                            visualizer.draw_bbox3d_and_center(
                                zoom_kpt2d_gt,
                                top_color=_BLUE,
                                bottom_color=_GREY,
                                linewidth=linewidth,
                                draw_center=True,
                            )
                        visualizer.draw_bbox3d_and_center(
                            zoom_kpt2d_est, top_color=_RED, bottom_color=_GREY, linewidth=linewidth, draw_center=True
                        )

                        im_gt_pred = visualizer.get_output().get_image()
                        vis_dict["zoom_im_gt_pred"] = im_gt_pred

                        show_titles = [_k for _k, _v in vis_dict.items()]
                        show_ims = [_v for _k, _v in vis_dict.items()]
                        ncol = 2
                        nrow = int(np.ceil(len(show_ims) / ncol))
                        grid_show(show_ims, show_titles, row=nrow, col=ncol)
                    # end vis ----------------------------------------------------------------------

            # -----------------------------------------------------------------------------------------
            if (idx + 1) % logging_interval == 0:
                duration = time.perf_counter() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(seconds=int(seconds_per_img * (total - num_warmup) - duration))
                logger.info(
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(idx + 1, total, seconds_per_img, str(eta))
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.perf_counter() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    mmcv.dump(result_dict, result_path)
    logger.info("Results saved to {}".format(result_path))


def get_pnp_ransac_pose(cfg, _input, out_dict, inst_i, out_i):
    """
    Args:
        _input: the instance input to a model.
        out_dict: the predictions
    """

    net_cfg = cfg.MODEL.POSE_NET
    out_coor_x = out_dict["coor_x"].detach()
    out_coor_y = out_dict["coor_y"].detach()
    out_coor_z = out_dict["coor_z"].detach()
    out_xyz = get_out_coor(cfg, out_coor_x, out_coor_y, out_coor_z)
    out_xyz = out_xyz.cpu().numpy()

    out_mask = get_out_mask(cfg, out_dict["mask"].detach())
    out_mask = out_mask.cpu().numpy()

    bbox_center_i = _input["bbox_center"][inst_i]
    cx_i, cy_i = bbox_center_i

    coord_2d_i = _input["roi_coord_2d"][inst_i].cpu().numpy().transpose(1, 2, 0)  # CHW->HWC
    im_H = _input["im_H"][inst_i].item()
    im_W = _input["im_W"][inst_i].item()

    K = _input["cam"][inst_i].cpu().numpy().copy()

    # get pose
    xyz_i = out_xyz[out_i].transpose(1, 2, 0)
    mask_i = np.squeeze(out_mask[out_i])

    (img_points, model_points,) = get_img_model_points_with_coords2d(
        mask_i,
        xyz_i,
        coord_2d_i,
        im_H=im_H,
        im_W=im_W,
        extent=_input["roi_extent"][inst_i].cpu().numpy(),
        mask_thr=net_cfg.GEO_HEAD.MASK_THR_TEST,
    )

    pnp_method = cv2.SOLVEPNP_EPNP
    # pnp_method = cv2.SOLVEPNP_ITERATIVE
    num_points = len(img_points)
    if num_points >= 4:
        pose_est = misc.pnp_v2(
            model_points,
            img_points,
            K,
            method=pnp_method,
            ransac=True,
            ransac_reprojErr=3,
            ransac_iter=100,
            # ransac_reprojErr=1,  # more accurate but ~10ms slower
            # ransac_iter=150,
        )
    else:
        logger.warning("num points: {}".format(len(img_points)))
        pose_est = -100 * np.ones((3, 4), dtype=np.float32)
    return pose_est


def get_img_model_points_with_coords2d(
    mask_pred_crop, xyz_pred_crop, coord2d_crop, im_H, im_W, extent, max_num_points=-1, mask_thr=0.5
):
    """
    from predicted crop_and_resized xyz, bbox top-left,
    get 2D-3D correspondences (image points, 3D model points)
    Args:
        mask_pred_crop: HW, predicted mask in roi_size
        xyz_pred_crop: HWC, predicted xyz in roi_size(eg. 64)
        coord2d_crop: HW2 coords 2d in roi size
        im_H, im_W
        extent: size of x,y,z
    """
    # [0, 1] --> [-0.5, 0.5] --> original
    xyz_pred_crop[:, :, 0] = (xyz_pred_crop[:, :, 0] - 0.5) * extent[0]
    xyz_pred_crop[:, :, 1] = (xyz_pred_crop[:, :, 1] - 0.5) * extent[1]
    xyz_pred_crop[:, :, 2] = (xyz_pred_crop[:, :, 2] - 0.5) * extent[2]

    coord2d_crop = coord2d_crop.copy()
    coord2d_crop[:, :, 0] = coord2d_crop[:, :, 0] * im_W
    coord2d_crop[:, :, 1] = coord2d_crop[:, :, 1] * im_H

    sel_mask = (
        (mask_pred_crop > mask_thr)
        & (abs(xyz_pred_crop[:, :, 0]) > 0.0001 * extent[0])
        & (abs(xyz_pred_crop[:, :, 1]) > 0.0001 * extent[1])
        & (abs(xyz_pred_crop[:, :, 2]) > 0.0001 * extent[2])
    )
    model_points = xyz_pred_crop[sel_mask].reshape(-1, 3)
    image_points = coord2d_crop[sel_mask].reshape(-1, 2)

    if max_num_points >= 4:
        num_points = len(image_points)
        max_keep = min(max_num_points, num_points)
        indices = [i for i in range(num_points)]
        random.shuffle(indices)
        model_points = model_points[indices[:max_keep]]
        image_points = image_points[indices[:max_keep]]
    return image_points, model_points
