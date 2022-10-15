import mmcv
import os.path as osp
import numpy as np
import sys
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../../"))

from lib.vis_utils.colormap import colormap
from core.utils.my_visualizer import MyVisualizer, _GREY, _GREEN, _BLUE
from lib.utils.mask_utils import cocosegm2mask
from core.utils.data_utils import crop_resize_by_warp_affine, read_image_mmcv
from core.gdrn_modeling.datasets.dataset_factory import register_datasets
from lib.pysixd import misc
from transforms3d.quaternions import quat2mat


out_size = 256
colors = colormap(rgb=False, maximum=255)


id2obj = {
    1: "ape",
    2: "benchvise",
    3: "bowl",
    4: "camera",
    5: "can",
    6: "cat",
    7: "cup",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}

# NOTE:
pred_path = (
    "output/gdrn/lm/a6_cPnP_lm13/inference_model_final_wo_optim/lm_13_test/a6-cPnP-lm13-test_lm_13_test_preds.pkl"
)
vis_dir = "output/gdrn/lm/a6_cPnP_lm13/inference_model_final_wo_optim/lm_13_test/vis_gt_pred"
mmcv.mkdir_or_exist(vis_dir)

print(pred_path)
preds = mmcv.load(pred_path)

dataset_name = "lm_13_test"
print(dataset_name)
register_datasets([dataset_name])

meta = MetadataCatalog.get(dataset_name)
print("MetadataCatalog: ", meta)
objs = meta.objs

dset_dicts = DatasetCatalog.get(dataset_name)
for d in tqdm(dset_dicts):
    K = d["cam"]
    file_name = d["file_name"]
    img = read_image_mmcv(file_name, format="BGR")

    scene_im_id_split = d["scene_im_id"].split("/")
    scene_id = scene_im_id_split[0]
    im_id = int(scene_im_id_split[1])

    imH, imW = img.shape[:2]
    annos = d["annotations"]
    masks = [cocosegm2mask(anno["segmentation"], imH, imW) for anno in annos]
    bboxes = [anno["bbox"] for anno in annos]
    bbox_modes = [anno["bbox_mode"] for anno in annos]
    bboxes_xyxy = np.array(
        [BoxMode.convert(box, box_mode, BoxMode.XYXY_ABS) for box, box_mode in zip(bboxes, bbox_modes)]
    )
    kpts_3d_list = [anno["bbox3d_and_center"] for anno in annos]
    quats = [anno["quat"] for anno in annos]
    transes = [anno["trans"] for anno in annos]
    Rs = [quat2mat(quat) for quat in quats]
    # 0-based label
    cat_ids = [anno["category_id"] for anno in annos]

    kpts_2d = [misc.project_pts(kpt3d, K, R, t) for kpt3d, R, t in zip(kpts_3d_list, Rs, transes)]

    obj_names = [objs[cat_id] for cat_id in cat_ids]

    kpts_2d_est = []
    est_Rs = []
    est_ts = []
    maxx, maxy, minx, miny = 0, 0, 1000, 1000
    for anno_i, anno in enumerate(annos):
        kpt_2d_gt = kpts_2d[anno_i]
        obj_name = obj_names[anno_i]
        R_est = preds[obj_name][file_name]["R"]
        t_est = preds[obj_name][file_name]["t"]
        est_Rs.append(R_est)
        est_ts.append(t_est)
        kpt_2d_est = misc.project_pts(kpts_3d_list[anno_i], K, R_est, t_est)
        kpts_2d_est.append(kpt_2d_est)

        for i in range(len(kpt_2d_est)):
            maxx, maxy, minx, miny = (
                max(maxx, kpt_2d_est[i][0]),
                max(maxy, kpt_2d_est[i][1]),
                min(minx, kpt_2d_est[i][0]),
                min(miny, kpt_2d_est[i][1]),
            )
            maxx, maxy, minx, miny = (
                max(maxx, kpt_2d_gt[i][0]),
                max(maxy, kpt_2d_gt[i][1]),
                min(minx, kpt_2d_gt[i][0]),
                min(miny, kpt_2d_gt[i][1]),
            )
    center = np.array([(minx + maxx) / 2, (miny + maxy) / 2])
    scale = max(maxx - minx, maxy - miny) * 3  # + 10
    crop_minx = max(0, center[0] - scale / 2)
    crop_miny = max(0, center[1] - scale / 2)
    crop_maxx = min(imW - 1, center[0] + scale / 2)
    crop_maxy = min(imH - 1, center[1] + scale / 2)
    scale = min(scale, min(crop_maxx - crop_minx, crop_maxy - crop_miny))

    zoomed_im = crop_resize_by_warp_affine(img, center, scale, out_size)
    K_zoom = K.copy()
    K_zoom[0, 2] -= center[0] - scale / 2
    K_zoom[1, 2] -= center[1] - scale / 2
    K_zoom[0, :] *= out_size / scale
    K_zoom[1, :] *= out_size / scale
    kpts_2d_gt_zoom = [misc.project_pts(kpt3d, K_zoom, R, t) for kpt3d, R, t in zip(kpts_3d_list, Rs, transes)]
    kpts_2d_est_zoom = [misc.project_pts(kpt3d, K_zoom, R, t) for kpt3d, R, t in zip(kpts_3d_list, est_Rs, est_ts)]

    # mmcv.imwrite(zoomed_im, save_path)
    # yapf: disable
    # kpts_2d_est_zoom = [np.array(
    #     [
    #         [(x - (center[0] - scale / 2)) * out_size / scale,
    #          (y - (center[1] - scale / 2)) * out_size / scale]
    #         for [x, y] in kpt_2d
    #     ]
    # ) for kpt_2d in kpts_2d_est]
    #
    # kpts_2d_gt_zoom = [np.array(
    #     [
    #         [(x - (center[0] - scale / 2)) * out_size / scale,
    #          (y - (center[1] - scale / 2)) * out_size / scale]
    #         for [x, y] in kpt_2d
    #     ]
    # ) for kpt_2d in kpts_2d]
    # yapf: enable

    linewidth = 3
    visualizer = MyVisualizer(zoomed_im[:, :, ::-1], meta)
    for kpt_2d_gt_zoom, kpt_2d_est_zoom in zip(kpts_2d_gt_zoom, kpts_2d_est_zoom):
        visualizer.draw_bbox3d_and_center(
            kpt_2d_gt_zoom,
            top_color=_BLUE,
            bottom_color=_GREY,
            linewidth=linewidth,
            draw_center=True,
        )
        visualizer.draw_bbox3d_and_center(
            kpt_2d_est_zoom,
            top_color=_GREEN,
            bottom_color=_GREY,
            linewidth=linewidth,
            draw_center=True,
        )
    vis_im = visualizer.get_output()
    save_path = osp.join(vis_dir, "{}_{:06d}_gt_est.png".format(scene_id, im_id))
    vis_im.save(save_path)
    # os.system("eog {}".format(save_path))
