import mmcv
import os.path as osp
import numpy as np
import sys
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import torch
import pandas as pd

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../../"))

from lib.vis_utils.colormap import colormap
from lib.utils.mask_utils import mask2bbox_xyxy, cocosegm2mask, get_edge
from core.utils.data_utils import crop_resize_by_warp_affine, read_image_mmcv
from core.gdrn_modeling.datasets.dataset_factory import register_datasets
from transforms3d.quaternions import quat2mat
from lib.egl_renderer.egl_renderer_v3 import EGLRenderer


out_size = 512
score_thr = 0.3
colors = colormap(rgb=False, maximum=255)

# object info
id2obj = {
    1: "002_master_chef_can",  # [1.3360, -0.5000, 3.5105]
    2: "003_cracker_box",  # [0.5575, 1.7005, 4.8050]
    3: "004_sugar_box",  # [-0.9520, 1.4670, 4.3645]
    4: "005_tomato_soup_can",  # [-0.0240, -1.5270, 8.4035]
    5: "006_mustard_bottle",  # [1.2995, 2.4870, -11.8290]
    6: "007_tuna_fish_can",  # [-0.1565, 0.1150, 4.2625]
    7: "008_pudding_box",  # [1.1645, -4.2015, 3.1190]
    8: "009_gelatin_box",  # [1.4460, -0.5915, 3.6085]
    9: "010_potted_meat_can",  # [2.4195, 0.3075, 8.0715]
    10: "011_banana",  # [-18.6730, 12.1915, -1.4635]
    11: "019_pitcher_base",  # [5.3370, 5.8855, 25.6115]
    12: "021_bleach_cleanser",  # [4.9290, -2.4800, -13.2920]
    13: "024_bowl",  # [-0.2270, 0.7950, -2.9675]
    14: "025_mug",  # [-8.4675, -0.6995, -1.6145]
    15: "035_power_drill",  # [9.0710, 20.9360, -2.1190]
    16: "036_wood_block",  # [1.4265, -2.5305, 17.1890]
    17: "037_scissors",  # [7.0535, -28.1320, 0.0420]
    18: "040_large_marker",  # [0.0460, -2.1040, 0.3500]
    19: "051_large_clamp",  # [10.5180, -1.9640, -0.4745]
    20: "052_extra_large_clamp",  # [-0.3950, -10.4130, 0.1620]
    21: "061_foam_brick",  # [-0.0805, 0.0805, -8.2435]
}
objects = list(id2obj.values())


def load_predicted_csv(fname):
    df = pd.read_csv(fname)
    info_list = df.to_dict("records")
    return info_list


def parse_Rt_in_csv(_item):
    return np.array([float(i) for i in _item.strip(" ").split(" ")])


tensor_kwargs = {"device": torch.device("cuda"), "dtype": torch.float32}
image_tensor = torch.empty((out_size, out_size, 4), **tensor_kwargs).detach()
seg_tensor = torch.empty((out_size, out_size, 4), **tensor_kwargs).detach()
# image_tensor = torch.empty((480, 640, 4), **tensor_kwargs).detach()

model_dir = "datasets/BOP_DATASETS/ycbv/models/"

model_paths = [osp.join(model_dir, f"obj_{obj_id:06d}.ply") for obj_id in id2obj]
texture_paths = [osp.join(model_dir, f"obj_{obj_id:06d}.png") for obj_id in id2obj]

ren = EGLRenderer(
    model_paths,
    texture_paths=texture_paths,
    vertex_scale=0.001,
    use_cache=True,
    width=out_size,
    height=out_size,
)

# NOTE: this is for ycbv_bop_test
pred_path = "output/gdrn/ycbv/a6_cPnP_AugAAETrunc_BG0.5_ycbv_real_pbr_visib20_20e_allObjs/a6-cPnP-AugAAETrunc-BG0.5-ycbv-real-pbr-visib20-20e-singleObjMerged-bop-test-iter0_ycbv-test.csv"

vis_dir = "output/gdrn/ycbv/a6_cPnP_AugAAETrunc_BG0.5_ycbv_real_pbr_visib20_20e_allObjs/ycbv_test_keyframe/vis_gt_pred"
mmcv.mkdir_or_exist(vis_dir)

print(pred_path)
preds_csv = load_predicted_csv(pred_path)
preds = {}
for item in preds_csv:
    im_key = "{}/{}".format(item["scene_id"], item["im_id"])
    item["time"] = float(item["time"])
    item["score"] = float(item["score"])
    item["R"] = parse_Rt_in_csv(item["R"]).reshape(3, 3)
    item["t"] = parse_Rt_in_csv(item["t"]) / 1000
    item["obj_name"] = id2obj[item["obj_id"]]
    if im_key not in preds:
        preds[im_key] = []
    preds[im_key].append(item)

dataset_name = "ycbv_test"
print(dataset_name)
register_datasets([dataset_name])

meta = MetadataCatalog.get(dataset_name)
print("MetadataCatalog: ", meta)
objs = meta.objs

dset_dicts = DatasetCatalog.get(dataset_name)
for d in tqdm(dset_dicts):
    K = d["cam"]
    file_name = d["file_name"]
    scene_im_id = d["scene_im_id"]
    img = read_image_mmcv(file_name, format="BGR")

    scene_im_id_split = d["scene_im_id"].split("/")
    scene_id = scene_im_id_split[0]
    im_id = int(scene_im_id_split[1])

    imH, imW = img.shape[:2]
    annos = d["annotations"]
    masks = [cocosegm2mask(anno["segmentation"], imH, imW) for anno in annos]
    fg_mask = sum(masks).astype("bool").astype("uint8")
    minx, miny, maxx, maxy = mask2bbox_xyxy(fg_mask)

    bboxes = [anno["bbox"] for anno in annos]
    bbox_modes = [anno["bbox_mode"] for anno in annos]
    bboxes_xyxy = np.array(
        [BoxMode.convert(box, box_mode, BoxMode.XYXY_ABS) for box, box_mode in zip(bboxes, bbox_modes)]
    )

    quats = [anno["quat"] for anno in annos]
    transes = [anno["trans"] for anno in annos]
    Rs = [quat2mat(quat) for quat in quats]
    # 0-based label
    cat_ids = [anno["category_id"] for anno in annos]
    obj_names = [objs[cat_id] for cat_id in cat_ids]

    gt_Rs = []
    gt_ts = []
    gt_labels = []

    for anno_i, anno in enumerate(annos):
        obj_name = obj_names[anno_i]
        gt_labels.append(objects.index(obj_name))  # 0-based label

        gt_Rs.append(Rs[anno_i])
        gt_ts.append(transes[anno_i])

    if scene_im_id not in preds:
        print(scene_im_id, "not detected")
        continue
    cur_preds = preds[scene_im_id]
    kpts_2d_est = []
    est_Rs = []
    est_ts = []
    est_labels = []
    for pred_i, pred in enumerate(cur_preds):
        try:
            R_est = pred["R"]
            t_est = pred["t"]
            score = pred["score"]
            obj_name = pred["obj_name"]
        except:
            continue
        if score < score_thr:
            continue

        est_Rs.append(R_est)
        est_ts.append(t_est)
        est_labels.append(objects.index(obj_name))  # 0-based label

    center = np.array([(minx + maxx) / 2, (miny + maxy) / 2])
    scale = max(maxx - minx, maxy - miny) * 1.5  # + 10
    crop_minx = max(0, center[0] - scale / 2)
    crop_miny = max(0, center[1] - scale / 2)
    crop_maxx = min(imW - 1, center[0] + scale / 2)
    crop_maxy = min(imH - 1, center[1] + scale / 2)
    scale = min(scale, min(crop_maxx - crop_minx, crop_maxy - crop_miny))

    zoomed_im = crop_resize_by_warp_affine(img, center, scale, out_size)
    im_zoom_gray = mmcv.bgr2gray(zoomed_im, keepdim=True)
    im_zoom_gray_3 = np.concatenate([im_zoom_gray, im_zoom_gray, im_zoom_gray], axis=2)
    # print(im_zoom_gray.shape)
    K_zoom = K.copy()
    K_zoom[0, 2] -= center[0] - scale / 2
    K_zoom[1, 2] -= center[1] - scale / 2
    K_zoom[0, :] *= out_size / scale
    K_zoom[1, :] *= out_size / scale

    gt_poses = [np.hstack([_R, _t.reshape(3, 1)]) for _R, _t in zip(gt_Rs, gt_ts)]
    est_poses = [np.hstack([_R, _t.reshape(3, 1)]) for _R, _t in zip(est_Rs, est_ts)]

    ren.render(
        est_labels,
        est_poses,
        K=K_zoom,
        image_tensor=image_tensor,
        background=im_zoom_gray_3,
    )
    ren_bgr = (image_tensor[:, :, :3].detach().cpu().numpy() + 0.5).astype("uint8")

    for gt_label, gt_pose in zip(gt_labels, gt_poses):
        ren.render([gt_label], [gt_pose], K=K_zoom, seg_tensor=seg_tensor)
        gt_mask = (seg_tensor[:, :, 0].detach().cpu().numpy() > 0).astype("uint8")
        gt_edge = get_edge(gt_mask, bw=3, out_channel=1)
        ren_bgr[gt_edge != 0] = np.array(mmcv.color_val("blue"))

    for est_label, est_pose in zip(est_labels, est_poses):
        ren.render([est_label], [est_pose], K=K_zoom, seg_tensor=seg_tensor)
        est_mask = (seg_tensor[:, :, 0].detach().cpu().numpy() > 0).astype("uint8")
        est_edge = get_edge(est_mask, bw=3, out_channel=1)
        ren_bgr[est_edge != 0] = np.array(mmcv.color_val("green"))

    vis_im = ren_bgr

    # vis_im_add = (im_zoom_gray_3 * 0.3 + ren_bgr * 0.7).astype("uint8")

    save_path_0 = osp.join(vis_dir, "{}_{:06d}_vis0.png".format(scene_id, im_id))
    mmcv.imwrite(zoomed_im, save_path_0)

    save_path_1 = osp.join(vis_dir, "{}_{:06d}_vis1.png".format(scene_id, im_id))
    mmcv.imwrite(vis_im, save_path_1)

    # if True:
    #     # grid_show([zoomed_im[:, :, ::-1], vis_im[:, :, ::-1]], ["im", "est"], row=1, col=2)
    #     # im_show = cv2.hconcat([zoomed_im, vis_im, vis_im_add])
    #     im_show = cv2.hconcat([zoomed_im, vis_im])
    #     cv2.imshow("im_est", im_show)
    #     if cv2.waitKey(0) == 27:
    #         break  # esc to quit
