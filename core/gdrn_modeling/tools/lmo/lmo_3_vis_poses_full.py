import mmcv
import os.path as osp
import numpy as np
import sys
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import torch

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../../"))

from lib.vis_utils.colormap import colormap
from lib.utils.mask_utils import cocosegm2mask, get_edge
from core.utils.data_utils import read_image_mmcv
from core.gdrn_modeling.datasets.dataset_factory import register_datasets
from transforms3d.quaternions import quat2mat
from lib.egl_renderer.egl_renderer_v3 import EGLRenderer

score_thr = 0.3
colors = colormap(rgb=False, maximum=255)

id2obj = {
    1: "ape",
    #  2: 'benchvise',
    #  3: 'bowl',
    #  4: 'camera',
    5: "can",
    6: "cat",
    #  7: 'cup',
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    #  13: 'iron',
    #  14: 'lamp',
    #  15: 'phone'
}
objects = list(id2obj.values())

width = 640
height = 480
tensor_kwargs = {"device": torch.device("cuda"), "dtype": torch.float32}
image_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()
seg_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()
# image_tensor = torch.empty((480, 640, 4), **tensor_kwargs).detach()

model_dir = "datasets/BOP_DATASETS/lmo/models/"

model_paths = [osp.join(model_dir, f"obj_{obj_id:06d}.ply") for obj_id in id2obj]

ren = EGLRenderer(model_paths, vertex_scale=0.001, use_cache=True, width=width, height=height)

# NOTE:
pred_path = "output/gdrn/lmo/a6_cPnP_AugAAETrunc_BG0.5_lmo_real_pbr0.1_40e/inference_model_final/lmo_test/a6-cPnP-AugAAETrunc-BG0.5-lmo-real-pbr0.1-40e-test_lmo_test_preds.pkl"

vis_dir = (
    "output/gdrn/lmo/a6_cPnP_AugAAETrunc_BG0.5_lmo_real_pbr0.1_40e/inference_model_final/lmo_test/lmo_vis_gt_pred_full"
)
mmcv.mkdir_or_exist(vis_dir)

print(pred_path)
preds = mmcv.load(pred_path)

dataset_name = "lmo_test"
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
    quats = [anno["quat"] for anno in annos]
    transes = [anno["trans"] for anno in annos]
    Rs = [quat2mat(quat) for quat in quats]
    # 0-based label
    cat_ids = [anno["category_id"] for anno in annos]

    obj_names = [objs[cat_id] for cat_id in cat_ids]

    est_Rs = []
    est_ts = []

    gt_Rs = []
    gt_ts = []

    labels = []

    for anno_i, anno in enumerate(annos):
        obj_name = obj_names[anno_i]

        try:
            R_est = preds[obj_name][file_name]["R"]
            t_est = preds[obj_name][file_name]["t"]
            score = preds[obj_name][file_name]["score"]
        except:
            continue
        if score < score_thr:
            continue

        labels.append(objects.index(obj_name))  # 0-based label

        est_Rs.append(R_est)
        est_ts.append(t_est)
        gt_Rs.append(Rs[anno_i])
        gt_ts.append(transes[anno_i])

    im_gray = mmcv.bgr2gray(img, keepdim=True)
    im_gray_3 = np.concatenate([im_gray, im_gray, im_gray], axis=2)

    gt_poses = [np.hstack([_R, _t.reshape(3, 1)]) for _R, _t in zip(gt_Rs, gt_ts)]
    poses = [np.hstack([_R, _t.reshape(3, 1)]) for _R, _t in zip(est_Rs, est_ts)]

    ren.render(labels, poses, K=K, image_tensor=image_tensor, background=im_gray_3)
    ren_bgr = (image_tensor[:, :, :3].detach().cpu().numpy() + 0.5).astype("uint8")

    for label, gt_pose, est_pose in zip(labels, gt_poses, poses):
        ren.render([label], [gt_pose], K=K, seg_tensor=seg_tensor)
        gt_mask = (seg_tensor[:, :, 0].detach().cpu().numpy() > 0).astype("uint8")

        ren.render([label], [est_pose], K=K, seg_tensor=seg_tensor)
        est_mask = (seg_tensor[:, :, 0].detach().cpu().numpy() > 0).astype("uint8")

        gt_edge = get_edge(gt_mask, bw=3, out_channel=1)
        est_edge = get_edge(est_mask, bw=3, out_channel=1)

        ren_bgr[gt_edge != 0] = np.array(mmcv.color_val("blue"))
        ren_bgr[est_edge != 0] = np.array(mmcv.color_val("green"))

    vis_im = ren_bgr

    save_path_0 = osp.join(vis_dir, "{}_{:06d}_vis0.png".format(scene_id, im_id))
    mmcv.imwrite(img, save_path_0)

    save_path = osp.join(vis_dir, "{}_{:06d}_vis1.png".format(scene_id, im_id))
    mmcv.imwrite(vis_im, save_path)

    # if True:
    #     # grid_show([img[:, :, ::-1], vis_im[:, :, ::-1]], ["im", "est"], row=1, col=2)
    #     # im_show = cv2.hconcat([img, vis_im, vis_im_add])
    #     im_show = cv2.hconcat([img, vis_im])
    #     cv2.imshow("im_est", im_show)
    #     if cv2.waitKey(0) == 27:
    #         break  # esc to quit

# ffmpeg -r 5 -f image2 -s 1920x1080 -pattern_type glob -i "./lmo_vis_gt_pred_full_video/*.png" -vcodec libx264 -crf 25  -pix_fmt yuv420p lmo_vis_video.mp4
