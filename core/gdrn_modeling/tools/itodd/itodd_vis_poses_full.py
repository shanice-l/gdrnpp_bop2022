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
from core.utils.data_utils import read_image_mmcv
from core.gdrn_modeling.datasets.dataset_factory import register_datasets
from transforms3d.quaternions import quat2mat
from lib.egl_renderer.egl_renderer_v3 import EGLRenderer
from lib.vis_utils.image import grid_show
from lib.vis_utils.image import vis_image_bboxes_cv2

score_thr = 0.3
colors = colormap(rgb=False, maximum=255)

# object info
id2obj = {i: str(i) for i in range(1, 28 + 1)}
objects = list(id2obj.values())


def load_predicted_csv(fname):
    df = pd.read_csv(fname)
    info_list = df.to_dict("records")
    return info_list


def parse_Rt_in_csv(_item):
    return np.array([float(i) for i in _item.strip(" ").split(" ")])


width = 1280
height = 960

tensor_kwargs = {"device": torch.device("cuda"), "dtype": torch.float32}
image_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()
seg_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()
# image_tensor = torch.empty((480, 640, 4), **tensor_kwargs).detach()

model_dir = "datasets/BOP_DATASETS/itodd/models/"

model_paths = [osp.join(model_dir, f"obj_{obj_id:06d}.ply") for obj_id in id2obj]

ren = EGLRenderer(
    model_paths,
    vertex_scale=0.001,
    use_cache=True,
    width=width,
    height=height,
)

# NOTE: this is for ycbv_bop_test
pred_path = osp.join(
    "output/gdrn/itodd/resnest50d_online_AugCosyAAEGray_mlBCE_Itodd_pbr_100e_bop_test",
    "all_itodd-test.csv",
)
vis_dir = "output/gdrn/itodd/resnest50d_online_AugCosyAAEGray_mlBCE_Itodd_pbr_100e_bop_test/vis"

bbox_path = "datasets/BOP_DATASETS/itodd/challenge2022-524061_itodd-test_bop.json"

mmcv.mkdir_or_exist(vis_dir)

preds_csv = load_predicted_csv(pred_path)
pred_bboxes = mmcv.load(bbox_path)
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

dataset_name = "itodd_bop_test"
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

    if scene_im_id not in preds:
        print(scene_im_id, "not detected")
        continue
    cur_preds = preds[scene_im_id]
    cur_bboxes = pred_bboxes[scene_im_id]
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

    bboxes = []
    labels = []
    for _i, bbox in enumerate(cur_bboxes):
        score = bbox["score"]
        if score < score_thr:
            continue
        x1, y1, w1, h1 = bbox["bbox_est"]
        x2 = x1 + w1
        y2 = y1 + h1
        bboxes.append(np.array([x1, y1, x2, y2]))
        labels.append(str(bbox["obj_id"]))

    img_bbox = vis_image_bboxes_cv2(
        img,
        bboxes,
        labels,
    )

    im_gray = mmcv.bgr2gray(img, keepdim=True)
    im_gray_3 = np.concatenate([im_gray, im_gray, im_gray], axis=2)

    est_poses = [np.hstack([_R, _t.reshape(3, 1)]) for _R, _t in zip(est_Rs, est_ts)]

    ren.render(
        est_labels,
        est_poses,
        K=K,
        image_tensor=image_tensor,
        background=im_gray_3,
    )
    ren_bgr = (image_tensor[:, :, :3].detach().cpu().numpy() + 0.5).astype("uint8")

    for est_label, est_pose in zip(est_labels, est_poses):
        ren.render([est_label], [est_pose], K=K, seg_tensor=seg_tensor)
        est_mask = (seg_tensor[:, :, 0].detach().cpu().numpy() > 0).astype("uint8")
        est_edge = get_edge(est_mask, bw=3, out_channel=1)
        ren_bgr[est_edge != 0] = np.array(mmcv.color_val("green"))

    vis_im = ren_bgr

    show = False
    if show:
        grid_show([img_bbox[:, :, ::-1], vis_im[:, :, ::-1]], ["im", "est"], row=1, col=2)
        # im_show = cv2.hconcat([img, vis_im, vis_im_add])
        # im_show = cv2.hconcat([img, vis_im])
        # cv2.imshow("im_est", im_show)
        # if cv2.waitKey(0) == 27:
        #     break  # esc to quit
    else:
        save_path_0 = osp.join(vis_dir, "{}_{:06d}_vis0.png".format(scene_id, im_id))
        mmcv.imwrite(img_bbox, save_path_0)

        save_path_1 = osp.join(vis_dir, "{}_{:06d}_vis1.png".format(scene_id, im_id))
        mmcv.imwrite(vis_im, save_path_1)
# ffmpeg -r 5 -f image2 -s 1920x1080 -pattern_type glob -i "./ycbv_vis_gt_pred_full_video/*.png" -vcodec libx264 -crf 25  -pix_fmt yuv420p ycbv_vis_video.mp4
