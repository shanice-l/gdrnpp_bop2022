import hashlib
import logging
import os
import os.path as osp
import sys

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)
import time
from collections import OrderedDict
import mmcv
import numpy as np
from tqdm import tqdm
from transforms3d.quaternions import mat2quat, quat2mat
import ref
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from lib.pysixd import inout, misc
from lib.utils.mask_utils import binary_mask_to_rle, cocosegm2mask, mask2bbox_xywh
from lib.utils.utils import dprint, iprint, lazy_property


logger = logging.getLogger(__name__)
DATASETS_ROOT = osp.normpath(osp.join(PROJ_ROOT, "datasets"))


class LM_SYN_EGL_Dataset(object):
    """lm synthetic data by egl renderer."""

    def __init__(self, data_cfg):
        """
        Set with_depth and with_masks default to True,
        and decide whether to load them into dataloader/network later
        with_masks:
        """
        self.name = data_cfg["name"]
        self.data_cfg = data_cfg

        self.objs = data_cfg["objs"]  # selected objects

        self.dataset_root = data_cfg.get("dataset_root", osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/train_egl"))
        self.xyz_root = data_cfg.get("xyz_root", osp.join(self.dataset_root, "xyz_crop"))
        assert osp.exists(self.dataset_root), self.dataset_root

        self.gt_path = osp.join(self.dataset_root, "gt.json")
        assert osp.exists(self.gt_path)

        self.models_root = data_cfg["models_root"]  # BOP_DATASETS/lm/models
        self.scale_to_meter = data_cfg["scale_to_meter"]  # 0.001

        self.with_masks = data_cfg["with_masks"]
        self.with_depth = data_cfg["with_depth"]
        self.depth_factor = data_cfg.get("depth_factor", 10000.0)

        self.height = data_cfg["height"]  # 480
        self.width = data_cfg["width"]  # 640

        self.cache_dir = data_cfg.get("cache_dir", osp.join(PROJ_ROOT, ".cache"))  # .cache
        self.use_cache = data_cfg.get("use_cache", True)
        self.num_to_load = data_cfg["num_to_load"]  # -1
        self.filter_invalid = data_cfg.get("filter_invalid", True)
        ##################################################
        self.cam = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])

        # NOTE: careful! Only the selected objects
        self.cat_ids = [cat_id for cat_id, obj_name in ref.lm_full.id2obj.items() if obj_name in self.objs]
        # map selected objs to [0, num_objs-1]
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}  # id_map
        self.label2cat = {label: cat for cat, label in self.cat2label.items()}
        self.obj2label = OrderedDict((obj, obj_id) for obj_id, obj in enumerate(self.objs))
        ##########################################################

    def __call__(self):  # LM_SYN_EGL_Dataset
        """Load light-weight instance annotations of all images into a list of
        dicts in Detectron2 format.

        Do not load heavy data into memory in this file, since we will
        load the annotations of all images into memory.
        """
        # cache the dataset_dicts to avoid loading masks from files
        hashed_file_name = hashlib.md5(
            (
                "".join([str(fn) for fn in self.objs])
                + "dataset_dicts_{}_{}_{}_{}_{}_{}".format(
                    self.name,
                    self.dataset_root,
                    self.with_masks,
                    self.with_depth,
                    self.num_to_load,
                    __name__,
                )
            ).encode("utf-8")
        ).hexdigest()
        cache_path = osp.join(self.cache_dir, "dataset_dicts_{}_{}.pkl".format(self.name, hashed_file_name))

        if osp.exists(cache_path) and self.use_cache:
            logger.info("load cached dataset dicts from {}".format(cache_path))
            return mmcv.load(cache_path)

        t_start = time.perf_counter()

        logger.info("loading dataset dicts: {}".format(self.name))
        self.num_instances_without_valid_segmentation = 0
        self.num_instances_without_valid_box = 0
        dataset_dicts = []  #######################################################
        gt_dict = mmcv.load(self.gt_path)
        unique_im_id = 0
        for str_im_id, annos in tqdm(gt_dict.items()):
            int_im_id = int(str_im_id)

            rgb_path = osp.join(self.dataset_root, "rgb/{:06d}.jpg".format(int_im_id))
            assert osp.exists(rgb_path), rgb_path

            depth_path = osp.join(self.dataset_root, "depth/{:06d}.png".format(int_im_id))
            record = {
                "dataset_name": self.name,
                "file_name": osp.relpath(rgb_path, PROJ_ROOT),
                "depth_file": osp.relpath(depth_path, PROJ_ROOT),
                "height": self.height,
                "width": self.width,
                "image_id": unique_im_id,
                "scene_im_id": f"0/{int_im_id}",  # for pose evaluation
                "cam": self.cam,
                "depth_factor": self.depth_factor,
                "img_type": "syn_egl",  # NOTE: has background
            }
            unique_im_id += 1
            insts = []
            for anno_i, anno in enumerate(annos):
                obj_id = anno["obj_id"]
                if obj_id not in self.cat_ids:
                    continue
                cur_label = self.cat2label[obj_id]  # 0-based label
                pose = np.array(anno["pose"])
                R = pose[:3, :3]
                t = pose[:3, 3]
                quat = mat2quat(R).astype("float32")
                proj = (record["cam"] @ t.T).T
                proj = proj[:2] / proj[2]

                mask_vis_rle = anno["mask_visib"]
                mask_full_rle = anno["mask_full"]
                bbox = anno["bbox"]

                x1, y1, w, h = bbox
                if self.filter_invalid:
                    if h <= 1 or w <= 1:
                        self.num_instances_without_valid_box += 1
                        continue
                mask_vis = cocosegm2mask(mask_vis_rle, self.height, self.width)
                vis_area = mask_vis.sum()
                if vis_area < 30:  # filter out too small or nearly invisible instances
                    self.num_instances_without_valid_segmentation += 1
                    continue
                mask_full = cocosegm2mask(mask_full_rle, self.height, self.width)
                full_area = mask_full.sum()
                if full_area > 0:
                    visib_fract = vis_area / full_area
                else:
                    visib_fract = 0

                xyz_path = osp.join(self.xyz_root, f"{int_im_id:06d}_{anno_i:06d}-xyz.pkl")
                assert osp.exists(xyz_path), xyz_path
                inst = {
                    "category_id": cur_label,  # 0-based label
                    "bbox": bbox,  # TODO: load both bbox_obj and bbox_visib
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "pose": pose,
                    "quat": quat,
                    "trans": t,
                    "centroid_2d": proj,  # absolute (cx, cy)
                    "segmentation": mask_vis_rle,
                    "mask_full": mask_full_rle,
                    "visib_fract": visib_fract,
                    "xyz_path": xyz_path,
                }

                model_info = self.models_info[str(obj_id)]
                inst["model_info"] = model_info
                # TODO: using full mask and full xyz
                for key in ["bbox3d_and_center"]:
                    inst[key] = self.models[cur_label][key]
                insts.append(inst)
            if len(insts) == 0:  # filter im without anno
                continue
            record["annotations"] = insts
            dataset_dicts.append(record)

        if self.num_instances_without_valid_segmentation > 0:
            logger.warning(
                "Filtered out {} instances without valid segmentation. "
                "There might be issues in your dataset generation process.".format(
                    self.num_instances_without_valid_segmentation
                )
            )
        if self.num_instances_without_valid_box > 0:
            logger.warning(
                "Filtered out {} instances without valid box. "
                "There might be issues in your dataset generation process.".format(self.num_instances_without_valid_box)
            )
        ##########################################################################
        if self.num_to_load > 0:
            self.num_to_load = min(int(self.num_to_load), len(dataset_dicts))
            dataset_dicts = dataset_dicts[: self.num_to_load]
        logger.info("loaded {} dataset dicts, using {}s".format(len(dataset_dicts), time.perf_counter() - t_start))

        mmcv.mkdir_or_exist(osp.dirname(cache_path))
        mmcv.dump(dataset_dicts, cache_path, protocol=4)
        logger.info("Dumped dataset_dicts to {}".format(cache_path))
        return dataset_dicts

    @lazy_property
    def models_info(self):
        models_info_path = osp.join(self.models_root, "models_info.json")
        assert osp.exists(models_info_path), models_info_path
        models_info = mmcv.load(models_info_path)  # key is str(obj_id)
        return models_info

    @lazy_property
    def models(self):
        """Load models into a list."""
        cache_path = osp.join(self.models_root, "models_{}.pkl".format("_".join(self.objs)))
        if osp.exists(cache_path) and self.use_cache:
            # dprint("{}: load cached object models from {}".format(self.name, cache_path))
            return mmcv.load(cache_path)

        models = []
        for obj_name in self.objs:
            model = inout.load_ply(
                osp.join(
                    self.models_root,
                    f"obj_{ref.lm_full.obj2id[obj_name]:06d}.ply",
                ),
                vertex_scale=self.scale_to_meter,
            )
            # NOTE: the bbox3d_and_center is not obtained from centered vertices
            # for BOP models, not a big problem since they had been centered
            model["bbox3d_and_center"] = misc.get_bbox3d_and_center(model["pts"])

            models.append(model)
        logger.info("cache models to {}".format(cache_path))
        mmcv.dump(models, cache_path, protocol=4)
        return models

    def image_aspect_ratio(self):
        return self.width / self.height  # 4/3


########### register datasets ############################################################


def get_lm_metadata(obj_names, ref_key):
    """task specific metadata."""

    data_ref = ref.__dict__[ref_key]

    cur_sym_infos = {}  # label based key
    loaded_models_info = data_ref.get_models_info()

    for i, obj_name in enumerate(obj_names):
        obj_id = data_ref.obj2id[obj_name]
        model_info = loaded_models_info[str(obj_id)]
        if "symmetries_discrete" in model_info or "symmetries_continuous" in model_info:
            sym_transforms = misc.get_symmetry_transformations(model_info, max_sym_disc_step=0.01)
            sym_info = np.array([sym["R"] for sym in sym_transforms], dtype=np.float32)
        else:
            sym_info = None
        cur_sym_infos[i] = sym_info

    meta = {"thing_classes": obj_names, "sym_infos": cur_sym_infos}
    return meta


LM_13_OBJECTS = [
    "ape",
    "benchvise",
    "camera",
    "can",
    "cat",
    "driller",
    "duck",
    "eggbox",
    "glue",
    "holepuncher",
    "iron",
    "lamp",
    "phone",
]  # no bowl, cup
LM_OCC_OBJECTS = [
    "ape",
    "can",
    "cat",
    "driller",
    "duck",
    "eggbox",
    "glue",
    "holepuncher",
]
lm_model_root = "BOP_DATASETS/lm/models/"
lmo_model_root = "BOP_DATASETS/lmo/models/"
################################################################################

SPLITS_LM_EGL = dict(
    lm_egl_13_train=dict(
        name="lm_egl_13_train",
        objs=LM_13_OBJECTS,  # selected objects
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/train_egl"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/models"),
        xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/train_egl/xyz_crop"),
        scale_to_meter=0.001,
        with_masks=True,  # (load masks but may not use it)
        with_depth=True,  # (load depth path here, but may not use it)
        depth_factor=10000.0,
        height=480,
        width=640,
        cache_dir=osp.join(PROJ_ROOT, ".cache"),
        use_cache=True,
        num_to_load=-1,
        filter_scene=True,
        ref_key="lm_full",
    ),
    lmo_egl_train=dict(
        name="lmo_egl_train",
        objs=LM_OCC_OBJECTS,  # selected objects
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/train_egl"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/models"),
        # NOTE: soft link to lm/train_egl
        xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/train_egl/xyz_crop"),
        scale_to_meter=0.001,
        with_masks=True,  # (load masks but may not use it)
        with_depth=True,  # (load depth path here, but may not use it)
        depth_factor=10000.0,
        height=480,
        width=640,
        cache_dir=osp.join(PROJ_ROOT, ".cache"),
        use_cache=True,
        num_to_load=-1,
        filter_invalid=True,
        ref_key="lmo_full",
    ),
)

# single obj splits
for obj in ref.lm_full.objects:
    for split in ["train"]:
        name = "lm_egl_{}_{}".format(obj, split)
        if split in ["train"]:
            filter_invalid = True
        else:
            raise ValueError("{}".format(split))
        if name not in SPLITS_LM_EGL:
            SPLITS_LM_EGL[name] = dict(
                name=name,
                objs=[obj],  # only this obj
                dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/train_egl"),
                models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/models"),
                xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/train_egl/xyz_crop"),
                scale_to_meter=0.001,
                with_masks=True,  # (load masks but may not use it)
                with_depth=True,  # (load depth path here, but may not use it)
                depth_factor=10000.0,
                height=480,
                width=640,
                cache_dir=osp.join(PROJ_ROOT, ".cache"),
                use_cache=True,
                num_to_load=-1,
                filter_invalid=filter_invalid,
                ref_key="lm_full",
            )

# lmo single objs
for obj in ref.lmo_full.objects:
    for split in ["train"]:
        name = "lmo_egl_{}_{}".format(obj, split)
        if split in ["train"]:
            filter_invalid = True
        else:
            raise ValueError("{}".format(split))
        if name not in SPLITS_LM_EGL:
            SPLITS_LM_EGL[name] = dict(
                name=name,
                objs=[obj],  # only this obj
                dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/train_egl"),
                models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/models"),
                # NOTE: soft link to lm/train_egl
                xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/train_egl/xyz_crop"),
                scale_to_meter=0.001,
                with_masks=True,  # (load masks but may not use it)
                with_depth=True,  # (load depth path here, but may not use it)
                depth_factor=10000.0,
                height=480,
                width=640,
                cache_dir=osp.join(PROJ_ROOT, ".cache"),
                use_cache=True,
                num_to_load=-1,
                filter_invalid=filter_invalid,
                ref_key="lmo_full",
            )


def register_with_name_cfg(name, data_cfg=None):
    """Assume pre-defined datasets live in `./datasets`.

    Args:
        name: datasnet_name,
        data_cfg: if name is in existing SPLITS, use pre-defined data_cfg
            otherwise requires data_cfg
            data_cfg can be set in cfg.DATA_CFG.name
    """
    dprint("register dataset: {}".format(name))
    if name in SPLITS_LM_EGL:
        used_cfg = SPLITS_LM_EGL[name]
    else:
        assert data_cfg is not None, f"dataset name {name} is not registered"
        used_cfg = data_cfg
    DatasetCatalog.register(name, LM_SYN_EGL_Dataset(used_cfg))
    # something like eval_types
    MetadataCatalog.get(name).set(
        ref_key=used_cfg["ref_key"],
        objs=used_cfg["objs"],
        eval_error_types=["ad", "rete", "proj"],
        evaluator_type="bop",
        **get_lm_metadata(obj_names=used_cfg["objs"], ref_key=used_cfg["ref_key"]),
    )


def get_available_datasets():
    return list(SPLITS_LM_EGL.keys())


#### tests ###############################################
def test_vis():
    dset_name = sys.argv[1]
    assert dset_name in DatasetCatalog.list()

    meta = MetadataCatalog.get(dset_name)
    dprint("MetadataCatalog: ", meta)
    objs = meta.objs

    t_start = time.perf_counter()
    dicts = DatasetCatalog.get(dset_name)
    logger.info("Done loading {} samples with {:.3f}s.".format(len(dicts), time.perf_counter() - t_start))

    dirname = "output/{}-data-vis".format(dset_name)
    os.makedirs(dirname, exist_ok=True)
    for d in dicts:
        img = read_image_mmcv(d["file_name"], format="BGR")
        depth = mmcv.imread(d["depth_file"], "unchanged") / 10000.0

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
        K = d["cam"]
        kpts_2d = [misc.project_pts(kpt3d, K, R, t) for kpt3d, R, t in zip(kpts_3d_list, Rs, transes)]
        # # TODO: visualize pose and keypoints
        labels = [objs[cat_id] for cat_id in cat_ids]
        for _i in range(len(annos)):
            img_vis = vis_image_mask_bbox_cv2(
                img,
                masks[_i : _i + 1],
                bboxes=bboxes_xyxy[_i : _i + 1],
                labels=labels[_i : _i + 1],
            )
            img_vis_kpts2d = misc.draw_projected_box3d(img_vis.copy(), kpts_2d[_i])
            xyz_path = annos[_i]["xyz_path"]
            xyz_info = mmcv.load(xyz_path)
            x1, y1, x2, y2 = xyz_info["xyxy"]
            xyz_crop = xyz_info["xyz_crop"].astype(np.float32)
            xyz = np.zeros((imH, imW, 3), dtype=np.float32)
            xyz[y1 : y2 + 1, x1 : x2 + 1, :] = xyz_crop
            xyz_show = get_emb_show(xyz)
            xyz_crop_show = get_emb_show(xyz_crop)
            img_xyz = img.copy() / 255.0
            mask_xyz = ((xyz[:, :, 0] != 0) | (xyz[:, :, 1] != 0) | (xyz[:, :, 2] != 0)).astype("uint8")
            fg_idx = np.where(mask_xyz != 0)
            img_xyz[fg_idx[0], fg_idx[1], :] = xyz_show[fg_idx[0], fg_idx[1], :3]
            img_xyz_crop = img_xyz[y1 : y2 + 1, x1 : x2 + 1, :]
            img_vis_crop = img_vis[y1 : y2 + 1, x1 : x2 + 1, :]
            # diff mask
            diff_mask_xyz = np.abs(masks[_i] - mask_xyz)[y1 : y2 + 1, x1 : x2 + 1]

            grid_show(
                [
                    img[:, :, [2, 1, 0]],
                    img_vis[:, :, [2, 1, 0]],
                    img_vis_kpts2d[:, :, [2, 1, 0]],
                    depth,
                    # xyz_show,
                    diff_mask_xyz,
                    xyz_crop_show,
                    img_xyz[:, :, [2, 1, 0]],
                    img_xyz_crop[:, :, [2, 1, 0]],
                    img_vis_crop,
                ],
                [
                    "img",
                    "vis_img",
                    "img_vis_kpts2d",
                    "depth",
                    "diff_mask_xyz",
                    "xyz_crop_show",
                    "img_xyz",
                    "img_xyz_crop",
                    "img_vis_crop",
                ],
                row=3,
                col=3,
            )


if __name__ == "__main__":
    """Test the  dataset loader.

    Usage:
        python -m this_module dataset_name
    """
    from lib.vis_utils.image import grid_show
    from lib.utils.setup_logger import setup_my_logger

    import detectron2.data.datasets  # noqa # add pre-defined metadata
    from lib.vis_utils.image import vis_image_mask_bbox_cv2
    from core.utils.utils import get_emb_show
    from core.utils.data_utils import read_image_mmcv

    print("sys.argv:", sys.argv)
    logger = setup_my_logger(name="core")
    register_with_name_cfg(sys.argv[1])
    print("dataset catalog: ", DatasetCatalog.list())

    test_vis()
