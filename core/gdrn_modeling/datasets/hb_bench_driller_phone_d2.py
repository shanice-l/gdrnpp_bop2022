# NOTE: different from Self6D-v1 which uses hb-v1, this uses hb_bop conventions
import hashlib
import logging
import os
import os.path as osp
import sys
import time
from collections import OrderedDict
import mmcv
import numpy as np
from tqdm import tqdm
from transforms3d.quaternions import mat2quat, quat2mat
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)

import ref

from lib.pysixd import inout, misc
from lib.utils.mask_utils import binary_mask_to_rle, cocosegm2mask
from lib.utils.utils import dprint, iprint, lazy_property


logger = logging.getLogger(__name__)
DATASETS_ROOT = osp.normpath(osp.join(PROJ_ROOT, "datasets"))


class HB_BenchDrillerPhone:
    """a test sequence (test sequence 2) of HomebrewedDB contains 3 objects in
    linemod."""

    def __init__(self, data_cfg):
        self.name = data_cfg["name"]
        self.data_cfg = data_cfg

        self.objs = data_cfg["objs"]  # selected objects

        self.dataset_root = dataset_root = data_cfg["dataset_root"]
        self.ann_files = data_cfg["ann_files"]
        self.models_root = data_cfg["models_root"]  # models_lm
        self.scale_to_meter = data_cfg["scale_to_meter"]

        # use the images with converted K
        cam_type = data_cfg["cam_type"]
        assert cam_type in ["linemod", "hb"]
        self.cam_type = cam_type
        if cam_type == "linemod":  # linemod K
            self.cam = np.array(
                [[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]],
                dtype="float32",
            )
            self.rgb_root = osp.join(dataset_root, "sequence/rgb_lmK")
            self.depth_root = osp.join(dataset_root, "sequence/depth_lmK")
            self.mask_visib_root = osp.join(dataset_root, "sequence/mask_visib_lmK")
        else:  # hb
            self.cam = np.array(
                [[537.4799, 0, 318.8965], [0, 536.1447, 238.3781], [0, 0, 1]],
                dtype="float32",
            )
            self.rgb_root = osp.join(dataset_root, "sequence/rgb")
            self.depth_root = osp.join(dataset_root, "sequence/depth")
            self.mask_visib_root = osp.join(dataset_root, "sequence/mask_visib")
        assert osp.exists(self.rgb_root), self.rgb_root

        self.with_masks = data_cfg.get("with_masks", True)
        self.with_depth = data_cfg.get("with_depth", True)

        self.height = data_cfg["height"]
        self.width = data_cfg["width"]
        self.cache_dir = data_cfg.get("cache_dir", osp.join(PROJ_ROOT, ".cache"))  # .cache
        self.use_cache = data_cfg.get("use_cache", True)
        self.num_to_load = data_cfg["num_to_load"]  # -1
        self.filter_invalid = data_cfg["filter_invalid"]
        ##################################################

        # NOTE: careful! Only the selected objects
        self.cat_ids = [cat_id for cat_id, obj_name in ref.hb_bdp.id2obj.items() if obj_name in self.objs]
        # map selected objs to [0, num_objs-1]
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}  # id_map
        self.label2cat = {label: cat for cat, label in self.cat2label.items()}
        self.obj2label = OrderedDict((obj, obj_id) for obj_id, obj in enumerate(self.objs))
        ##########################################################

    def __call__(self):
        hashed_file_name = hashlib.md5(
            (
                "".join([str(fn) for fn in self.objs])
                + "dataset_dicts_{}_{}_{}_{}_{}".format(
                    self.name, self.dataset_root, self.with_masks, self.with_depth, __name__
                )
            ).encode("utf-8")
        ).hexdigest()
        cache_path = osp.join(
            self.cache_dir,
            "dataset_dicts_{}_{}.pkl".format(self.name, hashed_file_name),
        )

        if osp.exists(cache_path) and self.use_cache:
            logger.info("load cached dataset dicts from {}".format(cache_path))
            return mmcv.load(cache_path)

        t_start = time.perf_counter()
        dataset_dicts = []
        im_id_global = 0

        logger.info("loading dataset dicts: {}".format(self.name))
        self.num_instances_without_valid_segmentation = 0
        self.num_instances_without_valid_box = 0
        # NOTE: converted from gt_v1, obj_id --> obj_id+1
        gt_path = osp.join(self.dataset_root, "sequence/gt_v2.json")
        gt_dict = mmcv.load(gt_path)

        # determine which images to load by self.ann_files
        sel_im_ids = []
        for ann_file in self.ann_files:
            with open(ann_file, "r") as f:
                for line in f:
                    line = line.strip("\r\n")
                    cur_im_id = int(line)
                    if cur_im_id not in sel_im_ids:
                        sel_im_ids.append(cur_im_id)

        for str_im_id, annos in tqdm(gt_dict.items()):  # str im ids
            int_im_id = int(str_im_id)
            if int_im_id not in sel_im_ids:
                continue
            rgb_path = osp.join(self.rgb_root, "color_{:06d}.png".format(int_im_id))
            depth_path = osp.join(self.depth_root, "{:06d}.png".format(int_im_id))

            scene_id = 2  # dummy (because in the whole test set, its scene id is 2)
            record = {
                "dataset_name": self.name,
                "file_name": osp.relpath(rgb_path, PROJ_ROOT),
                "depth_file": osp.relpath(depth_path, PROJ_ROOT),
                "depth_factor": 1 / self.scale_to_meter,
                "height": self.height,
                "width": self.width,
                "image_id": im_id_global,
                "scene_im_id": "{}/{}".format(scene_id, int_im_id),  # for evaluation
                "cam": self.cam,
                "img_type": "real",
            }
            im_id_global += 1

            inst_annos = []
            for anno_i, anno in enumerate(annos):
                obj_id = anno["obj_id"]
                cls_name = ref.hb_bdp.id2obj[obj_id]
                if cls_name not in self.objs:
                    continue
                if cls_name not in ref.hb_bdp.objects:  # only support 3 objects
                    continue

                cur_label = self.cat2label[obj_id]

                R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                t = np.array(anno["cam_t_m2c"], dtype="float32") / 1000.0
                pose = np.hstack([R, t.reshape(3, 1)])
                if self.cam_type == "hb":
                    bbox = anno["obj_bb"]
                    bbox_mode = BoxMode.XYWH_ABS
                elif self.cam_type == "linemod":
                    # get bbox from projected points
                    bbox = misc.compute_2d_bbox_xyxy_from_pose_v2(
                        self.models[cur_label]["pts"].astype("float32"),
                        pose.astype("float32"),
                        self.cam,
                        width=self.width,
                        height=self.height,
                        clip=True,
                    )
                    bbox_mode = BoxMode.XYXY_ABS
                    x1, y1, x2, y2 = bbox
                    w = x2 - x1
                    h = y2 - y1
                else:
                    raise ValueError("Wrong cam type: {}".format(self.cam_type))

                if self.filter_invalid:
                    if h <= 1 or w <= 1:
                        self.num_instances_without_valid_box += 1
                        continue

                mask_visib_file = osp.join(
                    self.mask_visib_root,
                    "{:06d}_{:06d}.png".format(int_im_id, anno_i),
                )
                assert osp.exists(mask_visib_file), mask_visib_file
                # load mask visib  TODO: load both mask_visib and mask_full
                mask_single = mmcv.imread(mask_visib_file, "unchanged")
                area = mask_single.sum()
                if area < 3:  # filter out too small or nearly invisible instances
                    self.num_instances_without_valid_segmentation += 1
                    continue
                mask_rle = binary_mask_to_rle(mask_single, compressed=True)

                quat = mat2quat(R).astype("float32")

                proj = (record["cam"] @ t.T).T
                proj = proj[:2] / proj[2]

                inst = {
                    "category_id": cur_label,  # 0-based label
                    "bbox": bbox,
                    "bbox_mode": bbox_mode,
                    "pose": pose,
                    "quat": quat,
                    "trans": t,
                    "centroid_2d": proj,  # absolute (cx, cy)
                    "segmentation": mask_rle,
                }

                # NOTE: currently no xyz
                # if "test" not in self.name:
                #     xyz_path = osp.join(xyz_root, f"{int_im_id:06d}_{anno_i:06d}.pkl")
                #     assert osp.exists(xyz_path), xyz_path
                #     inst["xyz_path"] = xyz_path

                model_info = self.models_info[str(obj_id)]
                inst["model_info"] = model_info
                for key in ["bbox3d_and_center"]:
                    inst[key] = self.models[cur_label][key]
                inst_annos.append(inst)
            if len(inst_annos) == 0 and self.filter_invalid:  # filter im without anno
                continue
            record["annotations"] = inst_annos
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
        cache_path = osp.join(PROJ_ROOT, ".cache", "models_{}.pkl".format("_".join(self.objs)))
        if osp.exists(cache_path) and self.use_cache:
            # logger.info("load cached object models from {}".format(cache_path))
            return mmcv.load(cache_path)

        models = []
        for obj_name in self.objs:
            if obj_name not in ref.hb_bdp.objects:
                models.append(None)
                continue
            model = inout.load_ply(
                osp.join(self.models_root, "obj_{:06d}.ply".format(ref.hb_bdp.obj2id[obj_name])),
                vertex_scale=self.scale_to_meter,
            )
            # NOTE: the bbox3d_and_center is not obtained from centered vertices
            # for BOP models, not a big problem since they had been centered
            model["bbox3d_and_center"] = misc.get_bbox3d_and_center(model["pts"])

            models.append(model)
        logger.info("cache models to {}".format(cache_path))
        mmcv.mkdir_or_exist(osp.dirname(cache_path))
        mmcv.dump(models, cache_path, protocol=4)
        return models

    def image_aspect_ratio(self):
        return self.width / self.height  # 4/3


########### register datasets ############################################################


def get_hb_bdp_metadata(obj_names, ref_key):
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


SPLITS_HB_BenchviseDrillerPhone = dict(
    # TODO: maybe add scene name
    hb_benchvise_driller_phone_all_lmK=dict(
        name="hb_benchvise_driller_phone_all_lmK",
        dataset_root=osp.join(DATASETS_ROOT, "hb_bench_driller_phone"),
        models_root=osp.join(DATASETS_ROOT, "hb_bench_driller_phone/models_lm/"),
        ann_files=[osp.join(DATASETS_ROOT, "hb_bench_driller_phone/image_set/all.txt")],
        objs=["benchvise", "driller", "phone"],
        use_cache=True,
        num_to_load=-1,
        cam_type="linemod",
        scale_to_meter=0.001,
        filter_invalid=False,
        height=480,
        width=640,
        ref_key="hb_bdp",
    ),
    hb_benchvise_driller_phone_all=dict(
        name="hb_benchvise_driller_phone_all",
        dataset_root=osp.join(DATASETS_ROOT, "hb_bench_driller_phone"),
        models_root=osp.join(DATASETS_ROOT, "hb_bench_driller_phone/models_lm/"),
        ann_files=[osp.join(DATASETS_ROOT, "hb_bench_driller_phone/image_set/all.txt")],
        objs=["benchvise", "driller", "phone"],
        use_cache=True,
        num_to_load=-1,
        cam_type="hb",  # NOTE: hb K
        scale_to_meter=0.001,
        filter_invalid=False,
        height=480,
        width=640,
        ref_key="hb_bdp",
    ),
    hb_benchvise_driller_phone_test_lmK=dict(
        name="hb_benchvise_driller_phone_test_lmK",
        dataset_root=osp.join(DATASETS_ROOT, "hb_bench_driller_phone"),
        models_root=osp.join(DATASETS_ROOT, "hb_bench_driller_phone/models_lm/"),
        ann_files=[osp.join(DATASETS_ROOT, "hb_bench_driller_phone/image_set/test.txt")],
        objs=["benchvise", "driller", "phone"],
        use_cache=True,
        num_to_load=-1,
        cam_type="linemod",
        scale_to_meter=0.001,
        filter_invalid=False,
        height=480,
        width=640,
        ref_key="hb_bdp",
    ),
    hb_benchvise_driller_phone_test=dict(
        name="hb_benchvise_driller_phone_test",
        dataset_root=osp.join(DATASETS_ROOT, "hb_bench_driller_phone"),
        models_root=osp.join(DATASETS_ROOT, "hb_bench_driller_phone/models_lm/"),
        ann_files=[osp.join(DATASETS_ROOT, "hb_bench_driller_phone/image_set/test.txt")],
        objs=["benchvise", "driller", "phone"],
        use_cache=True,
        num_to_load=-1,
        cam_type="hb",
        scale_to_meter=0.001,
        filter_invalid=False,
        height=480,
        width=640,
        ref_key="hb_bdp",
    ),
)


# add varying percent splits
VARY_PERCENT_SPLITS = [
    "test100",
    "train090",
    "train180",
    "train270",
    "train360",
    "train450",
    "train540",
    "train630",
    "train720",
    "train810",
    "train900",
]

# all objects
for _split in VARY_PERCENT_SPLITS:
    for cam_type in ["linemod", "hb"]:
        K_str = "_lmK" if cam_type == "linemod" else ""
        name = "hb_benchvise_driller_phone_{}{}".format(_split, K_str)
        if name not in SPLITS_HB_BenchviseDrillerPhone:
            SPLITS_HB_BenchviseDrillerPhone[name] = dict(
                name=name,
                dataset_root=osp.join(DATASETS_ROOT, "hb_bench_driller_phone"),
                models_root=osp.join(DATASETS_ROOT, "hb_bench_driller_phone/models_lm/"),
                ann_files=[osp.join(DATASETS_ROOT, f"hb_bench_driller_phone/image_set/{_split}.txt")],
                objs=["benchvise", "driller", "phone"],
                use_cache=True,
                num_to_load=-1,
                cam_type=cam_type,
                scale_to_meter=0.001,
                filter_invalid=False,
                height=480,
                width=640,
                ref_key="hb_bdp",
            )

# single obj splits
for obj in ref.hb_bdp.objects:
    for split in ["test", "train", "all"] + VARY_PERCENT_SPLITS:
        for cam_type in ["linemod", "hb"]:
            K_str = "_lmK" if cam_type == "linemod" else ""
            name = "hb_bdp_{}_{}{}".format(obj, split, K_str)
            if name not in SPLITS_HB_BenchviseDrillerPhone:
                SPLITS_HB_BenchviseDrillerPhone[name] = dict(
                    name=name,
                    dataset_root=osp.join(DATASETS_ROOT, "hb_bench_driller_phone"),
                    models_root=osp.join(DATASETS_ROOT, "hb_bench_driller_phone/models_lm/"),
                    ann_files=[osp.join(DATASETS_ROOT, f"hb_bench_driller_phone/image_set/{split}.txt")],
                    objs=[obj],
                    use_cache=True,
                    num_to_load=-1,
                    cam_type=cam_type,
                    scale_to_meter=0.001,
                    filter_invalid=False,
                    height=480,
                    width=640,
                    ref_key="hb_bdp",
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
    if name in SPLITS_HB_BenchviseDrillerPhone:
        used_cfg = SPLITS_HB_BenchviseDrillerPhone[name]
    else:
        assert data_cfg is not None, f"dataset name {name} is not registered"
        used_cfg = data_cfg
    DatasetCatalog.register(name, HB_BenchDrillerPhone(used_cfg))
    # something like eval_types
    MetadataCatalog.get(name).set(
        ref_key=used_cfg["ref_key"],
        objs=used_cfg["objs"],
        eval_error_types=["ad", "rete", "proj"],
        evaluator_type="bop",  # NOTE: should not be bop
        **get_hb_bdp_metadata(obj_names=used_cfg["objs"], ref_key=used_cfg["ref_key"]),
    )


def get_available_datasets():
    return list(SPLITS_HB_BenchviseDrillerPhone.keys())


#### tests ###############################################
def test_vis():
    # python -m core.datasets.lm_dataset_d2 lmo_syn_vispy_train
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
        depth = mmcv.imread(d["depth_file"], "unchanged") / 1000.0

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

            grid_show(
                [
                    img[:, :, [2, 1, 0]],
                    img_vis[:, :, [2, 1, 0]],
                    img_vis_kpts2d[:, :, [2, 1, 0]],
                    depth,
                ],
                ["img", "vis_img", "img_vis_kpts2d", "depth"],
                row=2,
                col=2,
            )


if __name__ == "__main__":
    """Test the  dataset loader.

    Usage:
        python -m this_module dataset_name
    """
    from lib.vis_utils.image import grid_show
    from lib.utils.setup_logger import setup_my_logger
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    from lib.vis_utils.image import (
        vis_image_mask_bbox_cv2,
        vis_image_bboxes_cv2,
    )
    from lib.utils.mask_utils import cocosegm2mask
    from lib.utils.bbox_utils import xywh_to_xyxy
    from core.utils.data_utils import read_image_mmcv

    print("sys.argv:", sys.argv)
    logger = setup_my_logger(name="core")

    register_with_name_cfg(sys.argv[1])
    print("dataset catalog: ", DatasetCatalog.list())
    test_vis()
