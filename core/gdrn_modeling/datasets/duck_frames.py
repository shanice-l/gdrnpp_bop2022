import hashlib
import logging
import os
import os.path as osp
import time
from collections import OrderedDict

import cv2
import mmcv
import numpy as np
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))

import ref

from lib.pysixd import inout, misc
from lib.utils.utils import dprint, iprint, lazy_property


logger = logging.getLogger(__name__)

DATASETS_ROOT = osp.normpath(osp.join(PROJ_ROOT, "datasets"))


class DUCK_FRAMES_Dataset(object):
    def __init__(self, data_cfg):
        """Set with_depth default to True, and decide whether to load them into
        dataloader/network later."""
        self.data_cfg = data_cfg
        self.name = data_cfg["name"]
        self.root = data_cfg.get("root", "datasets/duck_fabi")
        self.idx_files = data_cfg["idx_files"]
        self.models_root = data_cfg["models_root"]
        self.objs = objs = data_cfg.get("objs", ref.lm_full.objects)
        self.scale_to_meter = data_cfg.get("scale_to_meter", 0.001)
        self.with_depth = data_cfg.get("with_depth", True)
        self.height = data_cfg.get("height", 720)
        self.width = data_cfg.get("width", 1280)
        self.depth_factor = data_cfg.get("depth_factor", 1000)
        self.cache_dir = data_cfg.get("cache_dir", ".cache")
        self.use_cache = data_cfg.get("use_cache", True)
        self.num_to_load = data_cfg.get("num_to_load", -1)
        self.filter_invalid = data_cfg.get("filter_invalid", False)

        #####################################################
        # NOTE: careful! Only the selected objects
        self.cat_ids = [cat_id for cat_id, obj_name in ref.lm_full.id2obj.items() if obj_name in objs]
        # map selected objs to [0, num_objs-1]
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}  # id_map
        self.label2cat = {label: cat for cat, label in self.cat2label.items()}
        self.obj2label = OrderedDict((obj, obj_id) for obj_id, obj in enumerate(self.objs))

        self.images = []

        for idx_file in self.idx_files:
            assert osp.exists(idx_file), idx_file
            with open(idx_file, "r") as f:
                for line in f:
                    file_name = line.strip("\r\n")
                    image_path = osp.join(self.root, file_name)
                    assert osp.exists(image_path), image_path
                    self.images.append(image_path)  # load rgb image

        assert len(self.images) > 0, "wrong len of images: {}".format(len(self.images))

        if self.num_to_load > 0:
            self.num_to_load = min(self.num_to_load, len(self.images))
        else:
            self.num_to_load = len(self.images)
        logger.info("Dataset has {} images".format(len(self.images)))
        logger.info("num images to load: {}".format(self.num_to_load))

    def get_sample_dict(self, index):
        record = {}
        img_file = self.images[index]
        record["dataset_name"] = self.name
        record["file_name"] = osp.relpath(img_file, PROJ_ROOT)
        record["height"] = self.height
        record["width"] = self.width
        image_name = img_file.split("/")[-1]
        scene_id = 0
        image_id = image_name.split(".")[0].split("_")[-1]
        record["image_id"] = self._unique_id
        record["scene_im_id"] = "{}/{}".format(scene_id, image_id)
        # record["cam"] = ref.lm_full.camera_matrix
        record["cam"] = np.array([[572.4114, 0, 645.2611], [0, 573.57043, 362.04899], [0, 0, 1]], dtype=np.float32)
        return record

    def __call__(self):
        """Load light-weight instance annotations of all images into a list of
        dicts in Detectron2 format.

        Do not load heavy data into memory in this file, since we will
        load the annotations of all images into memory.
        """
        # cache the dataset_dicts to avoid loading masks from files
        hashed_file_name = hashlib.md5(
            (
                "".join([str(fn) for fn in self.objs])
                + "dataset_dicts_{}_{}_{}_{}".format(
                    self.name,
                    self.root,
                    self.with_depth,
                    __name__,
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
        logger.info("loading dataset dicts")
        indices = [i for i in range(self.num_to_load)]

        self._unique_id = 0
        for index in tqdm(indices):
            sample_dict = self.get_sample_dict(index)
            if sample_dict is not None:
                dataset_dicts.append(sample_dict)
            self._unique_id += 1
        logger.info("loaded {} dataset dicts, using {}s".format(len(dataset_dicts), time.perf_counter() - t_start))
        mmcv.dump(dataset_dicts, cache_path, protocol=4)
        logger.info("Dumped dataset_dicts to {}".format(cache_path))
        return dataset_dicts

    @lazy_property
    def models(self):
        """Load models into a list."""
        cache_path = osp.join(self.models_root, "models_{}.pkl".format("_".join(self.objs)))
        if osp.exists(cache_path) and self.use_cache:
            # logger.info("load cached object models from {}".format(cache_path))
            return mmcv.load(cache_path)

        models = []
        for obj_name in self.objs:
            model = inout.load_ply(
                osp.join(self.models_root, f"obj_{ref.lm_full.obj2id[obj_name]:06d}.ply"),
                vertex_scale=self.scale_to_meter,
            )
            # NOTE: the bbox3d_and_center is not obtained from centered vertices
            # for BOP models, not a big problem since they had been centered
            model["bbox3d_and_center"] = misc.get_bbox3d_and_center(model["pts"])
            models.append(model)
        logger.info("cache models to {}".format(cache_path))
        mmcv.dump(models, cache_path, protocol=4)
        return models

    def __len__(self):
        return self.num_to_load

    def image_aspect_ratio(self):
        return self.width / self.height  # 4/3


SPLITS_DUCK_FRAMES = dict(
    duck_frames_lm=dict(
        name="duck_frames_lm",
        root=osp.join(DATASETS_ROOT, "duck_fabi"),
        idx_files=[osp.join(DATASETS_ROOT, "duck_fabi/duck_frames.txt")],
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/models"),
        objs=["duck"],
        scale_to_meter=0.001,
        with_depth=False,
        depth_factor=1000.0,
        height=720,
        width=1280,
        cache_dir=".cache",
        use_cache=True,
        num_to_load=-1,
        filter_scene=False,
        filter_invalid=False,
        ref_key="lmo_full",
    ),
    duck_frames=dict(
        name="duck_frames",
        root=osp.join(DATASETS_ROOT, "duck_fabi"),
        idx_files=[osp.join(DATASETS_ROOT, "duck_fabi/duck_frames.txt")],
        models_root=osp.join(DATASETS_ROOT, "duck_fabi/models"),
        objs=["duck"],
        scale_to_meter=0.001,
        with_depth=False,
        depth_factor=1000.0,
        height=720,
        width=1280,
        cache_dir=".cache",
        use_cache=True,
        num_to_load=-1,
        filter_scene=False,
        filter_invalid=False,
        ref_key="lm_duck_fabi",
    ),
)


def register_duck_frames():
    for dset_name, data_cfg in SPLITS_DUCK_FRAMES.items():
        # if comm.is_main_process():
        #     iprint('register dataset: {}'.format(dset_name))
        DatasetCatalog.register(dset_name, DUCK_FRAMES_Dataset(data_cfg))
        MetadataCatalog.get(dset_name).set(
            ref_key=data_cfg["ref_key"],
            objs=data_cfg["objs"],
            eval_error_types=["ad", "rete", "proj"],
            evaluator_type="bop",
            thing_classes=data_cfg["objs"],
        )


##################
def register_with_name_cfg(name, data_cfg=None):
    """Assume pre-defined datasets live in `./datasets`.

    Args:
        name: datasnet_name,
        data_cfg: if name is in existing SPLITS, use pre-defined data_cfg
            otherwise requires data_cfg
            data_cfg can be set in cfg.DATA_CFG.name
    """
    dprint("register dataset: {}".format(name))

    if name in SPLITS_DUCK_FRAMES:
        used_cfg = SPLITS_DUCK_FRAMES[name]
    else:
        assert data_cfg is not None, f"dataset name {name} is not registered"
        used_cfg = data_cfg
    DatasetCatalog.register(name, DUCK_FRAMES_Dataset(used_cfg))
    # something like eval_types
    MetadataCatalog.get(name).set(
        ref_key=used_cfg["ref_key"],
        objs=used_cfg["objs"],
        eval_error_types=["ad", "rete", "proj"],
        evaluator_type="bop",
        thing_classes=used_cfg["objs"],
    )


def get_available_datasets():
    names = list(SPLITS_DUCK_FRAMES.keys())
    return names


if __name__ == "__main__":
    from lib.vis_utils.image import grid_show
    from detectron2.utils.logger import setup_logger

    logger = setup_logger(name="core")
    register_duck_frames()

    print("dataset catalog: ", DatasetCatalog.list())
    dset_name = "duck_frames"
    assert dset_name in DatasetCatalog.list()

    meta = MetadataCatalog.get(dset_name)
    dprint("MetadataCatalog: ", meta)

    t_start = time.perf_counter()
    dicts = DatasetCatalog.get(dset_name)
    logger.info("Done loading {} samples with {:.3f}s.".format(len(dicts), time.perf_counter() - t_start))

    dirname = "output/{}-data-vis".format(dset_name)
    os.makedirs(dirname, exist_ok=True)
    for d in dicts:
        img = cv2.imread(d["file_name"], cv2.IMREAD_COLOR).astype("float32") / 255.0
        cv2.imshow("color", img)
        k = cv2.waitKey()
        if k == 27:
            cv2.destroyAllWindows()
            break
