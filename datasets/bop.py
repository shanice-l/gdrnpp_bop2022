import json
import time
import warnings
from pathlib import Path

import gin
import imageio
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import interpolate
from tqdm import tqdm
from utils import MEMORY, CropResizeToAspectAugmentation
from utils.augmentation import *

from .bop_object_datasets import BOPObjectDataset
from .symmetries import make_bop_symmetries, make_se3

LD2DL = lambda LD: {k: [dic[k] for dic in LD] for k in LD[0]}

@MEMORY.cache(verbose=1)
def build_index(ds_dir, split, check_image_exists=False):
    scene_ids, cam_ids, view_ids = [], [], []

    annotations = dict()
    base_dir = ds_dir / split

    for scene_dir in tqdm(sorted(base_dir.iterdir()), desc="Running build_index"):
        scene_id = scene_dir.name
        if not scene_id.isdigit():
            continue
        annotations_scene = dict()
        for f in ('scene_camera.json', 'scene_gt_info.json', 'scene_gt.json'):
            path = (scene_dir / f)
            if path.exists():
                annotations_scene[f.split('.')[0]] = json.loads(path.read_text())
        annotations[scene_id] = annotations_scene

        if check_image_exists:
            rgb_dir = scene_dir / 'rgb'
            if not rgb_dir.exists():
                rgb_dir = scene_dir / 'gray'
            image_files = {img.stem for img in rgb_dir.glob("*")}

        for view_id in annotations_scene['scene_camera'].keys():
            view_id = int(view_id)
            view_id_str = f'{view_id:06d}'
            if check_image_exists and view_id_str not in image_files:
                continue

            cam_id = 'cam'
            scene_ids.append(int(scene_id))
            cam_ids.append(cam_id)
            view_ids.append(int(view_id))

    frame_index = pd.DataFrame({'scene_id': scene_ids, 'cam_id': cam_ids,
                                'view_id': view_ids, 'cam_name': cam_ids})
    return frame_index, annotations


def remap_bop_targets(targets):
    targets = targets.rename(columns={'im_id': 'view_id'})
    targets['label'] = targets['obj_id'].apply(lambda x: f'obj_{x:06d}')
    return targets

def keep_bop19(frame_index, ds_dir):
    targets = pd.read_json(ds_dir / 'test_targets_bop19.json')
    targets = remap_bop_targets(targets)
    targets = targets.loc[:, ['scene_id', 'view_id']].drop_duplicates()
    index = frame_index.merge(targets, on=['scene_id', 'view_id']).reset_index(drop=True)
    assert len(index) == len(targets)
    return index

def lin_interp(depth: np.ndarray) -> np.ndarray:
    x, y = np.meshgrid(np.arange(depth.shape[1]).astype("float32"),
                       np.arange(depth.shape[0]).astype("float32"))
    xx = x[depth > 0]
    yy = y[depth > 0]
    zz = depth[depth > 0]

    grid = interpolate.griddata((xx, yy), zz.ravel(),
                                (x, y), method='nearest')
    return grid

def find_image(filepath: Path):
    for suf in {'.png', '.tif', '.jpg'}:
        if filepath.with_suffix(suf).exists():
            return filepath.with_suffix(suf)
    raise FileNotFoundError(f"{filepath} doesn't exist with any suffix.")

@gin.configurable
class BOPDataset:

    frame_index = {}
    annotations = {}

    def __init__(self, ds_dir, split, use_augmentation=False, only_bop19_test=False, use_pre_interpolated_depth=False, load_depth=True, check_image_exists=False, visib_fract_minimum=-1.0):
        self.ds_dir = ds_dir = Path(ds_dir)
        assert ds_dir.exists(), f'Dataset does not exists: {ds_dir}'

        self.split = split
        self.base_dir = ds_dir / split
        self.use_pre_interpolated_depth = use_pre_interpolated_depth
        self.visib_fract_minimum = visib_fract_minimum

        if (ds_dir, split) in BOPDataset.frame_index:
            self.frame_index = BOPDataset.frame_index[(ds_dir, split)]
            self.annotations = BOPDataset.annotations[(ds_dir, split)]
        else:
            save_file_index_feather, save_file_annotations_bytes = build_index(ds_dir=ds_dir, split=split, check_image_exists=check_image_exists)
            self.frame_index = save_file_index_feather.reset_index(drop=True)

            if only_bop19_test:
                self.frame_index = keep_bop19(self.frame_index, ds_dir)

            self.annotations = save_file_annotations_bytes
            BOPDataset.frame_index[(ds_dir, split)] = self.frame_index
            BOPDataset.annotations[(ds_dir, split)] = self.annotations

        models_infos = json.loads((ds_dir / 'models' / 'models_info.json').read_text())
        self.all_labels = [f'obj_{int(obj_id):06d}' for obj_id in models_infos.keys()]
        self.use_augmentation = use_augmentation
        self.load_depth = load_depth

        self.obj_ds = BOPObjectDataset()
        for obj_key, obj_infos in self.obj_ds.objects.items():
            dict_symmetries = {k: obj_infos.get(k, []) for k in ('symmetries_discrete', 'symmetries_continuous')}
            self.obj_ds.objects[obj_key]['symmetries'] = make_bop_symmetries(dict_symmetries, scale=obj_infos['scale'])

        self.background_augmentations = VOCBackgroundAugmentation(voc_root='local_data/VOCdevkit/VOC2012', p=0.3)
        self.rgb_augmentations = [
            PillowBlur(p=0.4, factor_interval=(1, 3)),
            PillowSharpness(p=0.3, factor_interval=(0., 50.)),
            PillowContrast(p=0.3, factor_interval=(0.2, 50.)),
            PillowBrightness(p=0.5, factor_interval=(0.1, 6.0)),
            PillowColor(p=0.3, factor_interval=(0., 20.)),
        ]
        self.resize_augmentation = CropResizeToAspectAugmentation()

    def __len__(self):
        return len(self.frame_index)

    def __getitem__(self, frame_id):
        assert type(frame_id) is not torch.Tensor
        row = self.frame_index.iloc[frame_id]
        scene_id, view_id = row.scene_id, row.view_id
        view_id = int(view_id)
        view_id_str = f'{view_id:06d}'
        scene_id_str = f'{int(scene_id):06d}'
        scene_dir = self.base_dir / scene_id_str

        rgb_dir = scene_dir / 'rgb'
        if not rgb_dir.exists():
            rgb_dir = scene_dir / 'gray'
        rgb_path = find_image(rgb_dir / view_id_str)

        rgb = np.array(Image.open(rgb_path))
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=-1)
        rgb = rgb[..., :3]
        h, w = rgb.shape[:2]
        rgb = torch.as_tensor(rgb)

        cam_annotation = self.annotations[scene_id_str]['scene_camera'][str(view_id)]
        if 'cam_R_w2c' in cam_annotation:
            RC0 = np.array(cam_annotation['cam_R_w2c']).reshape(3, 3)
            tC0 = np.array(cam_annotation['cam_t_w2c']) * 0.001
            TC0 = make_se3(RC0, tC0)
        else:
            TC0 = make_se3(np.eye(3), np.zeros(3))
        K = np.array(cam_annotation['cam_K']).reshape(3, 3).astype(np.float32)
        T0C = TC0.inv()
        T0C = T0C.matrix().float().numpy()
        camera = dict(T0C=T0C, K=K, TWC=T0C, resolution=rgb.shape[:2])

        T0C = TC0.inv()

        objects = []
        mask = np.zeros((h, w), dtype=np.uint8)
        if 'scene_gt_info' in self.annotations[scene_id_str]:
            annotation = self.annotations[scene_id_str]['scene_gt'][str(view_id)]
            n_objects = len(annotation)
            visib = self.annotations[scene_id_str]['scene_gt_info'][str(view_id)]
            for n in range(n_objects):
                if visib[n]['visib_fract'] < self.visib_fract_minimum:
                    continue
                RCO = np.array(annotation[n]['cam_R_m2c']).reshape(3, 3)
                tCO = np.array(annotation[n]['cam_t_m2c']) * 0.001
                TCO = make_se3(RCO, tCO)
                T0O = T0C * TCO
                T0O = T0O.matrix().float().numpy()
                obj_id = annotation[n]['obj_id']
                name = f'obj_{int(obj_id):06d}'
                obj_infos = self.obj_ds[name]                
                assert obj_infos['label'] == name, [obj_infos['label'], name]
                bbox_visib = np.array(visib[n]['bbox_visib'])
                x, y, w, h = bbox_visib
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                obj = dict(label=name, name=name, TWO=T0O, T0O=T0O, diameter_m=obj_infos['diameter_m'],
                           visib_fract=visib[n]['visib_fract'], symmetries=obj_infos['symmetries'],
                           id_in_segm=n+1, bbox=[x1, y1, x2, y2])
                objects.append(obj)

            if 'test' in self.split:
                objects.clear()
                n_objects = 0
            elif len(objects) == 0:
                warnings.warn(f"No visible objects at index {frame_id}. Picking a new dataset index.")
                new_index = np.random.default_rng(frame_id).integers(len(self))
                return self[new_index]

            mask_path = scene_dir / 'mask_visib' / f'{view_id_str}_all.png'
            if mask_path.exists():
                mask = np.array(Image.open(mask_path))
            else:
                for n in range(n_objects):
                    mask_n = np.array(Image.open(scene_dir / 'mask_visib' / f'{view_id_str}_{n:06d}.png'))
                    mask[mask_n == 255] = n + 1

        visible_objects = set(np.unique(mask))
        objects = [o for o in objects if (o['id_in_segm'] in visible_objects)]

        mask = torch.as_tensor(mask)

        if self.load_depth:
            depth_path = find_image(scene_dir / ('interpolated_depth' if self.use_pre_interpolated_depth else 'depth') / view_id_str)
            depth = np.array(imageio.imread(depth_path).astype(np.float32))
            camera['interpolated_depth'] = depth * cam_annotation['depth_scale'] / 1000
            if not self.use_pre_interpolated_depth:
                camera['interpolated_depth'] = lin_interp(camera['interpolated_depth'])
        else:
            camera['interpolated_depth'] = np.zeros((480, 640), dtype=np.float32)

        obs = dict(
            objects=objects,
            camera=camera,
            frame_info=row.to_dict(),
        )
        rgb, mask, camera['interpolated_depth'], obs = self.resize_augmentation(rgb, mask, camera['interpolated_depth'], obs)
        camera['rgb_no_aug'] = rgb.numpy()

        if self.use_augmentation:
            rgb = self.background_augmentations(rgb, mask)
            camera['rgb_no_aug'] = rgb.numpy()
            for augmentation in self.rgb_augmentations:
                rgb = augmentation(rgb)
            rgb = torch.as_tensor(np.array(rgb))

        return rgb, mask, obs

def collate_fn(data):
    *tensors, obs_list = zip(*data)
    obs_list = LD2DL(obs_list)
    obs_list['frame_info'] = LD2DL(obs_list['frame_info'])
    obs_list['camera'] = LD2DL(obs_list['camera'])
    for k,v in obs_list['camera'].items():
        obs_list['camera'][k] = torch.as_tensor(np.array(v))
    return *map(torch.stack, tensors), obs_list
