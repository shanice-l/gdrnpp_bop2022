from typing import Iterable
import json
import os.path as osp
import numpy as np
from tqdm import tqdm
import trimesh
import logging

logging.getLogger("trimesh").setLevel(logging.WARNING)


class Obj:
    def __init__(self, obj_id, mesh: trimesh.Trimesh, diameter: float):
        self.obj_id = obj_id
        self.mesh = mesh
        self.diameter = diameter

        bounding_sphere = self.mesh.bounding_sphere.primitive
        self.offset, self.scale = bounding_sphere.center, bounding_sphere.radius

        self.mesh_norm = mesh.copy()
        self.mesh_norm.apply_translation(-self.offset)
        self.mesh_norm.apply_scale(1 / self.scale)

    def normalize(self, pts: np.ndarray):
        return (pts - self.offset) / self.scale

    def denormalize(self, pts_norm: np.ndarray):
        return pts_norm * self.scale + self.offset


def load_obj(models_root: str, obj_id: int):
    with open(osp.join(models_root, "models_info.json")) as f:
        models_info = json.load(f)
    mesh = trimesh.load_mesh(osp.join(models_root, f"obj_{obj_id:06d}.ply"))
    diameter = models_info[str(obj_id)]["diameter"]
    return Obj(obj_id, mesh, diameter)


def load_objs(models_root: str, obj_ids: Iterable[int] = None, show_progressbar=True):
    objs = []
    if obj_ids is None:
        obj_ids = sorted([int(p.name[4:10]) for p in models_root.glob("*.ply")])
    for obj_id in tqdm(obj_ids, "loading objects") if show_progressbar else obj_ids:
        objs.append(load_obj(models_root, obj_id))
    return objs, obj_ids
