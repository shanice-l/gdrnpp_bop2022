import json
from collections import OrderedDict
from pathlib import Path

import gin


@gin.configurable
class BOPObjectDataset:
    def __init__(self, ds_dir):
        ds_dir = Path(ds_dir) / "models"
        infos_file = ds_dir / 'models_info.json'
        infos = json.loads(infos_file.read_text())
        objects = OrderedDict()
        for obj_id, bop_info in infos.items():
            obj_id = int(obj_id)
            obj_label = f'obj_{obj_id:06d}'
            mesh_path = (ds_dir / obj_label).with_suffix('.ply').as_posix()
            obj = dict(
                label=obj_label,
                category=None,
                mesh_path=mesh_path,
                mesh_units='mm',
            )
            is_symmetric = False
            for k in ('symmetries_discrete', 'symmetries_continuous'):
                obj[k] = bop_info.get(k, [])
                if len(obj[k]) > 0:
                    is_symmetric = True
            obj['is_symmetric'] = is_symmetric
            obj['diameter'] = bop_info['diameter']
            scale = 0.001 if obj['mesh_units'] == 'mm' else 1.0
            obj['scale'] = scale
            obj['diameter_m'] = bop_info['diameter'] * scale
            objects[obj_label] = obj

        self.objects = objects
        self.ds_dir = ds_dir

    def __getitem__(self, idx):
        return self.objects[idx]

    def __len__(self):
        return len(self.objects)
