import sys
from pathlib import Path

import gin
import torch
import yaml

from .detector import Detector
from .mask_rcnn import DetectorMaskRCNN


def create_model_detector(cfg, n_classes):
    model = DetectorMaskRCNN(input_resize=cfg.input_resize,
                             n_classes=n_classes,
                             backbone_str=cfg.backbone_str,
                             anchor_sizes=cfg.anchor_sizes)
    return model

@gin.configurable()
@torch.no_grad()
def load_detector(run_id):
    EXP_DIR = Path("model_weights") / 'detector'
    run_dir = EXP_DIR / run_id
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.UnsafeLoader)
    label_to_category_id = cfg.label_to_category_id
    model = create_model_detector(cfg, len(label_to_category_id))
    ckpt = torch.load(run_dir / 'checkpoint.pth.tar', map_location=torch.device('cpu'))
    ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    try:
        model = model.cuda().eval()
        model.cfg = cfg
        model.config = cfg
        model = Detector(model)
    except RuntimeError:
        print("Memory error when loading the detector")
        sys.exit(8) # Custom error code for this specific case
    print(f"Loaded MaskRCNN model: {run_dir / 'checkpoint.pth.tar'}")
    return model
