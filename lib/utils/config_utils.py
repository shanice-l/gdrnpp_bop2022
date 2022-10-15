from omegaconf import OmegaConf
from mmcv import Config


def try_get_key(cfg, *keys, default=None):
    """# modified from detectron2 to also support mmcv Config.

    Try select keys from cfg until the first key that exists. Otherwise
    return default.
    """
    from detectron2.config import CfgNode

    if isinstance(cfg, CfgNode):
        cfg = OmegaConf.create(cfg.dump())
    elif isinstance(cfg, Config):  # mmcv Config
        cfg = OmegaConf.create(cfg._cfg_dict.to_dict())
    elif isinstance(cfg, dict):  # raw dict
        cfg = OmegaConf.create(cfg)

    for k in keys:
        none = object()
        p = OmegaConf.select(cfg, k, default=none)
        if p is not none:
            return p
    return default
