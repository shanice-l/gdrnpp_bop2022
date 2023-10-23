from omegaconf import OmegaConf
from mmengine.config import Config
import yaml
import json

def try_get_key(cfg, *keys, default=None):
    """# modified from detectron2 to also support mmcv Config.

    Try select keys from cfg until the first key that exists. Otherwise
    return default.
    """
    from detectron2.config import CfgNode

    if isinstance(cfg, CfgNode):
        cfg = OmegaConf.create(cfg.dump())
    elif isinstance(cfg, Config):  # mmcv Config
        # print("What's the type of this thing?")
        # # print(type(cfg._cfg_dict).__name__)
        # print("Printing out CFG string to see WTF it is:")
        # cfg_str = str(cfg._cfg_dict).replace("'", '"')
        # print(cfg_str)
        cfg_dict = json.dumps(cfg._cfg_dict.to_dict())
        # cfg_str = yaml.dump(cfg._cfg_dict.to_dict())
        # converted_dict = OmegaConf.to_container(cfg._cfg_dict.to_dict(), resolve=True)
        # cfg = OmegaConf.create(cfg_str)
        cfg = OmegaConf.create(cfg_dict)
    elif isinstance(cfg, dict):  # raw dict
        cfg = OmegaConf.create(cfg)

    for k in keys:
        none = object()
        p = OmegaConf.select(cfg, k, default=none)
        if p is not none:
            return p
    return default
