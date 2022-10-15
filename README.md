## GDRNPP for BOP2022

This repo provides code and models for GDRNPP.

## Path setting

```
# recommend using soft links (ln -sf)
datasets/
├── BOP_DATASETS   # https://bop.felk.cvut.cz/datasets/
    ├──lm
    ├──lmo
    ├──ycbv
    ├──icbin
    ├──hb
    ├──itodd
    ├──tless
├── VOCdevkit
└── coco
```

## Dependencies
See [INSTALL.md](./docs/INSTALL.md)

## Detection
TODO: tjw

## Pose Estimation

The difference between this repo and gdrn conference version mainly including:

* Domain Randomization: We used stronger domain randomization operations than the conference version during training.
* Network Architecture: We used a more powerful backbone Convnext rather than resnet-34,  and two  mask heads for predicting amodal mask and visible mask separately.
* Other training details, such as learning rate, weight decay, visible threshold, and bounding box type.

### Training 

`./core/gdrn_modeling/train_gdrn.sh <config_path> <gpu_ids> (other args)`

### Testing 

`./core/gdrn_modeling/test_gdrn.sh <config_path> <gpu_ids> <ckpt_path> (other args)`

The trained models can be found at [Onedrive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/liuxy21_mails_tsinghua_edu_cn/EgOQzGZn9A5DlaQhgpTtHBwBGWEB57mpYy4SbmpZJMmMyQ?e=0z9Xd2)


## Pose Refinement
TODO: rudy