## GDRNPP for BOP2022

This repo provides code and models for GDRNPP_BOP2022.

TODO: add authors

## Path Setting

### Dataset Preparation
Download the 6D pose datasets from the
[BOP website](https://bop.felk.cvut.cz/datasets/) and
[VOC 2012](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)
for background images.
Please also download the `image_sets` and `test_bboxes` from
here [OneDrive](#TODO).

The structure of `datasets` folder should look like below:
```
datasets/
├── BOP_DATASETS   # https://bop.felk.cvut.cz/datasets/
    ├──lm
    ├──lmo
    ├──ycbv
    ├──icbin
    ├──hb
    ├──itodd
    ├──tless
└──VOCdevkit
```


### MODELS

Download the trained models at [Onedrive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/liuxy21_mails_tsinghua_edu_cn/EgOQzGZn9A5DlaQhgpTtHBwBGWEB57mpYy4SbmpZJMmMyQ?e=0z9Xd2) and put them in the folder ./output.


## Requirements
* Ubuntu 16.04/18.04, CUDA 10.1/10.2, python >= 3.6, PyTorch >= 1.6, torchvision
* Install `detectron2` from [source](https://github.com/facebookresearch/detectron2)
* `sh scripts/install_deps.sh`
* Compile the cpp extension for `farthest points sampling (fps)`:
    ```
    sh core/csrc/compile.sh
    ```

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

## Pose Refinement
TODO: rudy