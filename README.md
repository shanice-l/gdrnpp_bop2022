## GDRNPP for BOP2022

This repo provides code and models for GDRNPP_BOP2022.

TODO: add authors

## Path Setting

### Dataset Preparation
Download the 6D pose datasets from the
[BOP website](https://bop.felk.cvut.cz/datasets/) and
[VOC 2012](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)
for background images.
Please also download the  `test_bboxes` from
here [OneDrive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/liuxy21_mails_tsinghua_edu_cn/Eq_2aCC0RfhNisW8ZezYtIoBGfJiRIZnFxbITuQrJ56DjA?e=hPbJz2) (password: groupji).

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

Download the trained models at [Onedrive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/liuxy21_mails_tsinghua_edu_cn/EgOQzGZn9A5DlaQhgpTtHBwB2Bwyx8qmvLauiHFcJbnGSw?e=EZ60La) (password: groupji) and put them in the folder `./output`.


## Requirements
* Ubuntu 16.04/18.04/20.04, CUDA 10.1/10.2, python >= 3.6, PyTorch >= 1.6, torchvision
* Install `detectron2` from [source](https://github.com/facebookresearch/detectron2)
* `sh scripts/install_deps.sh`
* Compile the cpp extension for `farthest points sampling (fps)`:
    ```
    sh core/csrc/compile.sh
    ```

## Detection

We adopt yolox as the detection method. We used stronger data augmentation and ranger optimizer.

### Training 

Download the pretrained model at [Onedrive](https://mailstsinghuaeducn-my.sharepoint.com/personal/liuxy21_mails_tsinghua_edu_cn/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fliuxy21%5Fmails%5Ftsinghua%5Fedu%5Fcn%2FDocuments%2Fbop%5Fchallenge%5F2022%2Fpretraied%5Fmodels%2Fyolox) (password: groupji) and put it in the folder `pretrained_models/yolox`. Then use the following command:

`./det/yolox/tools/train_yolox.sh <config_path> <gpu_ids> (other args)`

### Testing 

`./det/yolox/tools/test_yolox.sh <config_path> <gpu_ids> <ckpt_path> (other args)`

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
We utilize depth information to further refine the estimated pose.
We provide two types of refinement: fast refinement and iterative refinement.

For fast refinement, we compare the rendered object depth and the observed depth to refine translation.
Run

`./core/gdrn_modeling/test_gdrn_depth_refine.sh <config_path> <gpu_ids> <ckpt_path> (other args)`

For iterative refinement, please checkout to the pose_refine branch for details.
