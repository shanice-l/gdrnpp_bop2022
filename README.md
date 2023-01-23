# GDRNPP for BOP2022

This repo provides code and models for GDRNPP_BOP2022, **winner (most of the awards) of the BOP Challenge 2022 at ECCV'22 [[slides](http://cmp.felk.cvut.cz/sixd/workshop_2022/slides/bop_challenge_2022_results.pdf)]**.

## Path Setting

### Dataset Preparation
Download the 6D pose datasets from the
[BOP website](https://bop.felk.cvut.cz/datasets/) and
[VOC 2012](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)
for background images.
Please also download the  `test_bboxes` from
here [OneDrive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/liuxy21_mails_tsinghua_edu_cn/Eq_2aCC0RfhNisW8ZezYtIoBGfJiRIZnFxbITuQrJ56DjA?e=hPbJz2) (password: groupji) or [BaiDuYunPan](https://pan.baidu.com/s/1FzTO4Emfu-DxYkNG40EDKw)(password: vp58).

The structure of `datasets` folder should look like below:
```
datasets/
├── BOP_DATASETS   # https://bop.felk.cvut.cz/datasets/
    ├──tudl
    ├──lmo
    ├──ycbv
    ├──icbin
    ├──hb
    ├──itodd
    └──tless
└──VOCdevkit
```


### Models

Download the trained models at [Onedrive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/liuxy21_mails_tsinghua_edu_cn/EgOQzGZn9A5DlaQhgpTtHBwB2Bwyx8qmvLauiHFcJbnGSw?e=EZ60La) (password: groupji) or [BaiDuYunPan](https://pan.baidu.com/s/1LhXblEic6pYf1i6hOm6Otw)(password: 10t3) and put them in the folder `./output`.


## Requirements
* Ubuntu 18.04/20.04, CUDA 10.1/10.2/11.6, python >= 3.7, PyTorch >= 1.9, torchvision
* Install `detectron2` from [source](https://github.com/facebookresearch/detectron2)
* `sh scripts/install_deps.sh`
* Compile the cpp extension for `farthest points sampling (fps)`:
    ```
    sh core/csrc/compile.sh
    ```

## Detection

We adopt yolox as the detection method. We used stronger data augmentation and ranger optimizer.

### Training 

Download the pretrained model at [Onedrive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/liuxy21_mails_tsinghua_edu_cn/EkCTrRfHUZVEtD7eHwLkYSkBCTXlh9ekDteSzK6jM4oo-A?e=m0aNCy) (password: groupji) or [BaiDuYunPan](https://pan.baidu.com/s/1AU7DGCmZWsH9VgQnbTRjow)(password: aw68) and put it in the folder `pretrained_models/yolox`. Then use the following command:

`./det/yolox/tools/train_yolox.sh <config_path> <gpu_ids> (other args)`

### Testing 

`./det/yolox/tools/test_yolox.sh <config_path> <gpu_ids> <ckpt_path> (other args)`

## Pose Estimation

The difference between this repo and GDR-Net (CVPR2021) mainly including:

* Domain Randomization: We used stronger domain randomization operations than the conference version during training.
* Network Architecture: We used a more powerful backbone Convnext rather than resnet-34,  and two  mask heads for predicting amodal mask and visible mask separately.
* Other training details, such as learning rate, weight decay, visible threshold, and bounding box type.

### Training 

`./core/gdrn_modeling/train_gdrn.sh <config_path> <gpu_ids> (other args)`

For example:

`./core/gdrn_modeling/train_gdrn.sh configs/gdrn/ycbv/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv.py 0`

### Testing 

`./core/gdrn_modeling/test_gdrn.sh <config_path> <gpu_ids> <ckpt_path> (other args)`

For example:

`./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbv/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv.py 0 output/gdrn/ycbv/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv/model_final_wo_optim.pth`

## Pose Refinement

We utilize depth information to further refine the estimated pose.
We provide two types of refinement: fast refinement and iterative refinement.

For fast refinement, we compare the rendered object depth and the observed depth to refine translation.
Run

`./core/gdrn_modeling/test_gdrn_depth_refine.sh <config_path> <gpu_ids> <ckpt_path> (other args)`

For iterative refinement, please checkout to the [pose_refine branch](https://github.com/shanice-l/gdrnpp_bop2022/tree/pose_refine) for details.

## Citing GDRNPP

If you use GDRNPP in your research, please use the following BibTeX entries.

```BibTeX
@misc{liu2022gdrnpp_bop,
  author =       {Xingyu Liu and Ruida Zhang and Chenyangguang Zhang and 
                  Bowen Fu and Jiwen Tang and Xiquan Liang and Jingyi Tang and 
                  Xiaotian Cheng and Yukang Zhang and Gu Wang and Xiangyang Ji},
  title =        {GDRNPP},
  howpublished = {\url{https://github.com/shanice-l/gdrnpp_bop2022}},
  year =         {2022}
}

@InProceedings{Wang_2021_GDRN,
    title     = {{GDR-Net}: Geometry-Guided Direct Regression Network for Monocular 6D Object Pose Estimation},
    author    = {Wang, Gu and Manhardt, Fabian and Tombari, Federico and Ji, Xiangyang},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {16611-16621}
}
```

