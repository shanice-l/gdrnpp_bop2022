# GDRNPP Refinement
The refinement module of GDRNPP.

## Environment
Create a new anaconda environment using the provided .yaml file.
```
conda env create -f environment.yaml
conda activate refine
```

## Preparing Data
Link the datasets in the following folders with symlink.
```Shell
├── local_data
    ├── VOCdevkit
        ├── VOC2012
    ├── bop_datasets
        ├── hb
        ├── icbin
        ├── itodd
        ├── lm
        ├── lmo
        ├── tless
        ├── tudl
        ├── ycbv
```

## Training
We train a model separately for each of the BOP datasets. To train the TUDL refinement model, run
```
python train.py --dataset ycbv --batch_size 12 --num_inner_loops 10 --num_solver_steps 3 [--pbr_only]
```
To train a refinement model on other BOP datasets, replace `ycbv` with one of the following: `tless`, `lmo`, `hb`, `tudl`, `icbin`, `itodd`


## Testing
1. Generate GDRN results with save_gdrn.sh and put the trained RAFT models in folder "model_weights".
2. Run
```
./test_fast.sh <DATASET> <RESULT_PATH> <GPU_ID>
```
Use "test_fast_pbr.sh" for PBR results of YCB-V, T-LESS and TUD-L.
3. Run
```
python -m additional_scripts.convert_result --tar_dir <TAR_DIR> --dataset <DATASET>
```

## Acknowledgement

This repository makes extensive use of code from the [Coupled-Iterative-Refinement](https://github.com/princeton-vl/Coupled-Iterative-Refinement) and [Cosypose](https://github.com/ylabbe/cosypose) Github repository. We thank the authors for open sourcing their implementation.
