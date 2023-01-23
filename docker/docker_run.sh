#!/bin/bash

# prepare /datasets, /pretrained_models and /output folders as explained in the main README.md
gdrnpp_dir=${PWD%/*}

docker run \
--gpus all \
-it \
--shm-size=8gb --env="DISPLAY" \
--volume="${gdrnpp_dir}:/gdrnpp_bop2022" \
--volume="${gdrnpp_dir}/datasets:/gdrnpp_bop2022/datasets" \
--volume="${gdrnpp_dir}/pretrained_models:/gdrnpp_bop2022/pretrained_models" \
--volume="${gdrnpp_dir}/output:/gdrnpp_bop2022/output" \
--name=gdrnppv0 gdrnpp