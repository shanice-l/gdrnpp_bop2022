#!/bin/bash

# replace file paths of the three volumes with the paths to your /datasets, /pretrained_models and /output folders

docker run \
--gpus all \
-it \
--shm-size=8gb --env="DISPLAY" \
--volume="/home/hoenig/BOP/datasets:/gdrnpp_bop2022/datasets" \
--volume="/home/hoenig/BOP/gdrnets/gdrnpp_bop2022/pretrained_models:/gdrnpp_bop2022/pretrained_models" \
--volume="/home/hoenig/BOP/gdrnets/gdrnpp_bop2022/output:/gdrnpp_bop2022/output" \
--name=gdrnppv0 gdrnpp