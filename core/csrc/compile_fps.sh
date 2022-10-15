#!/usr/bin/env bash
# Compile cuda extensions under `csrc`
# assume you are in ROOT/core/csrc
# export CUDA_HOME="/usr/local/cuda"
this_dir=$(dirname "$0")

echo ""
echo "********build fps************"
cd $this_dir/fps
rm -rf build
python setup.py
