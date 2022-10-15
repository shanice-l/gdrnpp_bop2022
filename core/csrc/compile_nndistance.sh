#!/usr/bin/env bash
this_dir=$(dirname "$0")

echo ""
echo "********build torch_nndistance (chamfer distance)************"
cd $this_dir/torch_nndistance
rm -rf build
python setup.py build_ext --inplace
