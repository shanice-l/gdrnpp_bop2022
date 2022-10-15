#!/usr/bin/env bash

this_dir=$(dirname "$0")

echo ""
echo "********build fps************"
cd $this_dir/../core/csrc/fps/
rm -rf build
python setup.py


echo ""
echo "********build flow************"
cd ../flow/
rm -rf build/
python setup.py build_ext --inplace


echo ""
echo "********build ransac_voting************"
cd ../ransac_voting
rm -rf build
python setup.py build_ext --inplace


echo ""
echo "********build uncertainty pnp************"
cd ../uncertainty_pnp
# sh build_ceres.sh
rm -rf build/
python setup.py build_ext --inplace


echo ""
echo "********build torch_nndistance (chamfer distance)************"
cd ../torch_nndistance
rm -rf build
python setup.py build_ext --inplace


echo ""
echo "********build cpp egl renderer************"
cd ../../../lib/egl_renderer/
rm -rf build/
python setup.py build_ext --inplace
