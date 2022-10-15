#!/usr/bin/env bash
this_dir=$(dirname "$0")

echo ""
echo "********build ransac voting************"
cd $this_dir/ransac_voting
rm -rf build/
python setup.py build_ext --inplace
