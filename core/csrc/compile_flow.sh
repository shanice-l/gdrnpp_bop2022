#!/usr/bin/env bash
this_dir=$(dirname "$0")

echo ""
echo "********build flow************"
cd $this_dir/flow
rm -rf build/
python setup.py build_ext --inplace
