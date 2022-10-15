#!/usr/bin/env bash
set -x
this_dir=$(dirname "$0")

cd $this_dir
rm -rf build
python setup.py build_ext --inplace


# trouble shooting:
# Sometimes the soft symbol links of libGL.so libEGL.so might be broken,
# fix them and compile again should work.
