#!/usr/bin/env bash
set -x
VERSION=1.14.0

mkdir -p ceres
cd ceres
wget http://ceres-solver.org/ceres-solver-$VERSION.tar.gz
tar xvzf ceres-solver-$VERSION.tar.gz
cd ceres-solver-$VERSION
sed -i 's/\(^option(BUILD_SHARED_LIBS.*\)OFF/\1ON/' CMakeLists.txt
rm -rf -v build
mkdir build
cd build
cmake ..
make -j8
cd ../../../
mv -v ceres/ceres-solver-$VERSION/build/lib/libceres* ./lib/
