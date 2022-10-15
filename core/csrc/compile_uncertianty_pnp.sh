#!/usr/bin/env bash
this_dir=$(dirname "$0")

echo ""
echo "********build uncertainty pnp************"
cd $this_dir/uncertainty_pnp
# cd lib; ln -sf libceres.so.2 libceres.so; cd .. # (Ubuntu 16.04)
# sh build_ceres.sh  # (Ubuntu >= 18.04)
rm -rf build/
python setup.py build_ext --inplace

# test -------------------------------
# . ./scripts/init_env.sh
# ipython
# from core.csrc.uncertainty_pnp.un_pnp_utils import uncertainty_pnp

# trouble shooting -------------------
# ImportError: libspqr.so.2.0.2: cannot open shared object file: No such file or directory
# sudo find / -name 'libspqr.so.*'
# cp /usr/lib/x86_64-linux-gnu/libspqr.so.2.0.8 core/csrc/uncertainty_pnp/lib/libspqr.so.2.0.2

# ImportError: libcholmod.so.3.0.6: cannot open shared object file: No such file or directory
# sudo find / -name 'libcholmod.so.*'
# cp /usr/lib/x86_64-linux-gnu/libcholmod.so.3.0.11 core/csrc/uncertainty_pnp/lib/libcholmod.so.3.0.6

# ImportError: libcxsparse.so.3.1.4: cannot open shared object file: No such file or directory
# sudo find / -name 'libcxsparse.so.*'
# cp /usr/lib/x86_64-linux-gnu/libcxsparse.so.3.2.0 core/csrc/uncertainty_pnp/lib/libcxsparse.so.3.1.4
