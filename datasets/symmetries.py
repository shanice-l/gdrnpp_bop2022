import gin
import numpy as np
import torch
import transforms3d
from lietorch import SE3
from scipy.spatial.transform import Rotation as R
from utils import mat2SE3


def euler2quat(xyz, axes='sxyz'):
    """
    euler: sxyz
    quaternion: xyzw
    """
    wxyz = transforms3d.euler.euler2quat(*xyz, axes=axes)
    xyzw = [*wxyz[1:], wxyz[0]]
    return np.array(xyzw)

def make_se3(r, t):
    r_vec = R.from_matrix(r).as_quat()
    vec = torch.from_numpy(np.concatenate((t, r_vec)))
    return SE3.InitFromVec(vec)

@gin.configurable
def make_bop_symmetries(dict_symmetries, n_symmetries_continuous, scale, n_total_symmetries):
    # Note: See https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/misc.py
    sym_discrete = dict_symmetries.get('symmetries_discrete', [])
    sym_continous = dict_symmetries.get('symmetries_continuous', [])
    all_M_discrete = [SE3.Identity(1, dtype=torch.double)[0]]
    all_M_continuous = []
    all_M = []
    for sym_n in sym_discrete:
        M = np.array(sym_n).reshape(4, 4)
        M[:3, -1] *= scale
        M = mat2SE3(torch.as_tensor(M))
        all_M_discrete.append(M)
    for sym_n in sym_continous:
        assert np.allclose(sym_n['offset'], 0)
        axis = np.array(sym_n['axis'])
        assert axis.sum() == 1
        for n in range(n_symmetries_continuous):
            euler = axis * 2 * np.pi * n / n_symmetries_continuous
            q = torch.as_tensor(euler2quat(euler), dtype=torch.double)
            M = SE3.InitFromVec(torch.cat((torch.zeros(3), q)))
            all_M_continuous.append(M)
    for sym_d in all_M_discrete:
        if len(all_M_continuous) > 0:
            for sym_c in all_M_continuous:
                M = sym_c * sym_d
                all_M.append(M.matrix().numpy())
        else:
            all_M.append(sym_d.matrix().numpy())
    output = np.tile(np.eye(4, dtype=np.float32), (n_total_symmetries, 1, 1))
    output[:len(all_M)] = np.array(all_M)
    return output
