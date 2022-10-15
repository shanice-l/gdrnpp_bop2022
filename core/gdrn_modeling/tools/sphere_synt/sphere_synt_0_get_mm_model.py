"""scale ply models from m to mm."""
import os.path as osp
import sys
import numpy as np

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../../"))
from lib.pysixd import inout, misc
import ref


ref_key = "sphere_synt"
data_ref = ref.__dict__[ref_key]

model_dir = data_ref.model_dir
id2obj = data_ref.id2obj


def scale_ply(mesh_path, res_mesh_path, transform=None):
    """
    ply: m to mm
    :param mesh_path:
    :return:
    """
    f_res = open(res_mesh_path, "w")
    with open(mesh_path, "r") as f:
        i = 0
        # points = []
        for line in f:
            line = line.strip("\r\n")
            i += 1
            if i <= 18:
                res_line = line + "\n"
            line_list = line.split()

            n_vert_property = 10

            if len(line_list) == n_vert_property:
                xyz = [float(m) * 1000.0 for m in line_list[:3]]
                if transform is not None:
                    R = transform[:3, :3]
                    T = transform[:3, 3]
                    xyz = np.array(xyz)
                    xyz_new = R.dot(xyz.reshape((3, 1))) + T.reshape((3, 1))
                    xyz = xyz_new.reshape((3,))
                for i in range(3):
                    line_list[i] = "{}".format(xyz[i])
                res_line = " ".join(line_list) + "\n"
            else:
                res_line = "{}\n".format(line)

            # print(res_line)
            f_res.write(res_line)


def scale_ply_main():

    mesh_path = osp.join(model_dir, "obj_{:06d}_m.ply".format(1))

    if not osp.exists(mesh_path):
        print("{} not exists!".format(mesh_path))

    res_mesh_path = osp.join(model_dir, "obj_{:06d}.ply".format(1))
    scale_ply(mesh_path, res_mesh_path)

    print("result file: {}".format(res_mesh_path))


# =================================

if __name__ == "__main__":
    # check result with vimdiff
    scale_ply_main()
    print("{} finished".format(__file__))
