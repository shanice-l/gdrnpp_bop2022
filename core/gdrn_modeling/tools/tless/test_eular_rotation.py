import os.path as osp
import sys
from tqdm import tqdm
import math
import numpy as np
import random
import mmcv

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../../"))

from lib.pysixd import inout, misc, transform
import ref
from lib.utils.utils import dprint


def my_rand(a, b):
    return a + (b - a) * random.random()


def random_eular():
    range = 1
    # use eular formulation, three different rotation angles on 3 axis
    phi = my_rand(0, range * math.pi * 2)
    theta = my_rand(0, range * math.pi)
    psi = my_rand(0, range * math.pi * 2)

    return phi, theta, psi


def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0], [0, np.cos(theta[0]), -np.sin(theta[0])], [0, np.sin(theta[0]), np.cos(theta[0])]])

    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])], [0, 1, 0], [-np.sin(theta[1]), 0, np.cos(theta[1])]])

    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0], [np.sin(theta[2]), np.cos(theta[2]), 0], [0, 0, 1]])

    # R = np.dot(R_z, np.dot(R_y, R_x))
    R = R_x @ R_y @ R_z
    return R


def eular_to_mat(phi, theta, psi):
    R0 = []
    R0.append(math.cos(psi) * math.cos(phi) - math.cos(theta) * math.sin(phi) * math.sin(psi))
    R0.append(math.cos(psi) * math.sin(phi) + math.cos(theta) * math.cos(phi) * math.sin(psi))
    R0.append(math.sin(psi) * math.sin(theta))

    R1 = []
    R1.append(-(math.sin(psi)) * math.cos(phi) - math.cos(theta) * math.sin(phi) * math.cos(psi))
    R1.append(-(math.sin(psi)) * math.sin(phi) + math.cos(theta) * math.cos(phi) * math.cos(psi))
    R1.append(math.cos(psi) * math.sin(theta))

    R2 = []
    R2.append(math.sin(theta) * math.sin(phi))
    R2.append(-(math.sin(theta)) * math.cos(phi))
    R2.append(math.cos(theta))

    R = []
    R.append(R0)
    R.append(R1)
    R.append(R2)
    return np.array(R)


if __name__ == "__main__":
    # phi, theta, psi = random_eular()
    phi = theta = 0
    psi = math.pi / 2

    R_raw = eulerAnglesToRotationMatrix([phi, theta, psi])
    R = eulerAnglesToRotationMatrix([phi + math.pi, theta + math.pi, psi + math.pi])

    print("raw_R\n", R_raw)
    print("rotated_R\n", R)
