# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import torch
import numpy as np


def unit(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def lookatnp(eye_3x1, center_3x1, up_3x1):
    # 3 variables should be length 1
    camz = center_3x1 - eye_3x1
    camz /= np.sqrt(np.sum(camz**2))
    camx = np.cross(camz[:, 0], up_3x1[:, 0]).reshape(3, 1)
    camy = np.cross(camx[:, 0], camz[:, 0]).reshape(3, 1)

    # they are not guaranteed to be 1!!!
    mtx = np.concatenate([unit(camx), unit(camy), -camz], axis=1).transpose()
    shift = -(np.matmul(mtx, eye_3x1))
    return mtx, shift


def camera_info(param):
    theta = np.deg2rad(param[0])
    phi = np.deg2rad(param[1])

    camY = param[3] * np.sin(phi)
    temp = param[3] * np.cos(phi)
    camX = temp * np.cos(theta)
    camZ = temp * np.sin(theta)
    cam_pos = np.array([camX, camY, camZ])

    axisZ = cam_pos.copy()
    axisY = np.array([0, 1, 0], dtype=np.float32)
    axisX = np.cross(axisY, axisZ)
    axisY = np.cross(axisZ, axisX)

    # cam_mat = np.array([axisX, axisY, axisZ])
    cam_mat = np.array([unit(axisX), unit(axisY), unit(axisZ)])

    # for verify
    # mtx, shift = lookatnp(cam_pos_3xb.reshape(3, 1), np.zeros(shape=(3, 1), dtype=np.float32), np.array([0,1,0], dtype=np.float32).reshape(3, 1))
    # note, it is different from lookatnp
    # new_p = mtx * old_p + shift
    # new_p = cam_mat * (old_p - cam_pos)

    return cam_mat, cam_pos


#####################################################
def perspectiveprojectionnp(fovy, ratio=1.0, near=0.01, far=10.0):
    """
    fovy: radian, 2 * atan2(h, 2*fy)
    ratio: aspect_ratio, w/h, typically 4/3
    """
    tanfov = np.tan(fovy / 2.0)  # h/(2*fy)
    # top = near * tanfov
    # right = ratio * top
    # mtx = [near / right, 0, 0, 0, \
    #          0, near / top, 0, 0, \
    #          0, 0, -(far+near)/(far-near), -2*far*near/(far-near), \
    #          0, 0, -1, 0]
    mtx = [
        [1.0 / (ratio * tanfov), 0, 0, 0],
        [0, 1.0 / tanfov, 0, 0],
        [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1.0, 0],
    ]
    # return np.array(mtx, dtype=np.float32)
    # 2*fy/h/ratio=2*fy/w, 2*fy/h
    return np.array([[1.0 / (ratio * tanfov)], [1.0 / tanfov], [-1]], dtype=np.float32)


def projectiveprojection_real(cam, x0, y0, w, h, nc=0.01, fc=10.0):
    # this is for center view
    # NOTE: only return a 3x1 vector (diagonal??)
    q = -(fc + nc) / float(fc - nc)
    qn = -2 * (fc * nc) / float(fc - nc)
    fx = cam[0, 0]
    fy = cam[1, 1]
    px = cam[0, 2]
    py = cam[1, 2]
    """
    # transpose: compensate for the flipped image
    proj_T = [
            [2*fx/w,          0,                0,  0],
            [0,               2*fy/h,           0,  0],
            [(-2*px+w+2*x0)/w, (2*py-h+2*y0)/h, q,  -1],
            [0,               0,                qn, 0],
        ]
        sometimes: P[1,:] *= -1, P[2,:] *= -1
        # Third column is standard glPerspective and sets near and far planes
    """
    # Draw our images upside down, so that all the pixel-based coordinate systems are the same
    if isinstance(cam, np.ndarray):
        proj_T = np.zeros((4, 4), dtype=np.float32)
    elif isinstance(cam, torch.Tensor):
        proj_T = torch.zeros(4, 4).to(cam)
    else:
        raise TypeError("cam should be ndarray or tensor, got {}".format(type(cam)))
    proj_T[0, 0] = 2 * fx / w
    proj_T[1, 0] = -2 * cam[0, 1] / w  # =0
    proj_T[1, 1] = 2 * fy / h
    proj_T[2, 0] = (-2 * px + w + 2 * x0) / w
    proj_T[2, 1] = (+2 * py - h + 2 * y0) / h
    proj_T[2, 2] = q
    proj_T[3, 2] = qn
    proj_T[2, 3] = -1.0
    return proj_T


#####################################################
def camera_info_batch(param_bx4):

    bnum = param_bx4.shape[0]
    cam_mat_bx3x3 = []
    cam_pos_bx3 = []

    for i in range(bnum):
        param = param_bx4[i]
        cam_mat, cam_pos = camera_info(param)
        cam_mat_bx3x3.append(cam_mat)
        cam_pos_bx3.append(cam_pos)

    cam_mat_bx3x3 = np.stack(cam_mat_bx3x3, axis=0)
    cam_pos_bx3 = np.stack(cam_pos_bx3, axis=0)

    return cam_mat_bx3x3, cam_pos_bx3
