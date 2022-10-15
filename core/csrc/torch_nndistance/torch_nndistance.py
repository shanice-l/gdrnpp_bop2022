__version__ = "1.0.0"

import torch
from torch.autograd import Function
import os.path as osp
import sys

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, cur_dir)
import torch_nndistance_aten as _C


class NNDFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        #        ctx.xyz1 = xyz1[...]
        #        ctx.xyz2 = xyz2[...]
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        if not xyz1.is_cuda:
            _C.nnd_forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            dist1 = dist1.cuda()
            dist2 = dist2.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()
            _C.nnd_forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

        #        ctx.dist1 = dist1
        #        ctx.dist2 = dist2

        # print(batchsize, n, m)
        ctx.save_for_backward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        # print(ctx.idx1, ctx.idx2)
        xyz1, xyz2, dist1, dist2, idx1, idx2 = ctx.saved_tensors

        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        if not graddist1.is_cuda:
            _C.nnd_backward(
                xyz1,
                xyz2,
                gradxyz1,
                gradxyz2,
                graddist1,
                graddist2,
                idx1,
                idx2,
            )
        else:
            gradxyz1 = gradxyz1.cuda()
            gradxyz2 = gradxyz2.cuda()
            _C.nnd_backward_cuda(
                xyz1,
                xyz2,
                gradxyz1,
                gradxyz2,
                graddist1,
                graddist2,
                idx1,
                idx2,
            )
        #        print(gradxyz1)
        #        print(gradxyz2)
        #        print(dist1)
        #        print(dist2)
        #        print(idx1)
        #        print(idx2)
        return gradxyz1, gradxyz2


def nnd(xyz1, xyz2):
    return NNDFunction.apply(xyz1, xyz2)
