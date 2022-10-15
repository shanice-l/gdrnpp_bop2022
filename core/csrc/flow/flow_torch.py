# https://github.com/huyaoyu/ImageFlow/blob/master/OpticalFlow.py
# https://github.com/liyi14/mx-DeepIM/blob/master/lib/pair_matching/flow.py
# https://github.com/daniilidis-group/mvsec/blob/master/tools/gt_flow/compute_flow.py
# https://github.com/jmccormac/pySceneNetRGBD/blob/master/calculate_optical_flow.py
from torch.autograd import Function
import os.path as osp
import sys
from core.utils.pose_utils import calc_se3_torch_batch

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, cur_dir)
import flow_cuda


class FlowFunction(Function):
    @staticmethod
    def forward(ctx, depth_src, depth_tgt, pose_src, pose_tgt, K):
        """
        Args:
            depth_src: Bx1xHxW
            depth_tgt: Bx1xHxW
            pose_src: Bx3x4
            pose_tgt: Bx3x4
            K: Bx3x3
        Returns:
            flow: Bx2xHxW
            valid: Bx1xHxW
        """
        # KT: K x se3, Bx3x4
        # Kinv: Bx3x3
        se3_ren2obs = calc_se3_torch_batch(pose_src, pose_tgt)
        KT = K @ se3_ren2obs
        Kinv = K.inverse().contiguous()
        outputs = flow_cuda.forward(depth_src, depth_tgt, KT, Kinv)
        flow = outputs[0]
        valid = outputs[1]
        return flow, valid


flow = FlowFunction.apply
