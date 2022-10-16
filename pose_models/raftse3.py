import gin
import torch
import torch.nn as nn
from fastcore.all import store_attr
from torch_scatter import scatter_mean
from utils import get_ind, mat2SE3, vectorize_intrinsics
from utils.geom import projective_ops_rgbd as pops
from utils.geom.ba import BA, MoBA
from utils.geom.sampler_utils import bilinear_sampler, sample_depths

from pose_models.modules.clipping import GradientClip
from pose_models.modules.corr import CorrBlock
from pose_models.modules.extractor import BasicEncoder
from pose_models.modules.gru import ConvGRU

"""
Predicts damping factor for Jz^T * Jz
conditioned on the current hidden state
(only used in Modified BD-PnP)
"""
class GraphAgg(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu = nn.ReLU()

        self.eta = nn.Sequential(
            nn.Conv2d(128, 1, 3, padding=1),
            GradientClip(),
            nn.Softplus())

    def forward(self, net, ii):
        batch, num, ch, ht, wd = net.shape
        net = net.view(batch*num, ch, ht, wd)

        _, ix = torch.unique(ii, return_inverse=True)
        net = self.relu(self.conv1(net))

        net = net.view(batch, num, 128, ht, wd)
        net = scatter_mean(net, ix, dim=1)
        net = net.view(-1, 128, ht, wd)

        net = self.relu(self.conv2(net))

        eta = self.eta(net).view(batch, -1, ht, wd)

        return .01 * eta

class UpdateModule(nn.Module):
    def __init__(self, hdim: int, cdim: int, cor_planes):
        super().__init__()

        # Input extractors
        self.corr_encoder = nn.Sequential(
            nn.Conv2d(cor_planes, 128, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            GradientClip(),
            nn.ReLU())

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(3, 128, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            GradientClip(),
            nn.ReLU())

        self.dz_encoder = nn.Sequential(
            nn.Conv2d(1, 128, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            GradientClip(),
            nn.ReLU())

        # Output heads
        self.weight = nn.Sequential(
            nn.Conv2d(hdim, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, padding=1),
            GradientClip())

        self.delta = nn.Sequential(
            nn.Conv2d(hdim, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, padding=1),
            GradientClip())

        self.gru = ConvGRU(hdim, cdim + 128 + 64 + 64)

    def forward(self, net, inp, corr, resid, dz=None):
        batch, num, _, ht, wd = net.shape
        output_dim = (batch, num, -1, ht, wd)
        net = net.view(batch*num, -1, ht, wd)
        inp = inp.view(batch*num, -1, ht, wd)
        corr = corr.view(batch*num, -1, ht, wd)
        resid = resid.view(batch*num, -1, ht, wd)

        corr = self.corr_encoder(corr)
        resid = self.flow_encoder(resid)
        if hasattr(self, "dz_encoder"):
            net = self.gru(net, inp, corr, resid, self.dz_encoder(dz.view(batch*num, -1, ht, wd)))
        else:
            net = self.gru(net, inp, corr, resid)

        ### update variables ###
        delta = self.delta(net).view(*output_dim)
        weight = self.weight(net).view(*output_dim)

        delta = delta.permute(0,1,3,4,2).contiguous()
        weight = weight.permute(0,1,3,4,2).contiguous()

        net = net.view(*output_dim)
        return net, delta, weight

@gin.configurable
class RaftSe3(nn.Module):
    def __init__(self, hdim, cdim, fdim, solver_method):
        super().__init__()
        store_attr()
        self.fnet = BasicEncoder(input_dim=3, output_dim=fdim, norm_fn='instance', dres=4)
        self.cnet = BasicEncoder(input_dim=6, output_dim=hdim+cdim, norm_fn='none', dres=4)
        cor_planes = 4 * (2*3 + 1)**2
        if solver_method == "Modified BD-PnP":
            self.agg = GraphAgg()
            cor_planes += 2
        self.update = UpdateModule(hdim=hdim, cdim=cdim, cor_planes=cor_planes)

    def run_extractor(self, images):
        _, K, *_ = images.shape
        ii, jj = get_ind(K-1, images.device)
        context_output = self.cnet(torch.cat((images[:, ii], images[:, jj]), dim=2))
        net, inp = context_output.split([self.hdim, self.cdim], dim=2)
        net = torch.tanh(net.float())
        inp = torch.relu(inp.float())
        fmaps = self.fnet(images).float()
        corr_fn = CorrBlock(fmaps, (ii, jj))
        return corr_fn, net, inp

    @gin.configurable(module='raft_model')
    def forward(self, Gs, images, depths_fullres, masks_fullres,
    intrinsics_mat, labels, num_solver_steps, num_inner_loops, renderer):

        """
        The variable Gs is the object pose in world-coordinates,
        unlike "G" in the paper which is in camera-coordinates.
        Also, Gs[-1] is the pose of the object in the input image,
        and Gs[:-1] are the poses in the renders.
        """
        Gs = mat2SE3(Gs)

        DEVICE = depths_fullres.device
        B, N, full_H, full_W = depths_fullres.shape
        N -= 1
        intrinsics = vectorize_intrinsics(intrinsics_mat) / 4

        ii, jj = get_ind(N, DEVICE)

        depths = depths_fullres[:, :, 1::4, 1::4].clamp(min=1e-3)
        masks_lowres = masks_fullres[:, :, 1::4, 1::4]
        res_rep = torch.tensor((full_H, full_W), device=DEVICE).tile(B, 1)

        corr_fn, net, inp = self.run_extractor(images)

        # Syncing depth with current pose
        mask_lowres_current = masks_lowres.clone()
        if self.solver_method == "Modified BD-PnP":
            """We don't have depth in the input image, so initialize it
            using the depth of the current pose estimate (1st render)"""
            estimated_depth = depths[:,0]
            depths[:, -1] = estimated_depth.clamp(min=1e-3)

        coords1_xyz, valid_depth = pops.projective_transform(Gs, depths, intrinsics, ii, jj)
        mask_lowres_current[:, ii] = mask_lowres_current[:, ii] & valid_depth.bool()
        residual = torch.zeros_like(coords1_xyz)
        target = torch.zeros_like(coords1_xyz)

        output_lists = {"Gs": [Gs.data], "coords": [target], "depths": [depths], "masks": [mask_lowres_current],
        "weights": [torch.ones_like(target)], "resids": [torch.zeros_like(target)]}
        for step in range(num_inner_loops):
            Gs = Gs.detach()
            coords1_xyz = coords1_xyz.detach()
            residual = residual.detach()

            """Sample the correlation pyramid"""
            coords1, zinv_proj = coords1_xyz.split([2,1], dim=-1)
            corr = corr_fn(coords1)

            zinv = sample_depths(1.0/depths[:,jj], coords1)
            dz = (zinv - zinv_proj).clamp(-1.0, 1.0)
            if self.solver_method == "Modified BD-PnP":
                """The current depth estimate is constantly changing.
                These features may be useful for the update operator"""
                corr = torch.cat((corr, depths[:, ii, None], depths[:, jj, None]), dim=2)

            """Predict update to the hidden state, correspondence revisions, and new confidence weights"""
            net, delta, weight_logits = self.update(net, inp, corr, residual, dz)

            """Apply the correspondence revisions"""
            target = coords1_xyz + delta

            """Zero the weights for correspondences that we know must be incorrect"""
            weight = weight_logits.sigmoid() * mask_lowres_current[:, ii].unsqueeze(4)
            target_scaled = target.mul(4)[..., :2]
            sampled_mask, valid_mask_pixels = bilinear_sampler(masks_fullres[:, jj].unsqueeze(2).float(), target_scaled, use_mask=True)
            weight = weight * (sampled_mask * valid_mask_pixels).unsqueeze(-1)

            """Perform the pose update"""
            if self.solver_method == "Modified BD-PnP":
                eta = self.agg(net, ii)[:, [-1]]
                disps = torch.where(depths > 0.2, 1/depths, torch.full_like(depths, 100))
                Gs, _ = BA(target, weight, eta, Gs, disps, intrinsics, ii, jj)
            elif self.solver_method == "BD-PnP":
                Gs, _, _ = MoBA(target, weight, Gs, depths, intrinsics, num_solver_steps, ii, jj)

            if self.training or step == num_inner_loops - 1:
                output_lists["Gs"].append(Gs.data)
                output_lists["depths"].append(depths)
                output_lists["coords"].append(target)
                output_lists["resids"].append(residual)
                output_lists["masks"].append(mask_lowres_current)
                output_lists["weights"].append(weight_logits)

            # Syncing depth with current pose
            mask_lowres_current = masks_lowres.clone()
            if self.solver_method == "Modified BD-PnP":
                """Generate depth for the input image using the current pose estimate"""
                with torch.no_grad():
                    _, estimated_depth, _ = renderer(labels, Gs[:,-1].matrix(), intrinsics_mat[:,-1], res_rep, scale_res=(1/4))
                depths[:, -1] = estimated_depth.clamp(min=1e-3)

            coords1_xyz, valid_depth = pops.projective_transform(Gs, depths, intrinsics, ii, jj)
            mask_lowres_current[:, ii] = mask_lowres_current[:, ii] & valid_depth.bool()
            residual = target - coords1_xyz

        return output_lists
