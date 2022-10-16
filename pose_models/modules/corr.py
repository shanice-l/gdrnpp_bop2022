import sys
import warnings

import gin
import torch
import torch.nn.functional as F
from utils.geom.sampler_utils import bilinear_sampler

try:
    import corr_sampler
except:
    pass # corr_sampler not compiled

class CorrSampler(torch.autograd.Function):

    @staticmethod
    def forward(ctx, volume, coords, radius):
        ctx.save_for_backward(volume,coords)
        ctx.radius = radius
        corr, = corr_sampler.forward(volume, coords, radius)
        return corr

    @staticmethod
    def backward(ctx, grad_output):
        volume, coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_volume, = corr_sampler.backward(volume, coords, grad_output, ctx.radius)
        return grad_volume, None, None


@gin.configurable
class CorrBlock:
    def __init__(self, fmaps, inds, num_levels, radius):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmaps, inds)

        batch, num, h1, w1, h2, w2 = corr.shape
        corr = corr.reshape(batch*num*h1*w1, 1, h2, w2)
        
        for i in range(self.num_levels):
            self.corr_pyramid.append(
                corr.view(batch*num, h1, w1, h2//2**i, w2//2**i))
            corr = F.avg_pool2d(corr, 2, stride=2)
            
    def __call__(self, coords):
        r = self.radius
        batch, num, ht, wd, _ = coords.shape

        try:
            assert "corr_sampler" in sys.modules # Will go to 'except' if this fails
            out_pyramid = []
            coords_reshaped = coords.permute(0,1,4,2,3)
            coords_reshaped = coords_reshaped.contiguous().view(batch*num, 2, ht, wd)
            for i in range(self.num_levels):
                corr = CorrSampler.apply(self.corr_pyramid[i], coords_reshaped/2**i, r)
                out_pyramid.append(corr.view(batch, num, -1, ht, wd))
        except:
            out_pyramid = []
            warnings.warn("'corr_sampler' not compiled. Defaulting to the Pytorch correlation sampler implementation, which is a bit slower. Follow the intructions at https://github.com/princeton-vl/Coupled-Iterative-Refinement#optional-faster-implementation to compile the CUDA version.")
            for i in range(self.num_levels):
                corr = self.corr_pyramid[i]
                dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
                dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
                delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
                delta_lvl = delta.view(2*r+1, 2*r+1, 2)

                centroid_lvl = coords.reshape(batch*num*ht*wd, 1, 1, 2) / 2**i
                coords_lvl = centroid_lvl + delta_lvl

                corr = corr.view(batch*num*ht*wd,1,ht//2**i, wd//2**i)
                corr = bilinear_sampler(corr, coords_lvl)
                corr = corr.view(batch, num, ht, wd, -1).contiguous().permute(0,1,4,2,3)
                out_pyramid.append(corr.view(batch, num, -1, ht, wd))

        return torch.cat(out_pyramid, dim=2)

    @staticmethod
    def corr(fmaps, inds):
        fmap1 = fmaps[:, inds[0]]
        fmap2 = fmaps[:, inds[1]]

        batch, num, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.reshape(batch*num, dim, ht*wd) / 4.0
        fmap2 = fmap2.reshape(batch*num, dim, ht*wd) / 4.0
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        return corr.view(batch, num, ht, wd, ht, wd)
