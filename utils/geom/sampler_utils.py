import torch
import torch.nn.functional as F


def _bilinear_sampler(img, coords, mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def bilinear_sampler(img, coords, use_mask=False):
    """ Wrapper for bilinear sampler for inputs with extra batch dimensions """
    unflatten = False

    assert img.shape[0] == coords.shape[0]
    assert img.ndim == coords.ndim

    if len(img.shape) == 5:
        unflatten = True
        b, n, *_ = img.shape
        img = img.flatten(0,1)
        coords = coords.flatten(0,1)

    assert img.shape[0] == coords.shape[0]
    assert img.ndim == coords.ndim == 4

    if use_mask:
        img1, mask = _bilinear_sampler(img, coords, mask=True)
        if unflatten:
            return img1.view(b, n, *(img1.shape[2:])), mask.view(b, n, *(img1.shape[2:]))
        return img1, mask
    else:
        img1 = _bilinear_sampler(img, coords)
        assert img1.ndim == 4
        if unflatten:
            return img1.unflatten(0, (b, n))
        return img1

def sample_depths(depths, coords):
    depths = depths.unsqueeze(2)
    depths_proj = bilinear_sampler(depths, coords)
    return depths_proj.squeeze(2).unsqueeze(-1)
