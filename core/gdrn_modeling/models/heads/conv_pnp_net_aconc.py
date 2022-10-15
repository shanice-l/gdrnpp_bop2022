import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import normal_init, constant_init
from lib.torch_utils.layers.layer_utils import get_norm, get_nn_act_func
from lib.torch_utils.layers.dropblock import DropBlock2D, LinearScheduler


class ConvPnPNetAconC(nn.Module):
    def __init__(
        self,
        nIn,
        num_regions=8,
        mask_attention_type="none",
        featdim=128,
        rot_dim=6,
        num_stride2_layers=3,
        num_extra_layers=0,
        norm="GN",
        num_gn_groups=32,
        act="aconc",
        drop_prob=0.0,
        dropblock_size=5,
        flat_op="flatten",
        final_spatial_size=(8, 8),
        denormalize_by_extent=True,
    ):
        """
        Args:
            nIn: input feature channel
            spatial_pooltype: max | soft
            spatial_topk: 1
            flat_op: flatten | avg | avg-max | avg-max-min
        """
        super().__init__()
        self.featdim = featdim
        self.num_regions = num_regions
        self.mask_attention_type = mask_attention_type
        self.flat_op = flat_op
        self.denormalize_by_extent = denormalize_by_extent

        conv_act = get_nn_act_func(act)

        # -----------------------------------
        self.drop_prob = drop_prob
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=dropblock_size),
            start_value=0.0,
            stop_value=drop_prob,
            nr_steps=5000,
        )

        self.features = nn.ModuleList()
        for i in range(num_stride2_layers):
            _in_channels = nIn if i == 0 else featdim
            self.features.append(
                nn.Conv2d(
                    _in_channels,
                    featdim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            self.features.append(get_norm(norm, featdim, num_gn_groups=num_gn_groups))
            self.features.append(conv_act)
        for i in range(num_extra_layers):
            self.features.append(
                nn.Conv2d(
                    featdim,
                    featdim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
            self.features.append(get_norm(norm, featdim, num_gn_groups=num_gn_groups))
            self.features.append(conv_act)

        final_h, final_w = final_spatial_size
        fc_in_dim = {
            "flatten": featdim * final_h * final_w,
            "avg": featdim,
            "avg-max": featdim * 2,
            "avg-max-min": featdim * 3,
        }[flat_op]

        # self.fc1 = nn.Linear(featdim * 8 * 8 + 128, 1024)  # NOTE: 128 for extents feature
        self.fc1 = nn.Linear(fc_in_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc_r = nn.Linear(256, rot_dim)  # quat or rot6d
        # TODO: predict centroid and z separately
        self.fc_t = nn.Linear(256, 3)

        # feature for extent
        # self.extent_fc1 = nn.Linear(3, 64)
        # self.extent_fc2 = nn.Linear(64, 128)

        # init ------------------------------------
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)
        normal_init(self.fc_r, std=0.01)
        normal_init(self.fc_t, std=0.01)

    def forward(self, coor_feat, region=None, extents=None, mask_attention=None):
        """
        Args:
             since this is the actual correspondence
            x: (B,C,H,W)
            extents: (B, 3)
        Returns:

        """
        bs, in_c, fh, fw = coor_feat.shape
        if in_c in [3, 5] and self.denormalize_by_extent and extents is not None:
            coor_feat[:, :3, :, :] = (coor_feat[:, :3, :, :] - 0.5) * extents.view(bs, 3, 1, 1)
        # convs
        if region is not None:
            x = torch.cat([coor_feat, region], dim=1)
        else:
            x = coor_feat

        if self.mask_attention_type != "none":
            assert mask_attention is not None
            if self.mask_attention_type == "mul":
                x = x * mask_attention
            elif self.mask_attention_type == "concat":
                x = torch.cat([x, mask_attention], dim=1)
            else:
                raise ValueError(f"Wrong mask attention type: {self.mask_attention_type}")

        if self.drop_prob > 0:
            self.dropblock.step()  # increment number of iterations
            x = self.dropblock(x)

        for _i, layer in enumerate(self.features):
            x = layer(x)

        flat_conv_feat = x.flatten(2)  # [B,featdim,*]
        if self.flat_op == "flatten":
            flat_conv_feat = flat_conv_feat.flatten(1)
        elif self.flat_op == "avg":
            flat_conv_feat = flat_conv_feat.mean(-1)  # spatial global average pooling
        elif self.flat_op == "avg-max":
            flat_conv_feat = torch.cat([flat_conv_feat.mean(-1), flat_conv_feat.max(-1)[0]], dim=-1)
        elif self.flat_op == "avg-max-min":
            flat_conv_feat = torch.cat(
                [
                    flat_conv_feat.mean(-1),
                    flat_conv_feat.max(-1)[0],
                    flat_conv_feat.min(-1)[0],
                ],
                dim=-1,
            )
        else:
            raise ValueError(f"Invalid flat_op: {self.flat_op}")
        # extent feature
        # # TODO: use extent the other way: denormalize coords
        # x_extent = self.act(self.extent_fc1(extents))
        # x_extent = self.act(self.extent_fc2(x_extent))
        # x = torch.cat([x, x_extent], dim=1)
        #
        x = self.act(self.fc1(flat_conv_feat))
        x = self.act(self.fc2(x))
        #
        rot = self.fc_r(x)
        t = self.fc_t(x)
        return rot, t
