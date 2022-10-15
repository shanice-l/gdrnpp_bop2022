import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import normal_init, constant_init


class ConvFuseNet(nn.Module):
    def __init__(
        self,
        rgb_nIn,
        depth_nIn,
        nOut,
        num_layers=2,
    ):
        super().__init__()
        self.rgb_nIn = rgb_nIn
        self.depth_nIn = depth_nIn
        self.nOut = nOut

        self.features = nn.ModuleList()
        # TODO: justify functions ---------------------------------
        for i in range(num_layers):
            _in_channels = rgb_nIn + depth_nIn if i == 0 else nOut
            self.features.append(
                nn.Conv2d(
                    _in_channels,
                    nOut,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
            self.features.append(nn.BatchNorm2d(nOut))
            self.features.append(nn.ReLU(inplace=True))

        # TODO: try add avgpool2d
        # init ------------------------------------------------
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)

    def forward(self, rgb_feat, depth_feat):
        x = torch.cat([rgb_feat, depth_feat], dim=1)  # along channels
        for _i, layer in enumerate(self.features):
            x = layer(x)

        return x
