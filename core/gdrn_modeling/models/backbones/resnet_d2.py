from detectron2.modeling.backbone import BasicStem, BottleneckBlock, ResNet
from detectron2.checkpoint import DetectionCheckpointer
from torch import nn


class ResNet50_GN_D2(nn.Module):
    def __init__(self, in_channels=3, out_features=["res5"], weights="catalog://ImageNetPretrained/FAIR/R-50-GN"):
        super().__init__()
        self.resnet = ResNet(
            stem=BasicStem(in_channels=in_channels, out_channels=64, norm="GN"),
            stages=ResNet.make_default_stages(depth=50, stride_in_1x1=False, norm="GN"),
            out_features=out_features,
        )
        if weights != "":
            DetectionCheckpointer(self.resnet).load(weights)
        self.out_features = out_features

    def forward(self, x):
        out = self.resnet(x)
        out = [out[name] for name in self.out_features]
        return out
