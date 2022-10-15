import torch.nn as nn
import torch
from torchvision.models.resnet import BasicBlock, Bottleneck
from mmcv.cnn import normal_init, constant_init


# Specification
resnet_spec = {
    "resnet18": (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 18),
    "resnet34": (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 34),
    "resnet50": (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 50),
    "resnet101": (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 101),
    "resnet152": (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 152),
}


def get_resnet(model_name, in_chans=3, freeze=False, rot_concat=False, **kwargs):
    assert model_name in resnet_spec, model_name
    block_type, layers, channels, num_layers = resnet_spec[model_name]
    return ResNetBackboneNet(
        block_type,
        layers,
        in_channel=in_chans,
        freeze=freeze,
        rot_concat=rot_concat,
    )


class ResNetBackboneNet(nn.Module):
    def __init__(self, block, layers, in_channel=3, freeze=False, rot_concat=False):
        self.freeze = freeze
        self.rot_concat = rot_concat
        self.inplanes = 64
        super(ResNetBackboneNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):  # x.shape [32, 3, 256, 256]
        if self.freeze:
            with torch.no_grad():
                x = self.conv1(x)  # x.shape [32, 64, 128, 128]
                x = self.bn1(x)
                x = self.relu(x)
                x_low_feature = self.maxpool(x)  # x.shape [32, 64, 64, 64]
                x_f64 = self.layer1(x_low_feature)  # x.shape [32, 256, 64, 64]
                x_f32 = self.layer2(x_f64)  # x.shape [32, 512, 32, 32]
                x_f16 = self.layer3(x_f32)  # x.shape [32, 1024, 16, 16]
                x_high_feature = self.layer4(x_f16)  # x.shape [32, 2048, 8, 8]
                if self.rot_concat:
                    return (
                        x_high_feature.detach(),
                        x_f64.detach(),
                        x_f32.detach(),
                        x_f16.detach(),
                    )
                else:
                    return x_high_feature.detach()
        else:
            x = self.conv1(x)  # x.shape [32, 64, 128, 128]
            x = self.bn1(x)
            x = self.relu(x)
            x_low_feature = self.maxpool(x)  # x.shape [32, 64, 64, 64]
            x_f64 = self.layer1(x_low_feature)  # x.shape [32, 256, 64, 64]
            x_f32 = self.layer2(x_f64)  # x.shape [32, 512, 32, 32]
            x_f16 = self.layer3(x_f32)  # x.shape [32, 1024, 16, 16]
            x_high_feature = self.layer4(x_f16)  # x.shape [32, 2048, 8, 8]
            if self.rot_concat:
                return x_high_feature, x_f64, x_f32, x_f16
            else:
                return x_high_feature
