import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import normal_init, constant_init
from .resnet import resnet18, resnet50, resnet34


class Resnet18_8s(nn.Module):
    def __init__(
        self,
        fcdim=256,
        s8dim=128,
        s4dim=64,
        s2dim=32,
        raw_dim=32,
        pretrained=True,
        in_chans=3,
        concat_input=False,
    ):
        super(Resnet18_8s, self).__init__()
        self.concat_input = concat_input

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = resnet18(
            fully_conv=True,
            pretrained=pretrained,
            in_chans=in_chans,
            output_stride=8,
            remove_avg_pool_layer=True,
        )

        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Sequential(
            nn.Conv2d(resnet18_8s.inplanes, fcdim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fcdim),
            nn.ReLU(True),
        )
        self.resnet18_8s = resnet18_8s

        # x8s->128
        self.conv8s = nn.Sequential(
            nn.Conv2d(128 + fcdim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1, True),
        )
        self.up8sto4s = nn.UpsamplingBilinear2d(scale_factor=2)

        # x4s->64
        self.conv4s = nn.Sequential(
            nn.Conv2d(64 + s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1, True),
        )
        self.up4sto2s = nn.UpsamplingBilinear2d(scale_factor=2)

        # x2s->64
        self.conv2s = nn.Sequential(
            nn.Conv2d(64 + s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1, True),
        )
        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)

        if concat_input:
            in_raw_dim = s2dim + in_chans
        else:
            in_raw_dim = s2dim
        self.convraw = nn.Sequential(
            nn.Conv2d(in_raw_dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1, True),
            # nn.Conv2d(raw_dim, self.seg_dim + self.ver_dim, 1, 1),
        )

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x, feature_alignment=False):
        x2s, x4s, x8s, x16s, x32s, xfc = self.resnet18_8s(x)

        fm = self.conv8s(torch.cat([xfc, x8s], 1))
        fm = self.up8sto4s(fm)

        fm = self.conv4s(torch.cat([fm, x4s], 1))
        fm = self.up4sto2s(fm)

        fm = self.conv2s(torch.cat([fm, x2s], 1))
        fm = self.up2storaw(fm)

        if self.concat_input:
            out = self.convraw(torch.cat([fm, x], 1))
        else:
            out = self.convraw(fm)
        # NOTE: do prediction outside to allow more custom predictions
        return out


class Resnet50_8s(nn.Module):
    def __init__(
        self,
        fcdim=384,
        s8dim=256,
        s4dim=128,
        s2dim=64,
        raw_dim=64,
        pretrained=True,
        in_chans=3,
        concat_input=False,
    ):
        super(Resnet50_8s, self).__init__()
        self.concat_input = concat_input

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_8s = resnet50(
            fully_conv=True,
            pretrained=pretrained,
            in_chans=in_chans,
            output_stride=8,
            remove_avg_pool_layer=True,
        )

        # Randomly initialize the 1x1 Conv scoring layer
        resnet50_8s.fc = nn.Sequential(
            nn.Conv2d(resnet50_8s.inplanes, fcdim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fcdim),
            nn.ReLU(True),
        )
        self.resnet50_8s = resnet50_8s

        # x8s->128
        self.conv8s = nn.Sequential(
            nn.Conv2d(128 * 4 + fcdim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1, True),
        )
        self.up8sto4s = nn.UpsamplingBilinear2d(scale_factor=2)

        # x4s->64
        self.conv4s = nn.Sequential(
            nn.Conv2d(64 * 4 + s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1, True),
        )
        self.up4sto2s = nn.UpsamplingBilinear2d(scale_factor=2)

        # x2s->64
        self.conv2s = nn.Sequential(
            nn.Conv2d(64 + s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1, True),
        )
        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)

        if concat_input:
            in_raw_dim = s2dim + in_chans
        else:
            in_raw_dim = s2dim
        self.convraw = nn.Sequential(
            nn.Conv2d(in_raw_dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1, True),
            # nn.Conv2d(raw_dim, seg_dim + ver_dim, 1, 1),
        )

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x, feature_alignment=False):
        x2s, x4s, x8s, x16s, x32s, xfc = self.resnet50_8s(x)

        fm = self.conv8s(torch.cat([xfc, x8s], 1))
        fm = self.up8sto4s(fm)

        fm = self.conv4s(torch.cat([fm, x4s], 1))
        fm = self.up4sto2s(fm)

        fm = self.conv2s(torch.cat([fm, x2s], 1))
        fm = self.up2storaw(fm)

        if self.concat_input:
            out = self.convraw(torch.cat([fm, x], 1))
        else:
            out = self.convraw(fm)
        return out


class Resnet50_8s_2o(nn.Module):
    """stride 1 --> stride 8 --> stride 2 (out)"""

    def __init__(
        self,
        fcdim=384,
        s8dim=256,
        s4dim=128,
        s2dim=64,
        pretrained=True,
        in_chans=3,
        concat_input=False,
    ):
        super(Resnet50_8s_2o, self).__init__()
        self.concat_input = concat_input

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_8s = resnet50(
            fully_conv=True,
            pretrained=pretrained,
            in_chans=in_chans,
            output_stride=8,
            remove_avg_pool_layer=True,
        )

        # Randomly initialize the 1x1 Conv scoring layer
        resnet50_8s.fc = nn.Sequential(
            nn.Conv2d(resnet50_8s.inplanes, fcdim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fcdim),
            nn.ReLU(True),
        )
        self.resnet50_8s = resnet50_8s

        # x8s->128
        self.conv8s = nn.Sequential(
            nn.Conv2d(128 * 4 + fcdim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1, True),
        )
        self.up8sto4s = nn.UpsamplingBilinear2d(scale_factor=2)

        # x4s->64
        self.conv4s = nn.Sequential(
            nn.Conv2d(64 * 4 + s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1, True),
        )
        self.up4sto2s = nn.UpsamplingBilinear2d(scale_factor=2)

        # x2s->64
        if concat_input:
            in_dim = 64 + s4dim + in_chans
        else:
            in_dim = 64 + s4dim
        self.conv2s = nn.Sequential(
            nn.Conv2d(in_dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1, True),
            # nn.Conv2d(s2dim, seg_dim + ver_dim, 1, 1),
        )

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x, feature_alignment=False):
        x2s, x4s, x8s, x16s, x32s, xfc = self.resnet50_8s(x)

        fm = self.conv8s(torch.cat([xfc, x8s], 1))
        fm = self.up8sto4s(fm)

        fm = self.conv4s(torch.cat([fm, x4s], 1))
        fm = self.up4sto2s(fm)

        if self.concat_input:
            x_ds = F.interpolate(x, scale_factor=0.5, mode="bilinear")
            out = self.conv2s(torch.cat([fm, x2s, x_ds], 1))
        else:
            out = self.conv2s(torch.cat([fm, x2s], 1))
        return out


class Resnet34_8s(nn.Module):
    def __init__(
        self,
        fcdim=384,
        s8dim=256,
        s4dim=128,
        s2dim=64,
        raw_dim=64,
        pretrained=True,
        in_chans=3,
        concat_input=False,
    ):
        super(Resnet34_8s, self).__init__()
        self.concat_input = concat_input

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_8s = resnet34(
            fully_conv=True,
            pretrained=pretrained,
            in_chans=in_chans,
            output_stride=8,
            remove_avg_pool_layer=True,
        )

        # Randomly initialize the 1x1 Conv scoring layer
        resnet50_8s.fc = nn.Sequential(
            nn.Conv2d(resnet50_8s.inplanes, fcdim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fcdim),
            nn.ReLU(True),
        )
        self.resnet50_8s = resnet50_8s

        # x8s->128
        self.conv8s = nn.Sequential(
            nn.Conv2d(128 + fcdim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1, True),
        )
        self.up8sto4s = nn.UpsamplingBilinear2d(scale_factor=2)

        # x4s->64
        self.conv4s = nn.Sequential(
            nn.Conv2d(64 + s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1, True),
        )
        self.up4sto2s = nn.UpsamplingBilinear2d(scale_factor=2)

        # x2s->64
        self.conv2s = nn.Sequential(
            nn.Conv2d(64 + s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1, True),
        )
        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)

        if concat_input:
            in_raw_dim = in_chans + s2dim
        else:
            in_raw_dim = s2dim
        self.convraw = nn.Sequential(
            nn.Conv2d(in_raw_dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1, True),
            # nn.Conv2d(raw_dim, seg_dim + ver_dim, 1, 1),
        )

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x, feature_alignment=False):
        x2s, x4s, x8s, x16s, x32s, xfc = self.resnet50_8s(x)

        fm = self.conv8s(torch.cat([xfc, x8s], 1))
        fm = self.up8sto4s(fm)

        fm = self.conv4s(torch.cat([fm, x4s], 1))
        fm = self.up4sto2s(fm)

        fm = self.conv2s(torch.cat([fm, x2s], 1))
        fm = self.up2storaw(fm)

        if self.concat_input:
            out = self.convraw(torch.cat([fm, x], 1))
        else:
            out = self.convraw(fm)
        return out


class Resnet18_8s_detector(nn.Module):
    def __init__(self):
        super(Resnet18_8s_detector, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        self.resnet18_8s = resnet18(
            fully_conv=True,
            pretrained=True,
            output_stride=8,
            remove_avg_pool_layer=True,
        )
        self.resnet18_8s.fc = nn.Conv2d(self.resnet18_8s.inplanes, 1, 3, 1, 1)

    def forward(self, x):
        _, _, _, _, _, xfc = self.resnet18_8s(x)
        return xfc


class Resnet18_8s_detector_v2(nn.Module):
    def __init__(self, base_detector):
        super(Resnet18_8s_detector_v2, self).__init__()
        self.base_detector = base_detector
        self.out_conv = nn.Conv2d(128, 1, 3, 1, 1)

    def forward(self, x):
        x = self.base_detector.resnet18_8s.conv1(x)
        x = self.base_detector.resnet18_8s.bn1(x)
        x2s = self.base_detector.resnet18_8s.relu(x)
        x = self.base_detector.resnet18_8s.maxpool(x2s)
        x4s = self.base_detector.resnet18_8s.layer1(x)
        x8s = self.base_detector.resnet18_8s.layer2(x4s)
        return self.out_conv(x8s)


if __name__ == "__main__":
    # test varying input size
    import numpy as np

    for k in range(50):
        hi, wi = np.random.randint(0, 29), np.random.randint(0, 49)
        h, w = 256 + hi * 8, 256 + wi * 8
        print(h, w)
        img = np.random.uniform(-1, 1, [1, 3, h, w]).astype(np.float32)
        net = Resnet50_8s(2, 2).cuda()
        out = net(torch.tensor(img).cuda())
