"""RefineNet-LightWeight. No RCU, only LightWeight-CRP block."""

import math

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch
from torch.autograd import Variable
import torchvision.models as models


model_urls = {
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


# Helpers / wrappers
def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class CRPBlock(nn.Module):

    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv1x1(in_planes if (i == 0) else out_planes,
                            out_planes, stride=1,
                            bias=False))
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = top + x
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.05)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, expansion=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes*expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes*expansion)
        self.conv2 = nn.Conv2d(inplanes*expansion, inplanes*expansion, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=inplanes*expansion)
        self.bn2 = nn.BatchNorm2d(inplanes*expansion)
        self.conv3 = nn.Conv2d(inplanes*expansion, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MobileCount(nn.Module):

    def __init__(self, num_classes=1, pretrained=False):
        self.inplanes = 32
        block = Bottleneck
        layers = [1, 2, 3, 4]
        super(MobileCount, self).__init__()

        # implement of mobileNetv2
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0], stride=1, expansion=1)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2, expansion=6)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2, expansion=6)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2, expansion=6)

        self.dropout4 = nn.Dropout(p=0.5)
        self.p_ims1d2_outl1_dimred = conv1x1(512, 64, bias=False) # change 256 to 512
        self.mflow_conv_g1_pool = self._make_crp(64, 64, 4)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(64, 32, bias=False)

        self.dropout3 = nn.Dropout(p=0.5)
        self.p_ims1d2_outl2_dimred = conv1x1(256, 32, bias=False) # change 128 to 256
        self.adapt_stage2_b2_joint_varout_dimred = conv1x1(32, 32, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(32, 32, 4)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(32, 32, bias=False)

        self.p_ims1d2_outl3_dimred = conv1x1(128, 32, bias=False) # change 64 to 128
        self.adapt_stage3_b2_joint_varout_dimred = conv1x1(32, 32, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(32, 32, 4)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(32, 32, bias=False)

        self.p_ims1d2_outl4_dimred = conv1x1(64, 32, bias=False) # change 32 to 64
        self.adapt_stage4_b2_joint_varout_dimred = conv1x1(32, 32, bias=False)
        self.mflow_conv_g4_pool = self._make_crp(32, 32, 4)

        self.dropout_clf = nn.Dropout(p=0.5)
        # self.clf_conv = nn.Conv2d(256, num_classes, kernel_size=3, stride=1,
        #                           padding=1, bias=True)
        self.clf_conv = nn.Conv2d(32, 10, kernel_size=3, stride=1,
                                  padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride, expansion):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, expansion=expansion))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, expansion=expansion))

        return nn.Sequential(*layers)

    def forward(self, x_prev, x):
        size1 = x.shape[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_prev = self.conv1(x_prev)
        x_prev = self.bn1(x_prev)
        x_prev = self.relu(x_prev)
        x_prev = self.maxpool(x_prev)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        l1_prev = self.layer1(x_prev)
        l2_prev = self.layer2(l1_prev)
        l3_prev = self.layer3(l2_prev)
        l4_prev = self.layer4(l3_prev)    

        l4_cated = torch.cat((l4_prev,l4),1)
        l3_cated = torch.cat((l3_prev,l3),1)
        l2_cated = torch.cat((l2_prev,l2),1)
        l1_cated = torch.cat((l1_prev,l1),1)

        l4_cated = self.dropout4(l4_cated)
        x4 = self.p_ims1d2_outl1_dimred(l4_cated) # conv 1x1 (254,64)
        x4 = self.relu(x4) 
        x4 = self.mflow_conv_g1_pool(x4) # CRP
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4) # conv 1x1 (64,32)
        x4 = nn.Upsample(size=l3_cated.size()[2:], mode='bilinear')(x4) #change size


        l3_cated = self.dropout3(l3_cated)
        x3 = self.p_ims1d2_outl2_dimred(l3_cated)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3) # high res feature map conv
        x3 = x3 + x4 # FUSION
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3) #CRP
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2_cated.size()[2:], mode='bilinear')(x3)

        l2_cated = self.p_ims1d2_outl3_dimred(l2_cated)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(l2_cated)
        x2 = x2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1_cated.size()[2:], mode='bilinear')(x2)

        l1_cated = self.p_ims1d2_outl4_dimred(l1_cated)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(l1_cated)
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)

        x1 = self.dropout_clf(x1)
        out = self.clf_conv(x1) # 10 channel output

        out = F.upsample(out, size=size1, mode='bilinear')
        out = F.relu(out)
        return out
