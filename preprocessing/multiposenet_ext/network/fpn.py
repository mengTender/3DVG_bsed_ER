# -*- coding:utf-8 -*-
'''RetinaFPN in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # fpn for detection subnet (RetinaNet) P6,P7
        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)  # p6
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)  # p7

        # pure fpn layers for detection subnet (RetinaNet)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # c5 -> p5
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # c4 -> p4
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # c3 -> p3
        # smooth
        self.toplayer0 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # smooth p5
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # smooth p4
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # smooth p3

        # pure fpn layers for keypoint subnet
        # Lateral layers
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # c5 -> p5
        self.flatlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # c4 -> p4
        self.flatlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # c3 -> p3
        self.flatlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)  # c2 -> p2
        # smooth
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # smooth p4
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # smooth p3
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # smooth p2

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: top feature map to be upsampled.
          y: lateral feature map.

        Returns:
          added feature map.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='nearest', align_corners=None) + y  # bilinear, False

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # pure fpn for detection subnet, RetinaNet
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p5 = self.toplayer0(p5)
        p4 = self.toplayer1(p4)
        p3 = self.toplayer2(p3)

        # pure fpn for keypoints estimation
        fp5 = self.toplayer(c5)
        fp4 = self._upsample_add(fp5,self.flatlayer1(c4))
        fp3 = self._upsample_add(fp4,self.flatlayer2(c3))
        fp2 = self._upsample_add(fp3,self.flatlayer3(c2))
        # Smooth
        fp4 = self.smooth1(fp4)
        fp3 = self.smooth2(fp3)
        fp2 = self.smooth3(fp2)

        return [[fp2,fp3,fp4,fp5],[p3, p4, p5, p6, p7]]

def FPN50():
    # [3,4,6,3] -> resnet50
    return FPN(Bottleneck, [3,4,6,3])

def FPN101():
    # [3,4,23,3] -> resnet101
    return FPN(Bottleneck, [3,4,23,3])
