import torch

from torchvision import models
from torch import nn


class FCN(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = models.resnet101(pretrained=True)

        layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(*layers[:5])
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.layer2 = layers[5]
        self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.layer3 = layers[6]
        self.upsample3 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.layer4 = layers[7]
        self.upsample4 = nn.Upsample(scale_factor=32, mode='bilinear')

        self.conv_final = nn.Conv2d(256 + 512 + 1024 + 2048, n_class, 1)

    def forward(self, x):
        x = self.layer1(x)
        up1 = self.upsample1(x)
        x = self.layer2(x)
        up2 = self.upsample2(x)
        x = self.layer3(x)
        up3 = self.upsample3(x)
        x = self.layer4(x)
        up4 = self.upsample4(x)

        merge = torch.cat([up1, up2, up3, up4], dim=1)
        merge = self.conv_final(merge)

        out = merge

        return out


def make_resnet101(**kwargs):
    return FCN(**kwargs)
