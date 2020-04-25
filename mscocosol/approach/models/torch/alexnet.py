from torch import nn

from torchvision import models


class FCNv1(nn.Module):
    """
    First variant to try
    """
    def __init__(self, n_class):
        super().__init__()

        self.n_class = 10

        backbone = models.alexnet()

        self._features = nn.Sequential(*list(backbone.children())[0][:-1])
        self._conv2d_transposed1 = nn.ConvTranspose2d(in_channels=256, out_channels=192, kernel_size=3, stride=2, padding=0)
        self._conv2d_transposed2 = nn.ConvTranspose2d(in_channels=192, out_channels=64, kernel_size=5, stride=2, padding=1)
        self._conv2d_transposed3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=11, stride=4, padding=2, output_padding=1)
        self._conv2d1 = nn.Conv2d(in_channels=64, out_channels=n_class, kernel_size=1)

        self._log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self._features(x)
        x = self._conv2d_transposed1(x)
        x = self._conv2d_transposed2(x)
        x = self._conv2d_transposed3(x)
        x = self._conv2d1(x)

        x = self._log_softmax(x)

        out = x
        return out


def make_alex_net_v1(**kwargs):
    return FCNv1(**kwargs)
