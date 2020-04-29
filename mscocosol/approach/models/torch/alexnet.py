import torch

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


class AlexNetV2(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.n_class = 10

        # --- Alex net
        backbone = models.alexnet()
        self._head_layers = list(list(backbone.children())[0])

        self._Conv2d_1 = self._head_layers[0]
        self._ReLU_2 = self._head_layers[1]
        self._MaxPool2d_3 = self._head_layers[2]
        self._Conv2d_4 = self._head_layers[3]
        self._ReLU_5 = self._head_layers[4]
        self._MaxPool2d_6 = self._head_layers[5]
        self._Conv2d_7 = self._head_layers[6]
        self._ReLU_8 = self._head_layers[7]
        self._Conv2d_9 = self._head_layers[8]
        self._ReLU_10 = self._head_layers[9]
        self._Conv2d_11 = self._head_layers[10]
        self._ReLU_12 = self._head_layers[11]
        # --- ---

        self._conv2d_transposed1 = nn.ConvTranspose2d(in_channels=256, out_channels=192, kernel_size=3, stride=2, padding=0)
        self._conv2d_transposed2 = nn.ConvTranspose2d(in_channels=192, out_channels=64, kernel_size=5, stride=2, padding=1)
        self._conv2d_transposed3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=11, stride=4, padding=2, output_padding=1)
        self._conv2d1 = nn.Conv2d(in_channels=64, out_channels=n_class, kernel_size=1)

        self._log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self._Conv2d_1(x)
        relu2 = self._ReLU_2(x)
        x = self._MaxPool2d_3(relu2)
        x = self._Conv2d_4(x)
        relu5 = self._ReLU_5(x)
        x = self._MaxPool2d_6(relu5)
        x = self._Conv2d_7(x)
        x = self._ReLU_8(x)
        x = self._Conv2d_9(x)
        x = self._ReLU_10(x)
        x = self._Conv2d_11(x)
        x = self._ReLU_12(x)

        conv2d_tr1 = self._conv2d_transposed1(x)
        x = torch.add(relu5, conv2d_tr1)

        conv2d_tr2 = self._conv2d_transposed2(x)
        x = torch.add(relu2, conv2d_tr2)

        x = self._conv2d_transposed3(x)
        x = self._conv2d1(x)

        x = self._log_softmax(x)

        out = x
        return out


def make_alex_net_v2(**kwargs):
    return AlexNetV2(**kwargs)


class AlexNetV3(nn.Module):
    """
    The network designed for image size 500x500
    """

    def __init__(self, n_class):
        super().__init__()

        self.n_class = 10

        # --- Alex net
        backbone = models.alexnet()
        self._head_layers = list(list(backbone.children())[0])

        self._Conv2d_1 = self._head_layers[0]
        self._ReLU_2 = self._head_layers[1]
        self._MaxPool2d_3 = self._head_layers[2]
        self._Conv2d_4 = self._head_layers[3]
        self._ReLU_5 = self._head_layers[4]
        self._MaxPool2d_6 = self._head_layers[5]
        self._Conv2d_7 = self._head_layers[6]
        self._ReLU_8 = self._head_layers[7]
        self._Conv2d_9 = self._head_layers[8]
        self._ReLU_10 = self._head_layers[9]
        self._Conv2d_11 = self._head_layers[10]
        self._ReLU_12 = self._head_layers[11]
        # --- ---

        self._fc6 = nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=7, padding=0)
        self._fc6_relu = nn.ReLU(inplace=True)

        self._fc7 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1, padding=0)
        self._fc7_relu = nn.ReLU(inplace=True)

        self._score_fr_sem = nn.Conv2d(in_channels=4096, out_channels=n_class, kernel_size=1, padding=0)
        self._conv2d_transposed1 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=7, stride=1, padding=0)

        self._relu_12_conv2d = nn.Conv2d(in_channels=256, out_channels=n_class, kernel_size=1)

        self._conv2d_transposed2 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=5, stride=2, padding=1)

        self._relu_5_conv2d = nn.Conv2d(in_channels=192, out_channels=n_class, kernel_size=1)

        self._conv2d_transposed3 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=5, stride=2, padding=1, output_padding=1)

        self._relu_2_conv2d = nn.Conv2d(in_channels=64, out_channels=n_class, kernel_size=1)

        self._conv2d_transposed4 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=11, stride=4, padding=2, output_padding=1)
        self._conv2d_final = nn.Conv2d(in_channels=n_class, out_channels=n_class, kernel_size=1)

        self._log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self._Conv2d_1(x)
        relu2 = self._ReLU_2(x)
        x = self._MaxPool2d_3(relu2)
        x = self._Conv2d_4(x)
        relu5 = self._ReLU_5(x)
        x = self._MaxPool2d_6(relu5)
        x = self._Conv2d_7(x)
        x = self._ReLU_8(x)
        x = self._Conv2d_9(x)
        x = self._ReLU_10(x)
        x = self._Conv2d_11(x)
        relu_12 = self._ReLU_12(x)

        x = self._fc6(relu_12)
        x = self._fc6_relu(x)

        x = self._fc7(x)
        x = self._fc7_relu(x)

        x = self._score_fr_sem(x)

        up1 = self._conv2d_transposed1(x)

        skip_relu12 = self._relu_12_conv2d(relu_12)

        x = torch.add(up1, skip_relu12)

        up2 = self._conv2d_transposed2(x)

        skip_relu5 = self._relu_5_conv2d(relu5)

        x = torch.add(up2, skip_relu5)

        up3 = self._conv2d_transposed3(x)

        skip_relu2 = self._relu_2_conv2d(relu2)

        x = torch.add(up3, skip_relu2)

        up4 = self._conv2d_transposed4(x)

        x = self._conv2d_final(up4)
        x = self._log_softmax(x)

        return x


def make_alex_net_v3(**kwargs):
    return AlexNetV3(**kwargs)


class AlexNetV4(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.n_class = 10

        # --- Alex net
        backbone = models.alexnet()
        self._head_layers = list(list(backbone.children())[0])

        self._Conv2d_1 = self._head_layers[0]
        self._ReLU_2 = self._head_layers[1]
        self._MaxPool2d_3 = self._head_layers[2]
        self._Conv2d_4 = self._head_layers[3]
        self._ReLU_5 = self._head_layers[4]
        self._MaxPool2d_6 = self._head_layers[5]
        self._Conv2d_7 = self._head_layers[6]
        self._ReLU_8 = self._head_layers[7]
        self._Conv2d_9 = self._head_layers[8]
        self._ReLU_10 = self._head_layers[9]
        self._Conv2d_11 = self._head_layers[10]
        self._ReLU_12 = self._head_layers[11]
        # --- ---

        self._fc6 = nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=7, padding=0)
        self._fc6_relu = nn.ReLU(inplace=True)
        self._fc6_drop = nn.Dropout(p=0.5, inplace=False)

        self._fc7 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1, padding=0)
        self._fc7_relu = nn.ReLU(inplace=True)
        self._fc7_drop = nn.Dropout(p=0.5, inplace=False)

        self._score_fr_sem = nn.Conv2d(in_channels=4096, out_channels=n_class, kernel_size=1, padding=0)
        self._conv2d_transposed1 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=7, stride=1, padding=0)

        self._relu_12_conv2d = nn.Conv2d(in_channels=256, out_channels=n_class, kernel_size=1)

        self._conv2d_transposed2 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=5, stride=2, padding=1)

        self._relu_5_conv2d = nn.Conv2d(in_channels=192, out_channels=n_class, kernel_size=1)

        self._conv2d_transposed3 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=5, stride=2, padding=1, output_padding=1)

        self._relu_2_conv2d = nn.Conv2d(in_channels=64, out_channels=n_class, kernel_size=1)

        self._conv2d_transposed4 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=11, stride=4, padding=2, output_padding=1)
        self._conv2d_final = nn.Conv2d(in_channels=n_class, out_channels=n_class, kernel_size=1)

        self._log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self._Conv2d_1(x)
        relu2 = self._ReLU_2(x)
        x = self._MaxPool2d_3(relu2)
        x = self._Conv2d_4(x)
        relu5 = self._ReLU_5(x)
        x = self._MaxPool2d_6(relu5)
        x = self._Conv2d_7(x)
        x = self._ReLU_8(x)
        x = self._Conv2d_9(x)
        x = self._ReLU_10(x)
        x = self._Conv2d_11(x)
        relu_12 = self._ReLU_12(x)

        x = self._fc6(relu_12)
        x = self._fc6_relu(x)
        x = self._fc6_drop(x)

        x = self._fc7(x)
        x = self._fc7_relu(x)
        x = self._fc7_drop(x)

        x = self._score_fr_sem(x)

        up1 = self._conv2d_transposed1(x)

        skip_relu12 = self._relu_12_conv2d(relu_12)

        x = torch.add(up1, skip_relu12)

        up2 = self._conv2d_transposed2(x)

        skip_relu5 = self._relu_5_conv2d(relu5)

        x = torch.add(up2, skip_relu5)

        up3 = self._conv2d_transposed3(x)

        skip_relu2 = self._relu_2_conv2d(relu2)

        x = torch.add(up3, skip_relu2)

        up4 = self._conv2d_transposed4(x)

        x = self._conv2d_final(up4)
        x = self._log_softmax(x)

        return x


def make_alex_net_v4(**kwargs):
    return AlexNetV4(**kwargs)


class AlexNetV5(nn.Module):
    def __init__(self, n_class):
        """
        Comparing to v5: reduce bottlneck from size 24 to 10
        """
        super().__init__()

        self.n_class = 10

        # --- Alex net
        backbone = models.alexnet()
        self._head_layers = list(list(backbone.children())[0])

        self._Conv2d_1 = self._head_layers[0]
        self._ReLU_2 = self._head_layers[1]
        self._MaxPool2d_3 = self._head_layers[2]
        self._Conv2d_4 = self._head_layers[3]
        self._ReLU_5 = self._head_layers[4]
        self._MaxPool2d_6 = self._head_layers[5]
        self._Conv2d_7 = self._head_layers[6]
        self._ReLU_8 = self._head_layers[7]
        self._Conv2d_9 = self._head_layers[8]
        self._ReLU_10 = self._head_layers[9]
        self._Conv2d_11 = self._head_layers[10]
        self._ReLU_12 = self._head_layers[11]
        # --- ---

        self._fc6 = nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=7, padding=0)
        self._fc6_relu = nn.ReLU(inplace=True)
        self._fc6_drop = nn.Dropout(p=0.5, inplace=False)

        self._fc7 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1, padding=0)
        self._fc7_relu = nn.ReLU(inplace=True)
        self._fc7_drop = nn.Dropout(p=0.5, inplace=False)

        self._max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self._score_fr_sem = nn.Conv2d(in_channels=4096, out_channels=n_class, kernel_size=1, padding=0)
        self._conv2d_transposed0 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=7, stride=2, padding=0)
        self._conv2d_transposed1 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=7, stride=1, padding=0)
        self._ident = nn.Identity()

        self._relu_12_conv2d = nn.Conv2d(in_channels=256, out_channels=n_class, kernel_size=1)

        self._conv2d_transposed2 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=5, stride=2, padding=1)

        self._relu_5_conv2d = nn.Conv2d(in_channels=192, out_channels=n_class, kernel_size=1)

        self._conv2d_transposed3 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=5, stride=2, padding=1, output_padding=1)

        self._relu_2_conv2d = nn.Conv2d(in_channels=64, out_channels=n_class, kernel_size=1)

        self._conv2d_transposed4 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=11, stride=4, padding=2, output_padding=1)
        self._conv2d_final = nn.Conv2d(in_channels=n_class, out_channels=n_class, kernel_size=1)

        self._log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self._Conv2d_1(x)
        relu2 = self._ReLU_2(x)
        x = self._MaxPool2d_3(relu2)
        x = self._Conv2d_4(x)
        relu5 = self._ReLU_5(x)
        x = self._MaxPool2d_6(relu5)
        x = self._Conv2d_7(x)
        x = self._ReLU_8(x)
        x = self._Conv2d_9(x)
        x = self._ReLU_10(x)
        x = self._Conv2d_11(x)
        relu_12 = self._ReLU_12(x)

        x = self._max_pool(relu_12)

        x = self._fc6(x)
        x = self._fc6_relu(x)
        x = self._fc6_drop(x)

        x = self._fc7(x)
        x = self._fc7_relu(x)
        x = self._fc7_drop(x)

        x = self._score_fr_sem(x)

        up0 = self._conv2d_transposed0(x)
        up1 = self._conv2d_transposed1(up0)
        x = up1[:, :, 0:-1, 0:-1]
        x = self._ident(x)

        skip_relu12 = self._relu_12_conv2d(relu_12)
        x = torch.add(x, skip_relu12)

        up2 = self._conv2d_transposed2(x)

        skip_relu5 = self._relu_5_conv2d(relu5)

        x = torch.add(up2, skip_relu5)

        up3 = self._conv2d_transposed3(x)

        skip_relu2 = self._relu_2_conv2d(relu2)

        x = torch.add(up3, skip_relu2)

        up4 = self._conv2d_transposed4(x)

        x = self._conv2d_final(up4)
        x = self._log_softmax(x)

        return x


def make_alex_net_v5(**kwargs):
    return AlexNetV5(**kwargs)


class AlexNetV6(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.n_class = 10

        # --- Alex net
        backbone = models.alexnet()
        self._head_layers = list(list(backbone.children())[0])

        self._Conv2d_1 = self._head_layers[0]
        self._ReLU_2 = self._head_layers[1]
        self._MaxPool2d_3 = self._head_layers[2]
        self._Conv2d_4 = self._head_layers[3]
        self._ReLU_5 = self._head_layers[4]
        self._MaxPool2d_6 = self._head_layers[5]
        self._Conv2d_7 = self._head_layers[6]
        self._ReLU_8 = self._head_layers[7]
        self._Conv2d_9 = self._head_layers[8]
        self._ReLU_10 = self._head_layers[9]
        self._Conv2d_11 = self._head_layers[10]
        self._ReLU_12 = self._head_layers[11]
        # --- ---

        self._fc6 = nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=3, padding=2, dilation=2)
        self._fc6_relu = nn.ReLU(inplace=True)
        self._fc6_drop = nn.Dropout(p=0.5, inplace=False)

        self._fc7 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1, padding=0)
        self._fc7_relu = nn.ReLU(inplace=True)
        self._fc7_drop = nn.Dropout(p=0.5, inplace=False)

        self._max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self._score_fr_sem = nn.Conv2d(in_channels=4096, out_channels=n_class, kernel_size=1, padding=0)

        self._conv2d_transposed1 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=7, stride=2, padding=3)
        self._ident = nn.Identity()

        self._relu_12_conv2d = nn.Conv2d(in_channels=256, out_channels=n_class, kernel_size=1)

        self._conv2d_transposed2 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=5, stride=2, padding=1)

        self._relu_5_conv2d = nn.Conv2d(in_channels=192, out_channels=n_class, kernel_size=1)

        self._conv2d_transposed3 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=5, stride=2, padding=1, output_padding=1)

        self._relu_2_conv2d = nn.Conv2d(in_channels=64, out_channels=n_class, kernel_size=1)

        self._conv2d_transposed4 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=11, stride=4, padding=2, output_padding=1)
        self._conv2d_final = nn.Conv2d(in_channels=n_class, out_channels=n_class, kernel_size=1)

        self._log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self._Conv2d_1(x)
        relu2 = self._ReLU_2(x)
        x = self._MaxPool2d_3(relu2)
        x = self._Conv2d_4(x)
        relu5 = self._ReLU_5(x)
        x = self._MaxPool2d_6(relu5)
        x = self._Conv2d_7(x)
        x = self._ReLU_8(x)
        x = self._Conv2d_9(x)
        x = self._ReLU_10(x)
        x = self._Conv2d_11(x)
        relu_12 = self._ReLU_12(x)

        x = self._max_pool(relu_12)

        x = self._fc6(x)
        x = self._fc6_relu(x)
        x = self._fc6_drop(x)

        x = self._fc7(x)
        x = self._fc7_relu(x)
        x = self._fc7_drop(x)

        x = self._score_fr_sem(x)

        up1 = self._conv2d_transposed1(x)
        x = up1[:, :, 0:-1, 0:-1]
        x = self._ident(x)

        skip_relu12 = self._relu_12_conv2d(relu_12)
        x = torch.add(x, skip_relu12)

        up2 = self._conv2d_transposed2(x)
        x = up2

        skip_relu5 = self._relu_5_conv2d(relu5)

        x = torch.add(up2, skip_relu5)

        up3 = self._conv2d_transposed3(x)
        x = up3

        skip_relu2 = self._relu_2_conv2d(relu2)

        x = torch.add(up3, skip_relu2)

        up4 = self._conv2d_transposed4(x)

        x = self._conv2d_final(up4)
        x = self._log_softmax(x)

        return x


def make_alex_net_v6(**kwargs):
    return AlexNetV6(**kwargs)

