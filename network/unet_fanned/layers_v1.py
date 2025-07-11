import torch
import torch.nn as nn
import torchvision.transforms.functional as tf


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        return self.up(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.relu = nn.ReLU(inplace=True)
        self.correctance = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), bias=False, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), bias=False, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        save = x
        x = self.double_conv(x)

        if self.in_channels != self.out_channels:
            save = self.correctance(save)

        torch.add(x, save)

        return self.relu(x)
