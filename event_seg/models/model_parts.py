""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN (Instance Normalization)] => ReLU) * 2 + optional dropout"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_rate=0.0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()  # Dropout AFTER activation
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv + dropout AFTER convolutions"""

    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)  # Pooling first
        self.conv = DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)  # Dropout inside DoubleConv

    def forward(self, x):
        x = self.maxpool(x)  # Downsample first
        return self.conv(x)  # Convolution with dropout AFTER activation


class Up(nn.Module):
    """Upscaling then double conv (Dropout ONLY inside DoubleConv)"""

    def __init__(self, in_channels, out_channels, bilinear=False, dropout_rate=0.0):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout_rate=dropout_rate)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)  # Dropout inside DoubleConv

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)  # Dropout inside DoubleConv


class OutConv(nn.Module):
    """Final 2D convolution layer"""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ========================================================
# 3D Variants (For Event-Based & Temporal Models)
# ========================================================

class DoubleConv3D(nn.Module):
    """(convolution => [BN (Instance Normalization)] => ReLU) * 2 for 3D data + optional dropout"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_rate=0.0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()  # Dropout AFTER activation
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """Downscaling with maxpool then double conv for 3D data + dropout AFTER convolutions"""

    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # Pool only in spatial dimensions
        self.conv = DoubleConv3D(in_channels, out_channels, dropout_rate=dropout_rate)  # Dropout inside DoubleConv3D


    def forward(self, x):
        if x.size(2) == 0:
            raise ValueError("Temporal dimension collapsed to zero. Adjust input size or pooling parameters.")
        x = self.maxpool(x)  # Downsample first
        return self.conv(x)  # Convolution with dropout AFTER activation


class Up3D(nn.Module):
    """Upscaling then double conv for 3D data (Dropout ONLY inside DoubleConv3D)"""

    def __init__(self, in_channels, out_channels, bilinear=True, dropout_rate=0.0):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2, dropout_rate=dropout_rate)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels, dropout_rate=dropout_rate)  # Dropout inside DoubleConv3D

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)  # Dropout inside DoubleConv3D


class OutConv3D(nn.Module):
    """Final 3D convolution layer"""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)