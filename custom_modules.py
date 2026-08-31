"""
Standalone PyTorch module definitions for CBAM, ASPP, and CoT3.
Used for YOLOv5-CASP / YOLOv8-CASP framework integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel Attention Module for CBAM."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """Spatial Attention Module for CBAM."""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(spatial))
        return x * out


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)."""

    def __init__(self, c1, c2=None, ratio=16, kernel_size=7):
        super().__init__()
        c = c1 if c2 is None else c2
        self.ca = ChannelAttention(c, reduction=ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        return self.sa(self.ca(x))


class ASPPConv(nn.Sequential):
    """Dilated Convolution Branch for ASPP."""

    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    """Global Pooling Branch for ASPP."""

    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) module with parallel dilated convolutions [1, 3, 5, 7]."""

    def __init__(self, c1, c2, rates=[1, 3, 5, 7]):
        super().__init__()
        out_channels = c2 // 5
        modules = [
            nn.Sequential(
                nn.Conv2d(c1, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            )
        ]
        for rate in rates[1:]:
            modules.append(ASPPConv(c1, out_channels, rate))
        modules.append(ASPPPooling(c1, out_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        return self.project(torch.cat(res, dim=1))


class CoTBlock(nn.Module):
    """Contextual Transformer (CoT) Block."""

    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )
        self.query_embed = nn.Conv2d(dim, dim, 1, bias=False)
        self.value_embed = nn.Conv2d(dim, dim, 1, bias=False)
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)
        q = self.query_embed(x)
        v = self.value_embed(x)
        kq = torch.cat([k1, q], dim=1)
        att = self.attention_embed(kq)
        att = self.softmax(att.view(bs, c, -1)).view(bs, c, h, w)
        k2 = att * v
        return k1 + k2


class CoTBottleneck(nn.Module):
    """Bottleneck module using CoTBlock."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        # Assuming Conv is imported or defined
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_, 1, 1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )
        self.cot = CoTBlock(c_)
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_, c2, 1, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cot(self.cv1(x))) if self.add else self.cv2(self.cot(self.cv1(x)))


class CoT3(nn.Module):
    """C3-style block with CoTBottleneck."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_, 1, 1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(c1, c_, 1, 1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )
        self.cv3 = nn.Sequential(
            nn.Conv2d(2 * c_, c2, 1, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.m = nn.Sequential(*(CoTBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
