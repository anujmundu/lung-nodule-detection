# custom_modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============ CBAM Module ============
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# ============ ASPP Module ============
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[-2:]
        x = self.avgpool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[1, 3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            ASPPConv(in_channels, out_channels, dilations[1]),
            ASPPConv(in_channels, out_channels, dilations[2]),
            ASPPConv(in_channels, out_channels, dilations[3]),
            ASPPPooling(in_channels, out_channels)
        ])
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        return self.project(res)

# ============ CoT3 Module ============
class CoT3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.conv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)
        self.conv_k = nn.Conv2d(2 * self.c, 2 * self.c, 3, 1, padding=1, groups=g)
        self.conv_v = nn.Conv2d(2 * self.c, 2 * self.c, 1, 1)
        self.conv_q = nn.Conv2d(2 * self.c, 2 * self.c, 1, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.conv_out = nn.Conv2d(2 * self.c, c2, 1, 1)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        x = self.conv1(x)
        k = self.conv_k(x)
        v = self.conv_v(x)
        q = self.conv_q(x)
        
        b, c, h, w = q.shape
        q = q.view(b, c, h * w).permute(0, 2, 1)
        k = k.view(b, c, h * w)
        attn = self.softmax(torch.bmm(q, k) / (c ** 0.5))
        v = v.view(b, c, h * w).permute(0, 2, 1)
        dynamic = torch.bmm(attn, v).permute(0, 2, 1).view(b, c, h, w)
        out = k.view(b, c, h, w) + dynamic
        out = self.conv_out(out)
        if self.shortcut:
            out = x[:, :self.c] + out
        return out