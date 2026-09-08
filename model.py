#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.py
========
PIMoG 神经网络定义。

组件一览
--------
积木块
  ConvBNRelu / SingleConv / DoubleConv / up_conv / ResidualBlock

网络
  Discriminator              图像真伪判别（GAN）
  U_Net_Encoder_Diffusion    U-Net 编码器：将 30-bit 消息扩散嵌入宿主图
  Extractor                  从特征图回归 30-bit 消息
  Decoder                    特征提取 + Extractor
  Encoder_Decoder            Encoder → Noise Layer → Decoder 端到端封装

数据流
------
  x [B,3,H,W], m [B,30]
      → Encoder → Encoded_image
      → Noiser  → Noised_image
      → Decoder → Decoded_message [B,30]

公开接口
--------
- Discriminator, U_Net_Encoder_Diffusion, Extractor, Decoder, Encoder_Decoder
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from Noise_Layer import Identity, ScreenShooting, ScreenShootingMB


# ---------------------------------------------------------------------------
# 基础积木
# ---------------------------------------------------------------------------
class ConvBNRelu(nn.Module):
    """Conv3×3 → BatchNorm → ReLU。"""

    def __init__(self, channels_in: int, channels_out: int, stride: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SingleConv(nn.Module):
    """单层 Conv → BN → ReLU（可指定 stride）。"""

    def __init__(self, inchannel: int, outchannel: int, s: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=s, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualBlock(nn.Module):
    """残差块；stride≠1 或通道变化时使用 1×1 shortcut。"""

    def __init__(self, inchannel: int, outchannel: int, s: int):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if s != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=s, bias=False),
                nn.BatchNorm2d(outchannel),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.left(x)
        out += self.shortcut(x)
        return F.relu(out)


class DoubleConv(nn.Module):
    """连续两层 Conv → BN → ReLU（U-Net 编码器/解码器单元）。"""

    def __init__(self, inchannel: int, outchannel: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class up_conv(nn.Module):
    """×2 上采样 + Conv → BN → ReLU。"""

    def __init__(self, inchannel: int, outchannel: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


# ---------------------------------------------------------------------------
# 判别器
# ---------------------------------------------------------------------------
class Discriminator(nn.Module):
    """
    轻量判别器：判断图像为「干净宿主」还是「含水印编码图」。

    Parameters
    ----------
    num_channels : int
        中间特征通道数（默认由 CLI ``--num_channels`` 传入）。
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.discriminator = nn.Sequential(
            ConvBNRelu(3, num_channels),
            ConvBNRelu(num_channels, num_channels),
            ConvBNRelu(num_channels, num_channels),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.linear = nn.Linear(num_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        D = self.discriminator(x)
        D.squeeze_(3).squeeze_(2)
        return self.linear(D)


# ---------------------------------------------------------------------------
# 编码器
# ---------------------------------------------------------------------------
class U_Net_Encoder_Diffusion(nn.Module):
    """
    U-Net 风格编码器：在多尺度特征上注入扩展后的水印消息。

    消息路径: Linear(30→256) → reshape 1×16×16 → Conv → 与各解码层拼接。
    """

    def __init__(self, inchannel: int = 3, outchannel: int = 3):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Globalpool = nn.MaxPool2d(kernel_size=4, stride=4)

        self.Conv1 = DoubleConv(inchannel, 16)
        self.Conv2 = DoubleConv(16, 32)
        self.Conv3 = DoubleConv(32, 64)

        self.Up4 = up_conv(64 * 3, 64)
        self.Conv7 = DoubleConv(64 * 3, 64)

        self.Up3 = up_conv(64, 32)
        self.Conv8 = DoubleConv(32 * 2 + 64, 32)

        self.Up2 = up_conv(32, 16)
        self.Conv9 = DoubleConv(16 * 2 + 64, 16)

        self.Conv_1x1 = nn.Conv2d(16, outchannel, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(30, 256)
        self.Conv_message = DoubleConv(1, 64)

    def forward(self, x: torch.Tensor, watermark: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, 3, H, W]
        watermark : Tensor [B, 30]

        Returns
        -------
        Tensor [B, 3, H, W]  含水印编码图（仍在约 [-1,1] 范围）
        """
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)

        x6 = self.Globalpool(x4)
        x7 = x6.repeat(1, 1, 4, 4)
        expanded_message = self.linear(watermark).view(-1, 1, 16, 16)
        expanded_message = self.Conv_message(expanded_message)
        x4 = torch.cat((x4, x7, expanded_message), dim=1)

        d4 = self.Up4(x4)
        expanded_message = self.linear(watermark).view(-1, 1, 16, 16)
        expanded_message = F.interpolate(
            expanded_message, size=(d4.shape[2], d4.shape[3]), mode="bilinear"
        )
        expanded_message = self.Conv_message(expanded_message)
        d4 = torch.cat((x3, d4, expanded_message), dim=1)
        d4 = self.Conv7(d4)

        d3 = self.Up3(d4)
        expanded_message = self.linear(watermark).view(-1, 1, 16, 16)
        expanded_message = F.interpolate(
            expanded_message, size=(d3.shape[2], d3.shape[3]), mode="bilinear"
        )
        expanded_message = self.Conv_message(expanded_message)
        d3 = torch.cat((x2, d3, expanded_message), dim=1)
        d3 = self.Conv8(d3)

        d2 = self.Up2(d3)
        expanded_message = self.linear(watermark).view(-1, 1, 16, 16)
        expanded_message = F.interpolate(
            expanded_message, size=(d2.shape[2], d2.shape[3]), mode="bilinear"
        )
        expanded_message = self.Conv_message(expanded_message)
        d2 = torch.cat((x1, d2, expanded_message), dim=1)
        d2 = self.Conv9(d2)

        return self.Conv_1x1(d2)


# ---------------------------------------------------------------------------
# 解码器
# ---------------------------------------------------------------------------
class Extractor(nn.Module):
    """从深层特征回归 30-bit 水印消息。"""

    def __init__(self, inchannel: int = 64):
        super().__init__()
        self.layer1 = SingleConv(inchannel, 64, 1)
        self.layer2 = nn.Sequential(ResidualBlock(64, 64, 1), ResidualBlock(64, 64, 2))
        self.layer3 = nn.Sequential(ResidualBlock(64, 64, 1), ResidualBlock(64, 64, 2))
        self.layer4 = nn.Sequential(ResidualBlock(64, 64, 1), ResidualBlock(64, 64, 2))
        self.layer5 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.linear = nn.Linear(256, 30)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out.squeeze_(1)
        out = out.view(-1, 1, 256)
        out = self.linear(out)
        out.squeeze_(1)
        return out


class Decoder(nn.Module):
    """特征主干 + Extractor，从（可能噪声化的）图像恢复消息。"""

    def __init__(self):
        super().__init__()
        self.extractor = Extractor()
        self.layer1 = nn.Sequential(
            SingleConv(3, 64, 1),
            SingleConv(64, 64, 1),
            SingleConv(64, 64, 1),
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.layer1(x)
        return self.extractor(x1)


# ---------------------------------------------------------------------------
# 端到端
# ---------------------------------------------------------------------------
class Encoder_Decoder(nn.Module):
    """
    Encoder → 可微噪声层 → Decoder。

    Parameters
    ----------
    distortion : str
        ``'Identity'`` / ``'ScreenShooting'`` / ``'ScreenShootingMB'``。
    embed_strength : float
        >0 时使用残差嵌入：``x + strength * tanh(Encoder(x,m))``，
        限制可见失真；=0 时为论文原版（Encoder 直接输出整图）。
    """

    def __init__(self, distortion: str, embed_strength: float = 0.0):
        super().__init__()
        self.Encoder = U_Net_Encoder_Diffusion()
        self.Decoder = Decoder()
        self.distortion = distortion
        self.embed_strength = float(embed_strength)
        if distortion == "Identity":
            self.Noiser = Identity()
        elif distortion == "ScreenShooting":
            self.Noiser = ScreenShooting()
        elif distortion == "ScreenShootingMB":
            self.Noiser = ScreenShootingMB()
        else:
            raise ValueError(f"未知 distortion: {distortion}")

    def encode(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """编码得到含水印图（可残差约束）。"""
        residual = self.Encoder(x, m)
        if self.embed_strength > 0:
            return torch.clamp(x + self.embed_strength * torch.tanh(residual), -1.0, 1.0)
        return residual

    def forward(self, x: torch.Tensor, m: torch.Tensor):
        """
        Returns
        -------
        Encoded_image, Noised_image, Decoded_message
        """
        Encoded_image = self.encode(x, m)
        Noised_image = self.Noiser(Encoded_image)
        Decoded_message = self.Decoder(Noised_image.float())
        return Encoded_image, Noised_image, Decoded_message
