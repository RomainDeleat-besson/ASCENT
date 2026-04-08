import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    """
    k: kernel
    p: padding
    d: dilation
    """
    if d > 1:
        # actual kernel-size
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, paddin g, groups, dilation,
    activation)."""

    default_act = nn.GELU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(
            c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True
        )
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    """Depth-wise convolution with args(ch_in, ch_out, kernel, stride, dilation, activation)."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


# Lightweight Cascade Multi-Receptive Fields Module
class CMRF(nn.Module):
    """CMRF Module with args(ch_in, ch_out, number, shortcut, groups, expansion)."""

    def __init__(self, c1, c2, N=8, shortcut=True, g=1, e=0.5):
        super().__init__()

        self.N = N
        self.c = int(c2 * e / self.N)
        self.add = shortcut and c1 == c2

        self.pwconv1 = Conv(c1, c2 // self.N, 1, 1)
        self.pwconv2 = Conv(c2 // 2, c2, 1, 1)
        self.m = nn.ModuleList(DWConv(self.c, self.c, k=3, act=False) for _ in range(N - 1))

    def forward(self, x):
        """Forward pass through CMRF Module."""
        x_residual = x
        x = self.pwconv1(x)

        x = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
        x.extend(m(x[-1]) for m in self.m)
        x[0] = x[0] + x[1]
        x.pop(1)

        y = torch.cat(x, dim=1)
        y = self.pwconv2(y)
        return x_residual + y if self.add else y


# Encoder in TinyU-Net
class UNetEncoder(nn.Module):
    """Encoder block for the U-Net architecture.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.

    Attributes:
        cmrf (CMRF): CMRF block for feature extraction.
        downsample (MaxPool2d): Max pooling layer for downsampling the feature maps.

    Forward pass:
        The forward method applies the CMRF block and downsamples the output
        feature maps.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cmrf = CMRF(in_channels, out_channels)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """Forward pass through the encoder.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tuple[Tensor, Tensor]: Downsampled tensor and the output of CMRF.
        """
        x = self.cmrf(x)
        return self.downsample(x), x


# Decoder in TinyU-Net
class UNetDecoder(nn.Module):
    """Decoder block for the U-Net architecture.

    Args:
        in_channels (int): The number of input channels from the previous layer.
        out_channels (int): The number of output channels.

    Attributes:
        cmrf (CMRF): CMRF block for processing the concatenated feature maps.
        upsample (callable): Function for upsampling the feature maps.

    Forward pass:
        The forward method upsamples the input tensor and concatenates it
        with the skip connection before passing it through the CMRF block.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cmrf = CMRF(in_channels, out_channels)
        self.upsample = F.interpolate

    def forward(self, x, skip_connection):
        """Forward pass through the decoder.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).
            skip_connection (Tensor): Skip connection tensor from the encoder.

        Returns:
            Tensor: Output tensor after processing through CMRF.
        """
        x = self.upsample(x, scale_factor=2, mode="bicubic", align_corners=False)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.cmrf(x)
        return x


class TinyUNet(nn.Module):
    """Tiny U-Net architecture for image segmentation.

    Args:
        in_channels (int): The number of input channels (default: 3).
        num_classes (int): The number of segmentation classes (default: 2).

    Attributes:
        encoder1 (UNetEncoder): First encoder layer.
        encoder2 (UNetEncoder): Second encoder layer.
        encoder3 (UNetEncoder): Third encoder layer.
        encoder4 (UNetEncoder): Fourth encoder layer.
        decoder4 (UNetDecoder): Fourth decoder layer.
        decoder3 (UNetDecoder): Third decoder layer.
        decoder2 (UNetDecoder): Second decoder layer.
        decoder1 (UNetDecoder): First decoder layer.
        final_conv (Conv2d): Final convolutional layer for output.
    """

    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]

        self.encoder1 = UNetEncoder(in_channels, 64)
        self.encoder2 = UNetEncoder(64, 128)
        self.encoder3 = UNetEncoder(128, 256)
        self.encoder4 = UNetEncoder(256, 512)

        self.decoder4 = UNetDecoder(in_filters[3], out_filters[3])
        self.decoder3 = UNetDecoder(in_filters[2], out_filters[2])
        self.decoder2 = UNetDecoder(in_filters[1], out_filters[1])
        self.decoder1 = UNetDecoder(in_filters[0], out_filters[0])
        self.final_conv = nn.Conv2d(out_filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        """Forward pass through the Tiny U-Net.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Output segmentation map of shape (B, num_classes, H, W).
        """
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)
        x, skip4 = self.encoder4(x)

        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        x = self.final_conv(x)
        return x


if __name__ == "__main__":
    model = TinyUNet(in_channels=1, num_classes=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    im = torch.rand((2, 1, 160, 160)).to(device)

    model(im)
