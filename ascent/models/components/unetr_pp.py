# Unetr++ model based on "UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation" https://github.com/Amshaker/unetr_plus_plus
from typing import Optional, Sequence, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_dropout_layer, get_norm_layer
from timm.models.layers import trunc_normal_
from torch import nn


def multiply_value_in(x: Union[list[int], tuple[int, ...]]):
    """Function that return the product of each element in a list."""
    result = 1
    for i in range(len(x)):
        result *= x[i]
    return result


def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Sequence[int], int] = 3,
    stride: Union[Sequence[int], int] = 1,
    act: Optional[Union[Tuple, str]] = Act.PRELU,
    norm: Union[Tuple, str] = Norm.INSTANCE,
    dropout: Optional[Union[Tuple, str, float]] = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
):
    """Get convolution layer.

    Args:
        spatial_dims: Number of spatial dimensions.
        in_channels: Number of channels of the input.
        out_channels: Number of channels of the output.
        kernel_size: kernel size of the padding.
        stride: stride size of the padding.
        act: Define the act layer.
        norm: Define the norm layer.
        dropout: Define the dropout layer.
        bias: bool whether to add bias or not.
        conv_only: bool
        is_transposed: bool
    """
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


def get_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:
    """Get padding layer.

    Args:
        kernel_size: kernel size of the padding.
        stride: stride size of the padding.
    """
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError(
            "padding value should not be negative, please change the kernel size and/or stride."
        )
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
    kernel_size: Union[Sequence[int], int],
    stride: Union[Sequence[int], int],
    padding: Union[Sequence[int], int],
) -> Union[Tuple[int, ...], int]:
    """Get padding output.

    Args:
        kernel_size: kernel size of the padding.
        stride: stride size of the padding.
        padding: define the padding operation.
    """
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError(
            "out_padding value should not be negative, please change the kernel size and/or stride."
        )
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]


class LayerNorm(nn.Module):
    """A Layer Normalisation."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        """Args:
        x: input image."""
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class TransformerBlock(nn.Module):
    """A transformer block, based on: "Shaker et al., UNETR++: Delving into Efficient and Accurate
    3D Medical Image Segmentation"."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        proj_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        pos_embed=False,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        self.spatial_dims = spatial_dims

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = EPA(
            input_size=input_size,
            hidden_size=hidden_size,
            proj_size=proj_size,
            num_heads=num_heads,
            channel_attn_drop=dropout_rate,
            spatial_attn_drop=dropout_rate,
        )
        self.conv51 = UnetResBlock(
            spatial_dims, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch"
        )
        self.conv52 = UnetResBlock(
            spatial_dims, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch"
        )
        self.conv8 = nn.Sequential(
            get_dropout_layer(("dropout", {"p": 0.1, "inplace": False}), spatial_dims),
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=1,
            ),
        )

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        """Args:
        x: input image."""
        if self.spatial_dims == 3:
            B, C, H, W, D = x.shape

            x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

            if self.pos_embed is not None:
                x = x + self.pos_embed

            attn = x + self.gamma * self.epa_block(self.norm(x))

            attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
            attn = self.conv51(attn_skip)
            attn = self.conv52(attn)
            x = attn_skip + self.conv8(attn)

        elif self.spatial_dims == 2:
            B, C, H, W = x.shape

            x = x.reshape(B, C, H * W).permute(0, 2, 1)

            if self.pos_embed is not None:
                x = x + self.pos_embed

            attn = x + self.gamma * self.epa_block(self.norm(x))

            attn_skip = attn.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
            attn = self.conv51(attn_skip)
            attn = self.conv52(attn)
            x = attn_skip + self.conv8(attn)
        else:
            raise NotImplementedError(
                "Dims should be 2 or 3 as the model is only for 2D or 3D images"
            )
        return x


class EPA(nn.Module):
    """Efficient Paired Attention Block, based on: "Shaker et al., UNETR++: Delving into Efficient
    and Accurate 3D Medical Image Segmentation".

    Args:
            input_size: number of channel of the input.
            hidden_size: number of channels of attention.
            proj_size: number of output channels.
            num_heads: number of head of attention block.
            qkv_bias: Define whether to add bias or not in transformer qkv.
            channel_attn_drop: dropout probability of channel attention.
            spatial_attn_drop: dropout probability of spatial attention.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        proj_size,
        num_heads=4,
        qkv_bias=False,
        channel_attn_drop=0.1,
        spatial_attn_drop=0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)

        # E and F are projection matrices used in spatial attention module to project keys and values from HWD-dimension to P-dimension
        self.E = nn.Linear(input_size, proj_size)
        self.F = nn.Linear(input_size, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

    def forward(self, x):
        """Args:
        x: input image."""
        B, N, C = x.shape

        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)

        qkvv = qkvv.permute(2, 0, 3, 1, 4)

        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_shared_projected = self.E(k_shared)

        v_SA_projected = self.F(v_SA)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature

        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)

        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        # Concat fusion
        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        x = torch.cat((x_SA, x_CA), dim=-1)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"temperature", "temperature2"}


class Upsample(nn.Module):
    """Up-sampling block."""

    def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=False):
        super().__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        """Args:
        x: input image."""
        return nn.functional.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )


class UnetOutBlock(nn.Module):
    """Unet Output block."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            dropout: dropout probability.
        """
        super().__init__()
        self.conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            dropout=dropout,
            bias=True,
            conv_only=True,
        )

    def forward(self, inp):
        """Args:
        inp: input image."""
        return self.conv(inp)


class UnetrUpBlock(nn.Module):
    """Unetr ++ Up-sampling blocks.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        upsample_kernel_size: convolution kernel size for transposed convolution layers.
        norm_name: feature normalization type and arguments.
        proj_size: projection size for keys and values in the spatial attention module.
        num_heads: number of heads inside each EPA module.
        out_size: spatial size for each decoder.
        depth: number of blocks for the current decoder stage.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        proj_size: int = 64,
        num_heads: int = 4,
        out_size: int = 0,
        depth: int = 3,
        conv_decoder: bool = False,
    ) -> None:
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_block = nn.ModuleList()

        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block
        # (see suppl. material in the paper)
        if conv_decoder:
            self.decoder_block.append(
                UnetResBlock(
                    spatial_dims,
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    norm_name=norm_name,
                )
            )
        else:
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(
                    TransformerBlock(
                        input_size=out_size,
                        hidden_size=out_channels,
                        proj_size=proj_size,
                        num_heads=num_heads,
                        dropout_rate=0.1,
                        pos_embed=True,
                        spatial_dims=spatial_dims,
                    )
                )
            self.decoder_block.append(nn.Sequential(*stage_blocks))

    def _init_weights(self, m):
        """Args:
        m: layer to initialize."""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip):
        """Args:
        inp: input image.
        skip: skip connection input."""
        out = self.transp_conv(inp)
        out = out + skip
        out = self.decoder_block[0](out)

        return out


class UnetResBlock(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        """
        Args:
                spatial_dims: number of spatial dimensions.
                in_channels: number of input channels.
                out_channels: number of output channels.
                kernel_size: convolution kernel size.
                stride: convolution stride.
                norm_name: feature normalization type and arguments.
                act_name: activation layer type and arguments.
                dropout: dropout probability.
        """
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            conv_only=True,
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(
            name=norm_name, spatial_dims=spatial_dims, channels=out_channels
        )
        self.norm2 = get_norm_layer(
            name=norm_name, spatial_dims=spatial_dims, channels=out_channels
        )
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv_layer(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                dropout=dropout,
                conv_only=True,
            )
            self.norm3 = get_norm_layer(
                name=norm_name, spatial_dims=spatial_dims, channels=out_channels
            )

    def forward(self, inp):
        """Args:
        inp: input image."""
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out


class UnetrPPEncoder(nn.Module):
    """Unetr ++ encoder.

    Args:
                input_size: spatial dimension of the input (H*W*D) at each stage of decoder.
                dims: Feature dimension at each stage of encoder.
                depths: number of blocks for each stage.
                kernel_size: Kernel size at each stage of the encoder.
                stride: Stride size at each stage of the encoder.
                proj_size: size of features for projection.
                num_heads: number of attention heads.
                spatial_dims: Number of dimension of the input image.
                in_channels: dimension of input channels.
                dropout: faction of the input units to drop.
                transformer_dropout_rate: faction of the input units to drop for the transformer block.
    """

    def __init__(
        self,
        input_size,
        dims,
        depths,
        kernel_size,
        stride,
        proj_size=None,
        num_heads=4,
        spatial_dims=3,
        in_channels=1,
        dropout=0.0,
        transformer_dropout_rate=0.1,
        **kwargs,
    ):
        super().__init__()
        if proj_size is None:
            proj_size = [64, 64, 64, 32]
        self.spatial_dims = spatial_dims

        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            get_conv_layer(
                spatial_dims,
                in_channels,
                dims[0],
                kernel_size=kernel_size[0],
                stride=stride[0],
                dropout=dropout,
                conv_only=True,
            ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(
                    spatial_dims,
                    dims[i],
                    dims[i + 1],
                    kernel_size=kernel_size[i + 1],
                    stride=stride[i + 1],
                    dropout=dropout,
                    conv_only=True,
                ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(
                    TransformerBlock(
                        input_size=input_size[i],
                        hidden_size=dims[i],
                        proj_size=proj_size[i],
                        num_heads=num_heads,
                        dropout_rate=transformer_dropout_rate,
                        pos_embed=True,
                        spatial_dims=spatial_dims,
                    )
                )
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Args:
        m: layer to initialize."""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        hidden_states = []
        x = self.downsample_layers[0](x)

        x = self.stages[0](x)

        hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

            if i == 3:  # Reshape the output of the last stage
                if self.spatial_dims == 3:
                    x = einops.rearrange(x, "b c h w d -> b (h w d) c")

                elif self.spatial_dims == 2:
                    x = einops.rearrange(x, "b c h w -> b (h w) c")
                else:
                    raise NotImplementedError(
                        "Dims should be 2 or 3 as the model is only for 2D or 3D images"
                    )
            hidden_states.append(x)
        return x, hidden_states

    def forward(self, x):
        """Args:
        x: input image."""
        x, hidden_states = self.forward_features(x)
        return x, hidden_states


class UNETR_PP(nn.Module):
    """UNETR++ based on: "Shaker et al., UNETR++: Delving into Efficient and Accurate 3D Medical
    Image Segmentation"."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        patch_size_: Union[list[int], tuple[int, ...]],
        feature_size: int,
        feature_size_multiplier: Union[list[int], tuple[int, ...]],
        depths: Union[list[int], tuple[int, ...]],
        kernel_size: list[list[int]],
        stride: list[list[int]],
        proj_feat_size: Union[list[int], tuple[int, ...]],
        num_heads: int = 4,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        dropout_rate: float = 0.0,
        deep_supervision=True,
        **kwargs,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            num_classes: number of classes to segment, dimension of output channels.
            patch_size: patch size of the inputs image.
            feature_size: dimension of network feature size.
            feature_size_multiplier: multipliers for the number of channels at each down/up-sampling level.
            depths: number of blocks for each stage.
            kernel_size: Kernel size at each stage of the Unetr.
            stride: Stride size at each stage of the Unetr.
            proj_feat_size: size of features for projection.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            deep_supervision: use deep supervision to compute the loss.
        """

        super().__init__()
        assert len(feature_size_multiplier) == len(depths) == len(kernel_size) == 4
        assert len(patch_size_) == len(proj_feat_size)

        self.patch_size = kwargs["patch_size"]
        self.in_channels = in_channels
        # self.deep_supervision = False

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.proj_feat_size = proj_feat_size
        self.hidden_feature_size = (
            feature_size * feature_size_multiplier[-1]
        )  # feature size at the deeper level of our model
        all_features_size = [
            feature_size * multiplier for multiplier in feature_size_multiplier
        ]  # feature size at each level of our model

        self.spatial_dims = len(patch_size_)

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        self.deep_supervision = deep_supervision

        kernel_size_decoder2 = kernel_size[0]
        kernel_size_decoder3 = kernel_size[1]
        kernel_size_decoder4 = kernel_size[2]
        kernel_size_decoder5 = kernel_size[3]

        out_size_decoder2 = multiply_value_in(patch_size_)
        out_size_decoder3 = out_size_decoder2 // (multiply_value_in(kernel_size_decoder2))
        out_size_decoder4 = out_size_decoder3 // (multiply_value_in(kernel_size_decoder3))
        out_size_decoder5 = out_size_decoder4 // (multiply_value_in(kernel_size_decoder4))
        out_size_hidden = out_size_decoder5 // (multiply_value_in(kernel_size_decoder5))

        self.unetr_pp_encoder = UnetrPPEncoder(
            input_size=[out_size_decoder3, out_size_decoder4, out_size_decoder5, out_size_hidden],
            dims=all_features_size,
            depths=depths,
            kernel_size=kernel_size,
            stride=stride,
            num_heads=num_heads,
            spatial_dims=self.spatial_dims,
            in_channels=self.in_channels,
        )

        self.encoder1 = UnetResBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=all_features_size[3],
            out_channels=all_features_size[2],
            kernel_size=3,
            upsample_kernel_size=kernel_size_decoder5,
            norm_name=norm_name,
            out_size=out_size_decoder5,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=all_features_size[2],
            out_channels=all_features_size[1],
            kernel_size=3,
            upsample_kernel_size=kernel_size_decoder4,
            norm_name=norm_name,
            out_size=out_size_decoder4,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=all_features_size[1],
            out_channels=all_features_size[0],
            kernel_size=3,
            upsample_kernel_size=kernel_size_decoder3,
            norm_name=norm_name,
            out_size=out_size_decoder3,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=all_features_size[0],
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=kernel_size_decoder2,
            norm_name=norm_name,
            out_size=out_size_decoder2,
            conv_decoder=True,
        )
        self.out1 = UnetOutBlock(
            spatial_dims=self.spatial_dims, in_channels=feature_size, out_channels=self.num_classes
        )
        if self.deep_supervision:
            self.out2 = UnetOutBlock(
                spatial_dims=self.spatial_dims,
                in_channels=feature_size * 2,
                out_channels=self.num_classes,
            )
            self.out3 = UnetOutBlock(
                spatial_dims=self.spatial_dims,
                in_channels=feature_size * 4,
                out_channels=self.num_classes,
            )

    def proj_feat(self, x, hidden_size, proj_feat_size):
        """Feature projection.

        Args:
            x: output image of the network.
            hidden_size: number of features of the input of projection.
            proj_feat_size: number of features of the output of projection.
        """
        if self.spatial_dims == 3:
            x = x.view(
                x.size(0), proj_feat_size[0], proj_feat_size[1], proj_feat_size[2], hidden_size
            )
            x = x.permute(0, 4, 1, 2, 3).contiguous()
        elif self.spatial_dims == 2:
            x = x.view(x.size(0), proj_feat_size[0], proj_feat_size[1], hidden_size)
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            raise NotImplementedError(
                "Dims should be 2 or 3 as the model is only for 2D or 3D images"
            )
        return x

    def forward(self, x_in):
        """Args:
        x_in: input image of the network."""
        x_output, hidden_states = self.unetr_pp_encoder(x_in)

        convBlock = self.encoder1(x_in)
        # Four encoders
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]

        # Four decoders
        dec4 = self.proj_feat(enc4, self.hidden_feature_size, self.proj_feat_size)
        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)

        out = self.decoder2(dec1, convBlock)
        if self.deep_supervision and self.training:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out1(out)

        return logits
