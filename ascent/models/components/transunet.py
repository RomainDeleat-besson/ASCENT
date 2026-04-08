import copy
import math
import platform
from collections import OrderedDict
from os.path import join, normpath

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torch.nn import Conv2d, Dropout, LayerNorm, Linear, Softmax

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def pjoin(path, *paths):
    """Cross-platform path join function. On Windows, converts backslashes to forward slashes.

    Args:
        path (str): Base path
        *paths (str): Additional path elements

    Returns:
        str: Normalized and joined path
    """
    p = join(path, *paths)
    if platform.system() == "Windows":
        return normpath(p).replace("\\", "/")
    else:
        return p


def np2th(weights, conv=False):
    """Converts NumPy weights to PyTorch tensor. Optionally reorders for conv layers.

    Args:
        weights (np.array): NumPy weight array
        conv (bool): Whether to transpose conv weights from HWIO to OIHW

    Returns:
        torch.Tensor: Converted tensor
    """
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):
    """A Conv2D layer with weight standardization.

    The weights are normalized by subtracting mean and dividing by std during forward pass.
    """

    def forward(self, x):
        """Forward pass.

        Args:
            - x: image 2D
        """
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    """Standardized 3x3 convolution with padding=1.

    Args:
        cin (int): Input channels
        cout (int): Output channels
        stride (int): Convolution stride
        groups (int): Grouped convolution
        bias (bool): Whether to include bias

    Returns:
        nn.Module: StdConv2d instance
    """
    return StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    """
    Returns a 1x1 StdConv2d (no padding)
    Args:
        cin (int): Input channels
        cout (int): Output channels
        stride (int): Convolution stride
        groups (int): Grouped convolution
        bias (bool): Whether to include bias

    Returns:
        nn.Module: StdConv2d instance
    """
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation bottleneck block used in ResNetV2.

    Args:
        cin (int): Input channels
        cout (int): Output channels (optional, default = cin)
        cmid (int): Middle channels (optional, default = cout // 4)
        stride (int): Stride for downsampling
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        """Forward pass through bottleneck block.

        Args:
            x (Tensor): Input tensor (B, C, H, W)

        Returns:
            Tensor: Output tensor after residual + bottleneck path
        """
        # Residual branch
        residual = x
        if hasattr(self, "downsample"):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        """Loads pretrained weights from a nested dictionary. The names are formed like
        "block/unit/layer_name".

        Args:
        weights (dict): Dictionary of pretrained weights
        n_block (str): Block name (e.g., 'block1')
        n_unit (str): Unit name within the block (e.g., 'unit01')
        """
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, "downsample"):
            proj_conv_weight = np2th(
                weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True
            )
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))


class ResNetV2(nn.Module):
    """Pre-activation (v2) ResNet backbone used for feature extraction.

    Args:
        block_units (list[int]): Number of bottleneck units in each ResNet block.
        width_factor (float): Scaling factor for the width (number of channels).

    Attributes:
        root (nn.Sequential): Initial convolution + group norm + ReLU.
        body (nn.Sequential): Residual blocks composed of PreActBottleneck units.
    """

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(
            OrderedDict(
                [
                    ("conv", StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
                    ("gn", nn.GroupNorm(32, width, eps=1e-6)),
                    ("relu", nn.ReLU(inplace=True)),
                    # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
                ]
            )
        )

        self.body = nn.Sequential(
            OrderedDict(
                [
                    (
                        "block1",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(cin=width, cout=width * 4, cmid=width),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 4, cout=width * 4, cmid=width
                                        ),
                                    )
                                    for i in range(2, block_units[0] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block2",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width * 4, cout=width * 8, cmid=width * 2, stride=2
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 8, cout=width * 8, cmid=width * 2
                                        ),
                                    )
                                    for i in range(2, block_units[1] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block3",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width * 8,
                                            cout=width * 16,
                                            cmid=width * 4,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 16, cout=width * 16, cmid=width * 4
                                        ),
                                    )
                                    for i in range(2, block_units[2] + 1)
                                ],
                            )
                        ),
                    ),
                ]
            )
        )

    def forward(self, x):
        """Forward pass through the ResNetV2 encoder.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            x (Tensor): Final output of the last block.
            features (List[Tensor]): List of intermediate features in reverse order.
        """
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, f"x {x.size()} should {right_size}"
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0 : x.size()[2], 0 : x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


def swish(x):
    """Swish activation function.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Activated output.
    """
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    """Multi-head self-attention layer for Transformer blocks.

    Args:
        hidden_size (int): Dimension of the hidden state.
        transformer_num_heads (int): Number of attention heads.
        transformer_attention_dropout_rate (float): Dropout rate.
        vis (bool): If True, stores attention weights for visualization.

    Attributes:
        query/key/value (Linear): Projections for attention.
        out (Linear): Output projection.
    """

    def __init__(
        self, hidden_size, transformer_num_heads, transformer_attention_dropout_rate, vis
    ):
        super().__init__()
        self.vis = vis
        self.num_attention_heads = transformer_num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(hidden_size, self.all_head_size)
        self.key = Linear(hidden_size, self.all_head_size)
        self.value = Linear(hidden_size, self.all_head_size)

        self.out = Linear(hidden_size, hidden_size)
        self.attn_dropout = Dropout(transformer_attention_dropout_rate)
        self.proj_dropout = Dropout(transformer_attention_dropout_rate)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        """Reshape input tensor for multi-head attention.

        Args:
            x (Tensor): Input tensor of shape (B, N, D)

        Returns:
            Tensor: Reshaped to (B, heads, N, head_dim)
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        """Forward pass for attention.

        Args:
            hidden_states (Tensor): Input hidden states.

        Returns:
            attention_output (Tensor): Output tensor after attention.
            weights (Tensor or None): Attention weights (if vis=True).
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    """Feed-forward MLP block in Transformer.

    Args:
        hidden_size (int): Input/output dimension.
        transformer_mlp_dim (int): Hidden dimension inside MLP.
        transformer_dropout_rate (float): Dropout rate.
    """

    def __init__(self, hidden_size, transformer_mlp_dim, transformer_dropout_rate):
        super().__init__()
        self.fc1 = Linear(hidden_size, transformer_mlp_dim)
        self.fc2 = Linear(transformer_mlp_dim, hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(transformer_dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        """Forward pass through the MLP.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after two linear layers and dropout.
        """
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.Image to patch embedding module
    with optional ResNet hybrid backbone.

    Args:
        img_size (tuple[int]): Size of input images.
        hidden_size (int): Dimension of the output embeddings.
        in_channels (int): Number of input channels.
        patches_grid (tuple[int] or None): Grid shape for hybrid mode.
        patches_size (tuple[int]): Size of each patch.
        resnet_num_layers (list[int]): Block unit counts for ResNet.
        resnet_width_factor (float): Width scaling for ResNet.
        dropout_rate (float): Dropout rate for embeddings.
    """

    def __init__(
        self,
        img_size,
        hidden_size,
        in_channels,
        patches_grid,
        patches_size,
        resnet_num_layers,
        resnet_width_factor,
        dropout_rate,
    ):
        super().__init__()
        self.hybrid = None

        if patches_grid is not None:  # ResNet
            grid_size = patches_grid
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            n_patches = (img_size[0] // patches_size[0]) * (img_size[1] // patches_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(
                block_units=resnet_num_layers, width_factor=resnet_width_factor
            )
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patches_size,
            stride=patches_size,
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))

        self.dropout = Dropout(dropout_rate)

    def forward(self, x):
        """Forward pass to produce patch + position embeddings.

        Args:
            x (Tensor): Input image tensor (B, C, H, W)

        Returns:
            embeddings (Tensor): Final patch + position embeddings (B, N, D)
            features (List[Tensor] or None): Intermediate features from ResNet (if hybrid)
        """
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    """
    A single Transformer encoder block consisting of:
    - Layer normalization
    - Multi-head self-attention with dropout
    - Residual connections
    - MLP (feed-forward network) with dropout

    Args:
        hidden_size (int): Dimensionality of input and output embeddings.
        transformer_mlp_dim (int): Hidden size of the MLP.
        transformer_dropout_rate (float): Dropout probability in the MLP.
        transformer_num_heads (int): Number of attention heads.
        transformer_attention_dropout_rate (float): Dropout probability in attention layers.
        vis (bool): If True, attention weights are returned for visualization.

    Forward input:
        x (Tensor): Input tensor of shape (batch_size, seq_len, hidden_size)

    Returns:
        x (Tensor): Output tensor after transformer block.
        attn (Tensor or None): Attention weights if vis=True, else None.
    """

    def __init__(
        self,
        hidden_size,
        transformer_mlp_dim,
        transformer_dropout_rate,
        transformer_num_heads,
        transformer_attention_dropout_rate,
        vis,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size, transformer_mlp_dim, transformer_dropout_rate)
        self.attn = Attention(
            hidden_size, transformer_num_heads, transformer_attention_dropout_rate, vis
        )

    def forward(self, x):
        """Forward pass through the transformer block.

        Args:
            x (Tensor): Input tensor of shape (seq_len, batch_size, hidden_size)

        Returns:
            Tensor: Output tensor of same shape as input.
            Tensor or None: Attention weights if vis=True, else None.
        """
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        """Load pretrained weights for this block from a dict.

        Args:
            weights (dict): State dict of weights.
            n_block (int): Index of this block in the transformer.
        """
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = (
                np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )
            key_weight = (
                np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )
            value_weight = (
                np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )
            out_weight = (
                np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    """Transformer Encoder consisting of multiple Transformer Blocks.

    Args:
        hidden_size (int): Dimensionality of token embeddings.
        transformer_num_layers (int): Number of transformer blocks.
        transformer_mlp_dim (int): Hidden size of MLP in each block.
        transformer_dropout_rate (float): Dropout rate inside transformer blocks.
        transformer_num_heads (int): Number of attention heads.
        transformer_attention_dropout_rate (float): Dropout rate for attention.
        vis (bool): If True, collect and return attention weights from all blocks.

    Forward input:
        hidden_states (Tensor): Input token embeddings, shape (seq_len, batch_size, hidden_size)

    Returns:
        Tensor: Final encoded output after last transformer block.
        List[Tensor]: List of attention weights from each block (if vis=True), else empty list.
    """

    def __init__(
        self,
        hidden_size,
        transformer_num_layers,
        transformer_mlp_dim,
        transformer_dropout_rate,
        transformer_num_heads,
        transformer_attention_dropout_rate,
        vis,
    ):
        super().__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(hidden_size, eps=1e-6)
        for _ in range(transformer_num_layers):
            layer = Block(
                hidden_size,
                transformer_mlp_dim,
                transformer_dropout_rate,
                transformer_num_heads,
                transformer_attention_dropout_rate,
                vis,
            )
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        """Forward pass through all transformer blocks.

        Args:
            hidden_states (Tensor): Input token embeddings.

        Returns:
            Tensor: Final encoded token embeddings.
            List[Tensor]: Attention maps from all blocks (if vis=True), else empty.
        """
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    """Vision Transformer model with patch embedding, optional hybrid ResNet embedding, and
    transformer encoder.

    Args:
        img_size (tuple): Input image size (height, width).
        hidden_size (int): Dimensionality of token embeddings.
        in_channels (int): Number of input channels (e.g. 3 for RGB).
        patches_grid (tuple): Number of patches (height, width).
        patches_size (tuple): Size of each patch.
        resnet_num_layers (tuple): ResNet layers used in hybrid embedding (if any).
        resnet_width_factor (int): Width multiplier for ResNet (if hybrid).
        dropout_rate (float): Dropout rate in embeddings.
        transformer_num_layers (int): Number of transformer blocks.
        transformer_mlp_dim (int): Hidden size in MLP layers.
        transformer_dropout_rate (float): Dropout rate in transformer.
        transformer_num_heads (int): Number of attention heads.
        transformer_attention_dropout_rate (float): Attention dropout rate.
        vis (bool): Whether to return attention maps for visualization.

    Forward input:
        x (Tensor): Input images, shape (batch_size, in_channels, height, width).

    Returns:
        Tensor: Encoded tokens from transformer.
        List[Tensor]: Attention maps if vis=True, else empty list.
    """

    def __init__(
        self,
        img_size,
        hidden_size,
        in_channels,
        patches_grid,
        patches_size,
        resnet_num_layers,
        resnet_width_factor,
        dropout_rate,
        transformer_num_layers,
        transformer_mlp_dim,
        transformer_dropout_rate,
        transformer_num_heads,
        transformer_attention_dropout_rate,
        vis,
    ):
        super().__init__()
        self.embeddings = Embeddings(
            img_size,
            hidden_size,
            in_channels,
            patches_grid,
            patches_size,
            resnet_num_layers,
            resnet_width_factor,
            dropout_rate,
        )
        self.encoder = Encoder(
            hidden_size,
            transformer_num_layers,
            transformer_mlp_dim,
            transformer_dropout_rate,
            transformer_num_heads,
            transformer_attention_dropout_rate,
            vis,
        )

    def forward(self, input_ids):
        """Forward pass for the Vision Transformer.

        Args:
            x (Tensor): Input images of shape (batch_size, in_channels, height, width).

        Returns:
            Tensor: Transformer output embeddings.
            List[Tensor]: Attention maps if vis=True.
        """
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super().__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    """Initialize a Decoder Block with upsampling and convolution layers.

    Args:
        in_channels (int): Number of input channels to the block.
        out_channels (int): Number of output channels after convolutions.
        skip_channels (int): Number of channels from the skip connection to concatenate.
        use_batchnorm (bool): Whether to use batch normalization after convolutions.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels=0,
        use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        """Forward pass for the Decoder Block.

        Args:
            x (Tensor): Input feature map tensor to upsample.
            skip (Tensor, optional): Feature map tensor from encoder for skip connection.

        Returns:
            Tensor: Output feature map after upsampling and convolutions.
        """
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):
    """Segmentation head that applies a conv layer and optional upsampling.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels (number of classes)
        kernel_size (int): Kernel size of the conv layer
        upsampling (int): Upsampling factor (>1 means upsampling is applied)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        )
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    """Decoder module that processes transformer hidden states into image-like feature maps.

    Args:
        hidden_size (int): Hidden size of the transformer output
        decoder_channels (tuple): Channels for each decoder block output
        n_skip (int): Number of skip connections to use
        skip_channels (list): Channels from skip connections to concatenate
    """

    def __init__(self, hidden_size, decoder_channels, n_skip, skip_channels):
        super().__init__()
        self.n_skip = n_skip
        head_channels = 512
        self.conv_more = Conv2dReLU(
            hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if n_skip != 0:
            skip_channels = skip_channels
            for i in range(4 - n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3 - i] = 0

        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch)
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        """Forward pass of the decoder.

        Args:
            hidden_states (tensor): Transformer output of shape (B, n_patch, hidden_size)
            features (list, optional): List of skip connection feature maps from encoder
        Returns:
            tensor: decoded feature map
        """
        (
            B,
            n_patch,
            hidden,
        ) = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class VisionTransformer(nn.Module):
    """Initialize Vision Transformer model for image segmentation.

    Args:
        img_size (tuple): Input image size (height, width).
        in_channels (int): Number of input image channels (e.g., 3 for RGB).
        num_classes (int): Number of output segmentation classes.
        patch_size (int): Size of the patches the image is divided into.
        patches_grid (tuple): Number of patches in height and width.
        hidden_size (int): Dimension of hidden transformer embeddings.
        resnet_num_layers (tuple): Number of layers in the ResNet backbone if used.
        resnet_width_factor (int): Width multiplier for ResNet channels.
        dropout_rate (float): Dropout rate applied to embeddings.
        transformer_num_layers (int): Number of transformer encoder blocks.
        transformer_mlp_dim (int): Dimension of the MLP layers inside transformer.
        transformer_dropout_rate (float): Dropout rate in transformer MLP.
        transformer_num_heads (int): Number of attention heads.
        transformer_attention_dropout_rate (float): Dropout on attention weights.
        n_skip (int): Number of skip connections used in decoder.
        skip_channels (list): Channels in skip connection feature maps.
        classifier (str): Type of classifier head, default is 'seg' for segmentation.
        decoder_channels (tuple): Channels in decoder blocks.
        zero_head (bool): Whether to zero-initialize the classification head.
        vis (bool): Whether to output attention maps for visualization.
        pretrained_path (str): Path to pretrained weights file.
    """

    def __init__(
        self,
        img_size=(224, 224),
        in_channels=3,
        num_classes=4,
        patch_size=16,
        patches_grid=(16, 16),
        hidden_size=768,
        resnet_num_layers=(3, 4, 9),
        resnet_width_factor=1,
        dropout_rate=0.1,
        transformer_num_layers=12,
        transformer_mlp_dim=3072,
        transformer_dropout_rate=0.1,
        transformer_num_heads=12,
        transformer_attention_dropout_rate=0.0,
        n_skip=3,
        skip_channels=[512, 256, 64, 16],
        classifier="seg",
        decoder_channels=(256, 128, 64, 16),
        zero_head=False,
        vis=False,
        pretrained_path="../pretrained/R50+ViT-B_16.npz",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = classifier
        self.pretrained_path = pretrained_path
        patches_size = (patch_size, patch_size)
        self.transformer = Transformer(
            img_size,
            hidden_size,
            in_channels,
            patches_grid,
            patches_size,
            resnet_num_layers,
            resnet_width_factor,
            dropout_rate,
            transformer_num_layers,
            transformer_mlp_dim,
            transformer_dropout_rate,
            transformer_num_heads,
            transformer_attention_dropout_rate,
            vis,
        )
        self.decoder = DecoderCup(hidden_size, decoder_channels, n_skip, skip_channels)
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=3,
        )

    def forward(self, x):
        """Forward pass of Vision Transformer model.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Output segmentation logits of shape (B, num_classes, H, W).
        """
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

    def load_from(self):
        """Load pretrained weights from .npz file into the transformer and ResNet backbone."""
        with torch.no_grad():
            weights = np.load(self.pretrained_path)

            res_weight = weights

            self.transformer.embeddings.patch_embeddings.weight.copy_(
                np2th(weights["embedding/kernel"], conv=True)
            )
            self.transformer.embeddings.patch_embeddings.bias.copy_(
                np2th(weights["embedding/bias"])
            )

            self.transformer.encoder.encoder_norm.weight.copy_(
                np2th(weights["Transformer/encoder_norm/scale"])
            )
            self.transformer.encoder.encoder_norm.bias.copy_(
                np2th(weights["Transformer/encoder_norm/bias"])
            )

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings

            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print(f"load_pretrained: grid-size from {gs_old} to {gs_new}")
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(res_weight["conv_root/kernel"], conv=True)
                )
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)
