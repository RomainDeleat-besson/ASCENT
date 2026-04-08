# Unetr++ model based on "UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation" https://github.com/Amshaker/unetr_plus_plus
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.blocks import Convolution
from monai.transforms import (
    EnsureChannelFirstd,
    EnsureTyped,
    RandRotate90d,
    RandSpatialCropd,
    ResizeWithPadOrCropd,
    ScaleIntensityd,
    SpatialPadd,
    SqueezeDimd,
)
from timm.layers import to_ntuple
from timm.models.layers import DropPath, trunc_normal_
from torch import nn

from src.data.datamodule import DataModule
from src.utils.transforms import LoadNpyd


class ContiguousGrad(torch.autograd.Function):
    """A custom autograd function to ensure that gradients are contiguous during the backward pass.

    This class provides a way to maintain contiguous gradients in PyTorch, which can be important
    for performance and memory layout in certain operations.
    """

    @staticmethod
    def forward(ctx, x):
        """Forward pass of the custom function.

        Args:
            ctx (torch.autograd.function.Context): The context object that can be used to
                                                    stash information for backward computation.
            x (Tensor): Input tensor of shape (N, ...).

        Returns:
            Tensor: Output tensor, identical to the input tensor.
        """
        return x

    @staticmethod
    def backward(ctx, grad_out):
        """Backward pass of the custom function.

        Args:
            ctx (torch.autograd.function.Context): The context object used in the forward pass.
            grad_out (Tensor): Gradient of the loss with respect to the output tensor.

        Returns:
            Tensor: Gradient of the loss with respect to the input tensor, ensuring it is contiguous.
        """
        return grad_out.contiguous()


class Mlp(nn.Module):
    """Multilayer Perceptron (MLP) module for feedforward neural network operations.

    This class implements a two-layer fully connected network with an optional
    activation function and dropout. The MLP can be used as a building block in
    larger neural network architectures.

    Args:
        in_features (int): Number of input features (dimensionality of input data).
        hidden_features (int, optional): Number of features in the hidden layer.
                                          If None, set to in_features. Defaults to None.
        out_features (int, optional): Number of output features. If None, set to in_features.
                                       Defaults to None.
        act_layer (callable, optional): Activation function to apply after the first
                                         linear layer. Defaults to nn.GELU.
        drop (float, optional): Dropout probability after each linear layer. Defaults to 0.0.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        act (callable): Activation function.
        fc2 (nn.Linear): Second fully connected layer.
        drop (nn.Dropout): Dropout layer.
    """

    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Forward pass through the MLP.

        Args:
            x (Tensor): Input tensor of shape (B, in_features), where B is the batch size.

        Returns:
            Tensor: Output tensor of shape (B, out_features) after applying the MLP.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """Partition the input tensor into non-overlapping windows.

    Args:
        x (Tensor): Input tensor of shape (B, C, D, H, W) or (B, C, H, W)
                    where B is the batch size, C is the number of channels,
                    and D, H, W are the spatial dimensions.
        window_size (list[int] or tuple[int]): The size of each window in
                                                each spatial dimension.

    Returns:
        Tensor: A tensor of shape (num_windows, window_size[0], window_size[1], ..., C),
                where num_windows is the total number of windows created.
    """
    # Get the shape of the input tensor
    B = x.shape[0]
    spatial_dims = x.shape[1:-1]  # All dimensions except batch and channel
    C = x.shape[-1]

    # Calculate the number of windows in each dimension and reshape accordingly
    num_windows = []
    for dim, size in zip(spatial_dims, window_size):
        num_windows += [dim // size, size]
    new_shape = [B] + num_windows + [C]

    # Reshape and permute to bring the window dimensions together
    x = x.view(*new_shape)
    permute_order = (
        [0]
        + list(range(1, len(new_shape) - 1, 2))
        + list(range(2, len(new_shape) - 1, 2))
        + [len(new_shape) - 1]
    )
    windows = x.permute(permute_order).contiguous().view(-1, *window_size, C)

    return windows


def window_reverse(windows, window_size, *spatial_dims):
    """Reverse the windowing operation and reconstruct the original tensor.

    Args:
        windows (Tensor): A tensor of shape (num_windows, window_size[0], window_size[1], ..., C),
                          where num_windows is the total number of windows created.
        window_size (list[int] or tuple[int]): The size of each window in each spatial dimension.
        spatial_dims (int): The original spatial dimensions before partitioning.

    Returns:
        Tensor: The reconstructed tensor of shape (B, C, D, H, W) or (B, C, H, W),
                matching the shape of the input tensor to `window_partition`.
    """
    if len(window_size) == 3:
        B = int(
            windows.shape[0]
            / (np.prod(spatial_dims) / window_size[0] / window_size[1] / window_size[2])
        )
    else:
        B = int(windows.shape[0] / (np.prod(spatial_dims) / window_size[0] / window_size[1]))

    # Calculate the original shape
    original_shape = (
        [B]
        + [dim // win for dim, win in zip(spatial_dims, window_size)]
        + list(window_size)
        + [-1]
    )
    x = windows.view(*original_shape)

    # Permute to restore the original dimensions
    permute_order = [0]
    for i in range(1, len(spatial_dims) + 1):
        permute_order += [i, i + len(spatial_dims)]
    permute_order += [len(original_shape) - 1]
    x = x.permute(permute_order).contiguous().view(B, *spatial_dims, -1)

    return x


class SwinTransformerBlock_kv(nn.Module):
    """Swin Transformer Block with custom key-value attention. This block operates on partitioned
    windows and includes a multi-head self-attention mechanism, along with an MLP block and
    residual connections.

    Args:
        dim (int): Dimension of the input feature (number of channels).
        input_resolution (tuple): Spatial resolution of the input (height, width, depth, etc.).
        num_heads (int): Number of attention heads in the multi-head self-attention.
        window_size (tuple): Size of the window for attention (e.g., (7, 7) for 2D).
        shift_size (int, optional): Shift size for shifted windows in shifted-window attention. Defaults to 0.
        mlp_ratio (float, optional): Ratio of the hidden dimension to the input dimension in the MLP. Defaults to 4.0.
        qkv_bias (bool, optional): Whether to add a learnable bias to Q, K, V projections. Defaults to True.
        qk_scale (float, optional): Custom scaling factor for QK projection. Defaults to None.
        drop (float, optional): Dropout rate applied to attention output and MLP. Defaults to 0.0.
        attn_drop (float, optional): Dropout rate applied within the attention layer. Defaults to 0.0.
        drop_path (float, optional): Drop path rate for stochastic depth. Defaults to 0.0.
        act_layer (nn.Module, optional): Activation function used in the MLP. Defaults to nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer used before attention and MLP. Defaults to nn.LayerNorm.

    Returns:
        Tensor: Transformed input tensor after applying attention, MLP, and residual connections.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        # Dynamically set shift size based on dimensions
        if tuple(self.input_resolution) == tuple(self.window_size):
            self.shift_size = [0] * len(self.window_size)  # Adapt for any number of dimensions

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention_kv(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

    def forward(self, x, mask_matrix, skip=None, x_up=None):
        """Forward pass for the Swin Transformer Block.

        Args:
            x (Tensor): Input tensor of shape (B, L, C), where B is batch size, L is sequence length (spatial dimensions),
                        and C is the number of channels (feature dimension).
            mask_matrix (Tensor): Attention mask matrix to mask out certain areas during attention.
            skip (Tensor): Skip connection tensor, used as one of the inputs for attention.
            x_up (Tensor): Upsampled input tensor, combined with the skip connection for attention.

        Returns:
            Tensor: Output tensor after applying attention, MLP, and residual connections.
        """
        assert self.shift_size == [0] * len(self.input_resolution)
        B, L, C = x.shape
        spatial_dims = self.input_resolution
        num_spatial_dims = len(spatial_dims)

        # Check if the total length matches the product of spatial dimensions
        assert L == np.prod(spatial_dims), "input feature has wrong size"

        shortcut = x
        skip = self.norm1(skip)
        x_up = self.norm1(x_up)

        # Reshape the input tensors based on the number of spatial dimensions
        skip = skip.view(B, *spatial_dims, C)
        x_up = x_up.view(B, *spatial_dims, C)

        # Pad the feature map to multiples of window size
        padding = [
            (self.window_size[i] - spatial_dims[i] % self.window_size[i]) % self.window_size[i]
            for i in range(len(spatial_dims))
        ]

        selected_pad = [0, 0]
        for pad in reversed(padding):
            selected_pad += [0, pad]

        # Apply padding
        skip = F.pad(skip, selected_pad)
        skip_spatial_dims = skip.shape[1:-1]

        x_up = F.pad(x_up, selected_pad)

        # Partition windows
        skip = window_partition(skip, self.window_size)
        skip = skip.view(-1, torch.prod(torch.tensor(self.window_size)), C)
        x_up = window_partition(x_up, self.window_size)
        x_up = x_up.view(-1, torch.prod(torch.tensor(self.window_size)), C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(skip, x_up)

        # Merge windows back
        attn_windows = attn_windows.view(-1, *self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, *skip_spatial_dims)

        # Reverse cyclic shift
        if min(self.shift_size) > 0:
            shifts = tuple(self.shift_size[:num_spatial_dims])
            dims = tuple(range(1, 1 + num_spatial_dims))
            x = torch.roll(shifted_x, shifts=shifts, dims=dims)
        else:
            x = shifted_x

        # Remove padding
        if any(pad > 0 for pad in padding):
            slices = [slice(None)] + [slice(0, dim) for dim in spatial_dims] + [slice(None)]
            x = x[slices].contiguous()

        x = x.view(B, L, C)

        # Apply MLP
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class WindowAttention_kv(nn.Module):
    """Window-based Multi-head Self-Attention (MHSA) with separate key-value projections. This
    class is used to perform attention on small windows of the input tensor, which allows for a
    more localized attention mechanism.

    Args:
        dim (int): The input feature dimension (number of channels).
        window_size (tuple): The size of the window (e.g., (7, 7) for 2D).
        num_heads (int): Number of attention heads in the multi-head self-attention mechanism.
        qkv_bias (bool, optional): Whether to use bias for query, key, and value projections. Default: True.
        qk_scale (float, optional): Custom scaling factor for the query-key dot product. Default: None.
        attn_drop (float, optional): Dropout rate for attention probabilities. Default: 0.0.
        proj_drop (float, optional): Dropout rate for the output projection. Default: 0.0.

    Returns:
        Tensor: The output tensor after applying key-value attention within windows.
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(np.prod([2 * size - 1 for size in window_size]), num_heads)
        )

        # Get pair-wise relative position index for each token inside the window
        coords = [torch.arange(ws) for ws in window_size]
        coords = torch.stack(torch.meshgrid(coords))  # Meshgrid for all dimensions
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        # Shift relative coordinates to start from 0
        for i in range(len(window_size)):
            relative_coords[:, :, i] += window_size[i] - 1

        # Calculate strides for flattened index
        if len(window_size) == 3:
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (
                2 * self.window_size[2] - 1
            )
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        else:
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        # Layers for key-value projection
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Initialize relative position bias table
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, skip, x_up, mask=None):
        """Forward pass for windowed key-value self-attention.

        Args:
            skip (Tensor): The skip connection tensor of shape (B, N, C).
            x_up (Tensor): The upsampled input tensor of shape (B, N, C).
            mask (Tensor, optional): The attention mask tensor. Default: None.

        Returns:
            Tensor: The output tensor after attention and projection.
        """
        B_, N, C = skip.shape

        kv = self.kv(skip)
        q = x_up

        kv = (
            kv.reshape(B_, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        q = q.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(np.prod(self.window_size), np.prod(self.window_size), -1)

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowAttention(nn.Module):
    """Standard window-based Multi-head Self-Attention (MHSA) mechanism where queries, keys, and
    values are derived from the same input tensor.

    Args:
        dim (int): Input feature dimension (number of channels).
        window_size (tuple): Window size (e.g., (7, 7) for 2D).
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): Bias for query, key, and value projections. Default: True.
        qk_scale (float, optional): Scale factor for query-key dot product. Default: None.
        attn_drop (float, optional): Dropout rate for attention probabilities. Default: 0.0.
        proj_drop (float, optional): Dropout rate for output projection. Default: 0.0.

    Returns:
        Tensor: The output tensor after self-attention and projection.
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(np.prod([2 * size - 1 for size in window_size]), num_heads)
        )

        # Get pair-wise relative position index for each token inside the window
        coords = [torch.arange(ws) for ws in window_size]
        coords = torch.stack(torch.meshgrid(coords))  # Meshgrid for all dimensions
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        # Shift relative coordinates to start from 0
        for i in range(len(window_size)):
            relative_coords[:, :, i] += window_size[i] - 1

        # Calculate strides for flattened index
        if len(window_size) == 3:
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (
                2 * self.window_size[2] - 1
            )
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        else:
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, pos_embed=None):
        """Forward pass of the window-based Multi-head Self-Attention (MHSA).

        Args:
            x (Tensor): Input tensor of shape (B, N, C).
            mask (Tensor, optional): Attention mask to apply to certain areas (e.g., to prevent attending to padding tokens). Default: None.

        Returns:
            Tensor: Output tensor after applying self-attention, with the same shape as the input (B, N, C).
        """
        B_, N, C = x.shape

        qkv = self.qkv(x)

        qkv = (
            qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1).contiguous()
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(np.prod(self.window_size), np.prod(self.window_size), -1)

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()
        if pos_embed is not None:
            x = x + pos_embed
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Initializes a Swin Transformer block that combines window-based self-attention and an MLP
    block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple): The spatial resolution (e.g., (H, W)) of the input feature map.
        num_heads (int): Number of attention heads in multi-head self-attention.
        window_size (int or tuple, optional): Size of the local window for window-based self-attention. Default is 7.
        shift_size (int, optional): Shift size for shifted windows in Shifted Window MSA. Default is 0.
        mlp_ratio (float, optional): Ratio of hidden dimension to input dimension in the MLP. Default is 4.0.
        qkv_bias (bool, optional): If True, apply a bias term to the QKV projections. Default is True.
        qk_scale (float or None, optional): Scaling factor for QK dot product attention. Default is None.
        drop (float, optional): Dropout rate applied to the output of attention and MLP layers. Default is 0.0.
        attn_drop (float, optional): Dropout rate for the attention scores. Default is 0.0.
        drop_path (float, optional): Stochastic depth drop path rate. Default is 0.0.
        act_layer (nn.Module, optional): Activation function used in the MLP. Default is nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer applied before attention and MLP. Default is nn.LayerNorm.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.input_resolution = input_resolution  # This can be of arbitrary dimensionality
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # Handle the case where the input resolution is smaller than the window size
        if tuple(self.input_resolution) == tuple(self.window_size):
            self.shift_size = [0] * len(self.input_resolution)

        self.norm1 = norm_layer(dim)

        # Initialize the attention mechanism for arbitrary dimensions
        self.attn = WindowAttention(
            dim,
            window_size=to_ntuple(len(self.window_size))(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

    def forward(self, x, mask_matrix):
        """Forward pass through the Swin Transformer block.

        Args:
            x (Tensor): Input tensor of shape (B, L, C), where:
                        - B is the batch size,
                        - L is the sequence length (flattened spatial dimensions),
                        - C is the number of input channels/features.

        Returns:
            Tensor: Output tensor of shape (B, L, C), after applying window-based self-attention and the MLP.
        """
        B, L, C = x.shape
        spatial_dims = self.input_resolution

        # Check that the input length matches the expected resolution
        assert L == np.prod(spatial_dims), "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, *spatial_dims, C)

        # Pad the feature map to multiples of window size
        padding = [
            (self.window_size[i] - spatial_dims[i] % self.window_size[i]) % self.window_size[i]
            for i in range(len(spatial_dims))
        ]

        selected_pad = [0, 0]
        for pad in reversed(padding):
            selected_pad += [0, pad]

        # Pad the feature map for all spatial dimensions
        x = F.pad(x, selected_pad)
        padded_shape = x.shape[1:-1]  # Get padded spatial dimensions

        # Apply cyclic shift if required
        if min(self.shift_size) > 0:
            shifted_x = torch.roll(
                x,
                shifts=[-s for s in self.shift_size],
                dims=tuple(range(1, len(spatial_dims) + 1)),
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # Partition windows for attention
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, np.prod(self.window_size), C)

        # Apply window-based attention
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows back
        attn_windows = attn_windows.view(-1, *self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, *padded_shape)

        # Reverse cyclic shift if applied
        if min(self.shift_size) > 0:
            x = torch.roll(
                shifted_x, shifts=self.shift_size, dims=tuple(range(1, len(spatial_dims) + 1))
            )
        else:
            x = shifted_x

        if any(pad > 0 for pad in padding):
            slices = [slice(None)] + [slice(0, dim) for dim in spatial_dims] + [slice(None)]
            x = x[slices].contiguous()

        x = x.view(B, L, C)

        # Add residual connection and apply feed-forward network
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer.

    This layer reduces the spatial dimensions of the input tensor and doubles the channel dimensions.

    Args:
        num_spatial_dims (int): Number of spatial dimensions (e.g., 3 for 3D images, 2 for 2D images).
        dim (int): Number of input channels/features.
        kernel (int or tuple): The size of the kernel for convolution.
        stride (int or tuple): Stride for the convolution operation.
        padding (int or tuple): Padding added to all sides of the input.
        norm_layer (nn.Module, optional): Normalization layer applied after feature extraction. Default is nn.LayerNorm.
    """

    def __init__(self, num_spatial_dims, dim, kernel, stride, padding, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = Convolution(
            spatial_dims=num_spatial_dims,
            in_channels=dim,
            out_channels=dim * 2,
            kernel_size=kernel,
            strides=stride,
            padding=padding,
        )

        self.norm = norm_layer(dim)

    def forward(self, x, *spatial_dims):
        """Forward pass of the Patch Merging layer.

        Args:
            x (Tensor): Input tensor of shape (B, L, C), where:
                        - B is the batch size,
                        - L is the sequence length (flattened spatial dimensions),
                        - C is the number of input channels.
            spatial_dims (tuple): The original spatial dimensions of the input (e.g., (H, W) for 2D).

        Returns:
            Tensor: The output tensor with reduced spatial dimensions and doubled number of channels.
        """
        B, L, C = x.shape
        assert L == np.prod(spatial_dims), "input feature has wrong size"
        x = x.view(B, *spatial_dims, C)

        x = F.gelu(x)
        x = self.norm(x)

        permute_order = (0,) + (len(spatial_dims) + 1,) + tuple(range(1, len(spatial_dims) + 1))
        x = x.permute(permute_order)  # Move channel dimension to the end

        x = self.reduction(x)

        permute_order = (0,) + tuple(range(2, len(spatial_dims) + 2)) + (1,)
        x = x.permute(permute_order).view(B, -1, 2 * C)
        return x


class PatchExpanding(nn.Module):
    """Patch Expanding Layer.

    This layer upsamples the spatial dimensions of the input tensor and reduces the number of channels by half.

    Args:
        num_spatial_dims (int): Number of spatial dimensions (e.g., 3 for 3D images, 2 for 2D images).
        dim (int): Number of input channels/features.
        kernel (int or tuple): The size of the kernel for transposed convolution.
        stride (int or tuple): Stride for the transposed convolution.
        padding (int or tuple): Padding applied for the output tensor during the upsampling.
        norm_layer (nn.Module, optional): Normalization layer applied before upsampling. Default is nn.LayerNorm.
    """

    def __init__(self, num_spatial_dims, dim, kernel, stride, padding, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.up = Convolution(
            spatial_dims=num_spatial_dims,
            in_channels=dim,
            out_channels=dim // 2,
            kernel_size=kernel,
            strides=stride,
            output_padding=padding,
            is_transposed=True,
            padding=0,
            norm=None,
            act=None,
        )

    def forward(self, x, *spatial_dims):
        """Forward pass of the Patch Expanding layer.

        Args:
            x (Tensor): Input tensor of shape (B, L, C), where:
                        - B is the batch size,
                        - L is the sequence length (flattened spatial dimensions),
                        - C is the number of input channels.
            spatial_dims (tuple): The original spatial dimensions of the input (e.g., (H, W) for 2D).

        Returns:
            Tensor: The output tensor with increased spatial dimensions and halved number of channels.
        """
        B, L, C = x.shape
        assert L == np.prod(spatial_dims), "input feature has wrong size"
        x = x.view(B, *spatial_dims, C)

        x = self.norm(x)

        permute_order = (0,) + (len(spatial_dims) + 1,) + tuple(range(1, len(spatial_dims) + 1))
        x = x.permute(permute_order)  # Move channel dimension to the end

        x = self.up(x)
        x = ContiguousGrad.apply(x)
        permute_order = (0,) + tuple(range(2, len(spatial_dims) + 2)) + (1,)
        x = x.permute(permute_order).view(B, -1, C // 2)

        return x


class BasicLayer(nn.Module):
    """A basic layer for the Swin Transformer model, which includes multiple Swin Transformer
    blocks and an optional downsampling (patch merging) layer.

    Args:
        dim (int): Number of input channels/features.
        input_resolution (tuple): Input resolution (spatial dimensions).
        depth (int): Number of Swin Transformer blocks in the layer.
        num_heads (int): Number of attention heads in the multi-head self-attention.
        window_size (tuple): Size of the attention window for each dimension.
        down_kernel (int or tuple): Kernel size for the downsampling layer (if used).
        down_stride (int or tuple): Stride size for the downsampling layer (if used).
        down_padding (int or tuple): Padding for the downsampling layer (if used).
        mlp_ratio (float, optional): Ratio of hidden dimension in the MLP to input dimension. Default is 4.0.
        qkv_bias (bool, optional): If True, adds bias to the query, key, and value projections. Default is True.
        qk_scale (float or None, optional): Custom scale factor for the query-key dot product. Default is None.
        drop (float, optional): Dropout rate for the output projection of the attention layer. Default is 0.0.
        attn_drop (float, optional): Dropout rate for attention weights. Default is 0.0.
        drop_path (float or list of floats, optional): Stochastic depth rate. If a list, it should have the same length as the number of blocks. Default is 0.0.
        norm_layer (nn.Module, optional): Normalization layer. Default is nn.LayerNorm.
        downsample (nn.Module or None, optional): Downsampling layer (e.g., PatchMerging). If None, no downsampling is applied. Default is None.
        i_layer (int or None, optional): Layer index for identification. Default is None.

    Attributes:
        blocks (nn.ModuleList): List of SwinTransformerBlock modules to process the input.
        downsample (nn.Module or None): Downsampling layer (Patch Merging) applied after the blocks, if specified.
        window_size (tuple): Size of the attention window for each dimension.
        shift_size (tuple): Size of the shift for window partitioning in the attention mechanism.
        depth (int): Number of Swin Transformer blocks.
        stride (tuple): Stride for the downsampling layer.
        padding (tuple): Padding for the downsampling layer.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        down_kernel,
        down_stride,
        down_padding,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        i_layer=None,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = [window_size[i] // 2 for i in range(len(window_size))]
        self.depth = depth
        self.i_layer = i_layer
        self.stride = down_stride
        self.padding = down_padding
        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=[0] * len(window_size) if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                num_spatial_dims=len(window_size),
                dim=dim,
                norm_layer=norm_layer,
                kernel=down_kernel,
                stride=down_stride,
                padding=down_padding,
            )
        else:
            self.downsample = None

    def forward(self, x, *spatial_dims):
        """Forward pass through the BasicLayer.

        Args:
            x (Tensor): Input tensor of shape (B, L, C), where:
                        - B is the batch size,
                        - L is the sequence length (flattened spatial dimensions),
                        - C is the number of input channels/features.
            spatial_dims (tuple): The original spatial dimensions of the input (e.g., (H, W) for 2D or (D, H, W) for 3D).

        Returns:
            Tensor: The output tensor after Swin Transformer blocks and optional downsampling.
            new_spatial_dims (tuple): The spatial dimensions after downsampling (if downsampling is applied).
        """
        attn_mask = None
        for blk in self.blocks:
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, *spatial_dims)
            new_spatial_dims = [
                (spatial_dims[i] + self.padding[i]) // self.stride[i]
                for i in range(len(spatial_dims))
            ]
            return x, *new_spatial_dims, x_down
        else:
            return x, *spatial_dims, x


class BasicLayer_up(nn.Module):
    """A basic layer for the decoder part of the Swin Transformer, which includes multiple Swin
    Transformer blocks and an upsampling layer. It also incorporates skip connections from the
    encoder part.

    Args:
        dim (int): Number of input channels/features.
        input_resolution (tuple): Input resolution (spatial dimensions).
        depth (int): Number of Swin Transformer blocks in the layer.
        num_heads (int): Number of attention heads in the multi-head self-attention.
        window_size (tuple): Size of the attention window for each dimension.
        up_kernel (int or tuple): Kernel size for the upsampling layer.
        up_stride (int or tuple): Stride size for the upsampling layer.
        up_padding (int or tuple): Padding for the upsampling layer.
        mlp_ratio (float, optional): Ratio of hidden dimension in the MLP to input dimension. Default is 4.0.
        qkv_bias (bool, optional): If True, adds bias to the query, key, and value projections. Default is True.
        qk_scale (float or None, optional): Custom scale factor for the query-key dot product. Default is None.
        drop (float, optional): Dropout rate for the output projection of the attention layer. Default is 0.0.
        attn_drop (float, optional): Dropout rate for attention weights. Default is 0.0.
        drop_path (float or list of floats, optional): Stochastic depth rate. If a list, it should have the same length as the number of blocks. Default is 0.0.
        norm_layer (nn.Module, optional): Normalization layer. Default is nn.LayerNorm.
        upsample (nn.Module): The upsampling layer (e.g., ConvTranspose) for upsampling features.
        i_layer (int or None, optional): Layer index for identification. Default is None.

    Attributes:
        blocks (nn.ModuleList): List of Swin Transformer blocks to process the input.
        Upsample (nn.Module): The upsampling layer to increase spatial resolution.
        window_size (tuple): Size of the attention window for each dimension.
        shift_size (tuple): Size of the shift for window partitioning in the attention mechanism.
        depth (int): Number of Swin Transformer blocks.
        stride (tuple): Stride for the upsampling layer.
        padding (tuple): Padding for the upsampling layer.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        up_kernel,
        up_stride,
        up_padding,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        upsample=None,
        i_layer=None,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = [window_size[i] // 2 for i in range(len(window_size))]
        self.depth = depth
        self.stride = up_stride
        self.padding = up_padding

        # build blocks
        self.blocks = nn.ModuleList()
        self.blocks.append(
            SwinTransformerBlock_kv(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=[0] * len(window_size),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
        )
        for i in range(depth - 1):
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i + 1] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
            )

        self.i_layer = i_layer
        self.Upsample = upsample(
            num_spatial_dims=len(window_size),
            dim=2 * dim,
            norm_layer=norm_layer,
            kernel=up_kernel,
            stride=up_stride,
            padding=up_padding,
        )

    def forward(self, x, skip, *spatial_dims):
        """Forward pass through the BasicLayer_up.

        Args:
            x (Tensor): Input tensor of shape (B, L, C), where:
                        - B is the batch size,
                        - L is the sequence length (flattened spatial dimensions),
                        - C is the number of input channels/features.
            skip (Tensor): Skip connection from the corresponding encoder layer.
            spatial_dims (tuple): The original spatial dimensions of the input (e.g., (H, W) for 2D or (D, H, W) for 3D).

        Returns:
            Tensor: The output tensor after Swin Transformer blocks and upsampling.
            spatial_dims (tuple): The updated spatial dimensions after upsampling.
        """
        x_up = self.Upsample(x, *spatial_dims)
        x = skip + x_up
        spatial_dims = [
            (spatial_dims[i] * self.stride[i]) + self.padding[i] for i in range(len(spatial_dims))
        ]
        attn_mask = None
        x = self.blocks[0](x, attn_mask, skip=skip, x_up=x_up)
        for i in range(self.depth - 1):
            x = self.blocks[i + 1](x, attn_mask)

        return x, *spatial_dims


class project(nn.Module):
    """A convolutional projection layer that applies two convolutional layers with optional
    activation and normalization. The first convolution changes the number of channels, while the
    second one maintains the number of channels.

    Args:
        spatial_dims (int): Number of spatial dimensions (e.g., 2 for 2D images, 3 for 3D volumes).
        in_dim (int): Number of input channels/features.
        out_dim (int): Number of output channels/features after the projection.
        stride (int or tuple): Stride size for the first convolutional layer.
        padding (int or tuple): Padding size for the first convolutional layer.
        activate (callable): Activation function to be applied after the first convolution.
        norm (callable): Normalization layer (e.g., nn.BatchNorm3d or nn.LayerNorm).
        last (bool, optional): If True, skips the second activation and normalization. Default is False.

    Attributes:
        out_dim (int): Number of output channels/features after the projection.
        conv1 (nn.Module): First convolutional layer to project input channels to output channels.
        conv2 (nn.Module): Second convolutional layer to maintain output channels.
        activate (callable): Activation function to be applied after the first convolution.
        norm1 (nn.Module): Normalization layer applied after the first convolution.
        norm2 (nn.Module or None): Normalization layer applied after the second convolution if not the last layer.
    """

    def __init__(self, spatial_dims, in_dim, out_dim, stride, padding, activate, norm, last=False):
        super().__init__()
        self.out_dim = out_dim
        self.conv1 = Convolution(
            spatial_dims, in_dim, out_dim, kernel_size=3, strides=stride, padding=padding
        )
        self.conv2 = Convolution(
            spatial_dims, out_dim, out_dim, kernel_size=3, strides=1, padding=1
        )
        self.activate = activate()
        self.norm1 = norm(out_dim)
        self.last = last
        if not last:
            self.norm2 = norm(out_dim)

    def forward(self, x):
        """Forward pass through the projection layer.

        Args:
            x (Tensor): Input tensor of shape (B, C, *spatial_dims), where:
                        - B is the batch size,
                        - C is the number of input channels,
                        - *spatial_dims represents the spatial dimensions.

        Returns:
            Tensor: Output tensor after applying convolutions, activation, and normalization.
                    Shape will be (B, out_dim, *spatial_dims).
        """
        x = self.conv1(x)
        x = self.activate(x)

        # norm1
        B, C, *spatial_dims = x.size()
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm1(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, *spatial_dims)

        x = self.conv2(x)
        if not self.last:
            x = self.activate(x)
            # norm2
            B, C, *spatial_dims = x.size()
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm2(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, *spatial_dims)
        return x


class PatchEmbed(nn.Module):
    """Patch embedding layer for transforming input tensors into patches and projecting them into a
    higher-dimensional space.

    Args:
        patch_size (int or tuple): Size of the patches to be extracted from the input tensor.
        in_chans (int): Number of input channels/features in the input tensor.
        embed_dim (int, optional): Dimension of the embedding space (output channels). Default is 96.
        norm_layer (nn.Module, optional): Normalization layer to be applied after embedding. Default is None.

    Attributes:
        patch_size (tuple): Size of the patches as a tuple based on the input dimensionality.
        in_chans (int): Number of input channels/features.
        embed_dim (int): Number of output channels/features after embedding.
        proj1 (project): First projection layer to reduce input channels to half of the embedding dimension.
        proj2 (project): Second projection layer to project the output of proj1 to the final embedding dimension.
        norm (nn.Module or None): Normalization layer, if provided. Otherwise, None.

    Methods:
        forward(x): Forward pass to apply patch embedding and normalization.
    """

    def __init__(self, patch_size, in_chans, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_ntuple(len(patch_size))(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if len(self.patch_size) == 3:
            stride1 = [patch_size[0] // 2, patch_size[1] // 2, 1]
            stride2 = [patch_size[0] // 2, patch_size[1] // 2, 1]
        else:
            stride1 = [patch_size[0] // 2, patch_size[1] // 2]
            stride2 = [patch_size[0] // 2, patch_size[1] // 2]

        self.proj1 = project(
            len(patch_size), in_chans, embed_dim // 2, stride1, 1, nn.GELU, nn.LayerNorm, False
        )
        self.proj2 = project(
            len(patch_size), embed_dim // 2, embed_dim, stride2, 1, nn.GELU, nn.LayerNorm, True
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function to apply patch embedding.

        Args:
            x (Tensor): Input tensor of shape (B, C, *spatial_dims), where:
                        - B is the batch size,
                        - C is the number of input channels,
                        - *spatial_dims are the spatial dimensions.

        Returns:
            Tensor: Output tensor of shape (B, embed_dim, *new_spatial_dims) after applying projections
                    and normalization, where *new_spatial_dims depends on the patch size and input shape.
        """
        # Get input size dynamically based on the number of dimensions
        dims = x.dim()  # Number of dimensions

        # Define sizes for each dimension based on the input shape
        size_list = x.size()

        # Padding for each dimension
        for dim in range(2, dims):  # Start from 2 to skip batch and channels dimensions
            if size_list[dim] % self.patch_size[dim - 2] != 0:
                padding_size = self.patch_size[dim - 2] - size_list[dim] % self.patch_size[dim - 2]
                pad = [0] * (2 * (dims - dim))  # Create padding for the correct dimensions
                pad[-1] = padding_size  # Update the corresponding padding
                x = F.pad(x, pad)
        # Perform projections
        x = self.proj1(x)
        x = self.proj2(x)

        # Normalization
        if self.norm is not None:
            new_size_list = x.size()
            x = x.flatten(start_dim=2).transpose(1, 2).contiguous()  # Flatten spatial dimensions
            x = self.norm(x)  # Normalize
            # Reshape back to the original dimensions
            output_shape = [-1, self.embed_dim] + list(new_size_list[2:])  # B, C, (new dims)
            x = x.transpose(1, 2).contiguous().view(output_shape)
        return x


class Encoder(nn.Module):
    """Encoder module for processing input images using a series of transformer layers.

    Args:
        img_patch_size (tuple): Size of the image patches for embedding.
        mini_patch_size (tuple): Size of the mini patches for embedding.
        in_chans (int): Number of input channels in the image.
        embed_dim (int): Dimension of the embedding space (output channels).
        depths (list of int): Number of layers in each stage of the encoder.
        num_heads (list of int): Number of attention heads for each stage.
        window_size (list of int): Size of the attention windows for each stage.
        kernel_down (list of int): Kernel size for downsampling in each stage.
        stride_down (list of int): Stride size for downsampling in each stage.
        padding_down (list of int): Padding size for downsampling in each stage.
        input_reduction (list of int): Reduction factors for input resolution at each stage.
        mlp_ratio (float, optional): Ratio of the hidden dimension to input dimension in MLP. Default is 4.0.
        qkv_bias (bool, optional): Whether to add a learnable bias to query, key, and value. Default is True.
        qk_scale (float, optional): Scaling factor for the query and key. Default is None.
        drop_rate (float, optional): Dropout rate for the dropout layer. Default is 0.0.
        attn_drop_rate (float, optional): Dropout rate for attention weights. Default is 0.0.
        drop_path_rate (float, optional): Stochastic depth rate. Default is 0.2.
        norm_layer (nn.Module, optional): Normalization layer to be applied to outputs. Default is nn.LayerNorm.
        patch_norm (bool, optional): Whether to apply normalization after patch embedding. Default is True.
        out_indices (tuple, optional): Indices of layers to return outputs from. Default is (0, 1, 2, 3).

    Attributes:
        pretrain_img_size (tuple): Size of the image patches used for pretraining.
        num_layers (int): Number of layers in the encoder.
        embed_dim (int): Dimension of the embedding space.
        patch_norm (bool): Flag indicating if normalization is applied after patch embedding.
        out_indices (tuple): Indices of layers from which to output features.
        patch_embed (PatchEmbed): Patch embedding layer for initial feature extraction.
        pos_drop (nn.Dropout): Dropout layer for positional encoding.
        layers (nn.ModuleList): List of transformer layers in the encoder.
        num_features (list): Number of features at each layer after embedding.
    """

    def __init__(
        self,
        img_patch_size,
        mini_patch_size,
        in_chans,
        embed_dim,
        depths,
        num_heads,
        window_size,
        kernel_down,
        stride_down,
        padding_down,
        input_reduction,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
    ):
        super().__init__()

        self.pretrain_img_size = img_patch_size

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=mini_patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=[
                    img_patch_size[i] // input_reduction[i_layer][i]
                    for i in range(len(img_patch_size))
                ],
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                down_kernel=kernel_down[i_layer] if (i_layer < self.num_layers - 1) else None,
                down_stride=stride_down[i_layer] if (i_layer < self.num_layers - 1) else None,
                down_padding=padding_down[i_layer] if (i_layer < self.num_layers - 1) else None,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                i_layer=i_layer,
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

    def forward(self, x):
        """Forward function to process input through the encoder layers.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W, ...) where:
                        - B is the batch size,
                        - C is the number of input channels,
                        - H and W are the spatial dimensions (height and width).

        Returns:
            list of Tensor: List of output tensors from the specified layers (out_indices), each tensor
                            having shape (B, num_features[i], *new_spatial_dims) where *new_spatial_dims
                            are the spatial dimensions after processing.
        """

        x = self.patch_embed(x)
        down = []
        B, C, *old_spatial_dims = x.size()

        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.pos_drop(x)

        new_spatial_dims = old_spatial_dims

        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, *new_spatial_dims, x = layer(x, *new_spatial_dims)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)
                permute_order = (
                    (0,)
                    + (len(old_spatial_dims) + 1,)
                    + tuple(range(1, len(old_spatial_dims) + 1))
                )
                out = (
                    x_out.view(-1, *old_spatial_dims, self.num_features[i])
                    .permute(permute_order)
                    .contiguous()
                )
                down.append(out)
            old_spatial_dims = new_spatial_dims
        return down


class Decoder(nn.Module):
    """Decoder module for reconstructing images from feature representations using a series of
    transformer layers.

    Args:
        img_patch_size (tuple): Size of the image patches for embedding.
        embed_dim (int): Dimension of the embedding space (output channels).
        depths (list of int): Number of layers in each stage of the decoder.
        num_heads (list of int): Number of attention heads for each stage.
        window_size (list of int): Size of the attention windows for each stage.
        input_reduction (list of int): Reduction factors for input resolution at each stage.
        kernel_up (list of int): Kernel size for upsampling in each stage.
        stride_up (list of int): Stride size for upsampling in each stage.
        padding_up (list of int): Padding size for upsampling in each stage.
        mlp_ratio (float, optional): Ratio of the hidden dimension to input dimension in MLP. Default is 4.0.
        qkv_bias (bool, optional): Whether to add a learnable bias to query, key, and value. Default is True.
        qk_scale (float, optional): Scaling factor for the query and key. Default is None.
        drop_rate (float, optional): Dropout rate for the dropout layer. Default is 0.0.
        attn_drop_rate (float, optional): Dropout rate for attention weights. Default is 0.0.
        drop_path_rate (float, optional): Stochastic depth rate. Default is 0.2.
        norm_layer (nn.Module, optional): Normalization layer to be applied to outputs. Default is nn.LayerNorm.

    Attributes:
        num_layers (int): Number of layers in the decoder.
        pos_drop (nn.Dropout): Dropout layer for positional encoding.
        layers (nn.ModuleList): List of transformer layers in the decoder.
        num_features (list): Number of features at each layer after embedding.

    Methods:
        forward(x, skips): Forward pass to process input features and skip connections through the decoder layers.
    """

    def __init__(
        self,
        img_patch_size,
        embed_dim,
        depths,
        num_heads,
        window_size,
        input_reduction,
        kernel_up,
        stride_up,
        padding_up,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.num_layers = len(depths)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers)[::-1]:
            layer = BasicLayer_up(
                dim=int(embed_dim * 2 ** (len(depths) - i_layer - 1)),
                input_resolution=[
                    img_patch_size[i] // input_reduction[i_layer][i]
                    for i in range(len(img_patch_size))
                ],
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                up_kernel=kernel_up[i_layer],
                up_stride=stride_up[i_layer],
                up_padding=padding_up[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpanding,
                i_layer=i_layer,
            )
            self.layers.append(layer)
        self.num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]

    def forward(self, x, skips):
        """Forward function to process input through the decoder layers.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W, ...) where:
                        - B is the batch size,
                        - C is the number of input channels,
                        - H and W are the spatial dimensions (height and width).
            skips (list of Tensor): List of skip connection tensors from the encoder,
                                    each tensor having shape (B, num_features[i], *spatial_dims).

        Returns:
            list of Tensor: List of output tensors from the decoder layers, each tensor having
                            shape (B, num_features[i], *spatial_dims) where *spatial_dims
                            are the spatial dimensions after processing.
        """
        outs = []
        B, C, *spatial_dims = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        for index, i in enumerate(skips):
            i = i.flatten(2).transpose(1, 2).contiguous()
            skips[index] = i
        x = self.pos_drop(x)

        for i in range(self.num_layers)[::-1]:
            layer = self.layers[i]

            x, *spatial_dims = layer(x, skips[i], *spatial_dims)
            out = x.view(-1, *spatial_dims, self.num_features[i])
            outs.append(out)
        return outs


class FinalPatchExpanding(nn.Module):
    """Final patch expansion layer that upsamples feature maps to the target number of classes.

    This layer uses a transposed convolution (deconvolution) to expand the input feature maps.
    It supports both 2D and 3D inputs.

    Args:
        dim (int): Number of input channels (features) to the layer.
        num_classes (int): Number of output channels, typically corresponding to the number of classes
                           in a segmentation task.
        patch_size (tuple): Size of the patches for the transposed convolution. This determines the
                            kernel size and stride for upsampling.

    Attributes:
        up (nn.Module): Transposed convolution layer (either `ConvTranspose2d` or `ConvTranspose3d`)
                        based on the input dimensionality.
    """

    def __init__(self, dim, num_classes, patch_size):
        super().__init__()

        if len(patch_size) == 3:
            self.up = nn.ConvTranspose3d(dim, num_classes, patch_size, stride=patch_size)
        else:
            self.up = nn.ConvTranspose2d(dim, num_classes, patch_size, stride=patch_size)

        # You can also include activation or normalization layers as needed, using MONAI's Conv layer
        # Example:
        # self.up = Conv[Conv.CONVTRANS, is_3d](dim, num_classes, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """Forward function to perform upsampling.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) for 2D or (B, C, D, H, W) for 3D,
                        where:
                        - B is the batch size,
                        - C is the number of input channels,
                        - H, W are the spatial dimensions for 2D, and
                        - D is the depth dimension for 3D.

        Returns:
            Tensor: Output tensor of shape (B, num_classes, new_H, new_W) for 2D or
                    (B, num_classes, new_D, new_H, new_W) for 3D, where `new_H`, `new_W`
                    and `new_D` depend on the transposed convolution operation.
        """
        # Check if input is 2D or 3D based on dimensions
        if x.dim() == 4:  # Batch, Channels, Height, Width (2D)
            x = x.permute(0, 3, 1, 2).contiguous()  # Change to (Batch, Channels, Height, Width)
        elif x.dim() == 5:  # Batch, Channels, Depth, Height, Width (3D)
            x = x.permute(
                0, 4, 1, 2, 3
            ).contiguous()  # Change to (Batch, Channels, Depth, Height, Width)

        x = self.up(x)
        return x


class nnFormer(nn.Module):
    """NnFormer model for image segmentation tasks, combining encoder-decoder architecture with
    patch-based processing.

    This model incorporates deep supervision to improve segmentation performance at different
    stages of the encoder-decoder structure.

    Args:
        in_channels (int): Number of input channels of the image (e.g., 1 for grayscale, 3 for RGB).
        num_classes (int): Number of output classes for the segmentation task.
        patch_size (Union[list[int], tuple[int, ...]]): Size of the patches to divide the input image.
        embedding_dim (int): Dimensionality of the embeddings produced by the model.
        depths (list[int]): List containing the number of layers for each encoder/decoder block.
        num_heads (list[int]): List containing the number of attention heads for each layer.
        mini_patch_size (Union[list[int], tuple[int, ...]]): Size of the mini patches used in projections.
        window_size (list[list[int]]): List of window sizes for the attention mechanism.
        input_reduction (list[list[int]]): List indicating how much to reduce the input spatial dimensions.
        kernel (list[list[int]]): List of kernel sizes for downsampling and upsampling operations.
        stride (list[list[int]]): List of strides for downsampling and upsampling operations.
        padding (list[list[int]]): List of padding sizes for convolutions.
        deep_supervision (bool, optional): If True, enables deep supervision at various stages. Defaults to True.

    Attributes:
        deep_supervision (bool): Indicates whether deep supervision is enabled.
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        patch_size (Union[list[int], tuple[int, ...]]): Size of the patches for processing.
        spatial_dims (int): Number of spatial dimensions based on the patch size.
        model_down (Encoder): Encoder part of the model.
        decoder (Decoder): Decoder part of the model.
        final (nn.ModuleList): List of final patch expanding layers for producing output segmentations.

    Methods:
        forward(x): Forward pass through the model, returning segmentation outputs based on the input tensor.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        patch_size: Union[list[int], tuple[int, ...]],
        embedding_dim: int,
        depths: list[int],
        num_heads: list[int],
        mini_patch_size: Union[list[int], tuple[int, ...]],
        window_size: list[list[int]],
        input_reduction: list[list[int]],
        kernel: list[list[int]],
        stride: list[list[int]],
        padding: list[list[int]],
        deep_supervision: bool = True,
    ):
        super().__init__()

        self.deep_supervision = deep_supervision
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.spatial_dims = len(patch_size)

        self.upscale_logits_ops = []
        self.upscale_logits_ops.append(lambda x: x)

        self.model_down = Encoder(
            img_patch_size=self.patch_size,
            window_size=window_size,
            embed_dim=embedding_dim,
            mini_patch_size=mini_patch_size,
            depths=depths,
            num_heads=num_heads,
            in_chans=self.in_channels,
            input_reduction=input_reduction,
            kernel_down=kernel[: len(depths) - 1],
            stride_down=stride[: len(depths) - 1],
            padding_down=padding[: len(depths) - 1],
        )
        self.decoder = Decoder(
            img_patch_size=self.patch_size,
            embed_dim=embedding_dim,
            window_size=window_size[::-1][1:],
            num_heads=num_heads[::-1][1:],
            depths=depths[::-1][1:],
            input_reduction=input_reduction[::-1][1:],
            kernel_up=kernel[len(depths) - 1 :],
            stride_up=stride[len(depths) - 1 :],
            padding_up=padding[len(depths) - 1 :],
        )

        self.final = []
        if self.deep_supervision:
            for i in range(len(depths) - 1):
                self.final.append(
                    FinalPatchExpanding(
                        embedding_dim * 2**i, num_classes, patch_size=mini_patch_size
                    )
                )

        else:
            self.final.append(
                FinalPatchExpanding(embedding_dim, num_classes, patch_size=mini_patch_size)
            )

        self.final = nn.ModuleList(self.final)

    def forward(self, x):
        """Forward function to process input through the nnFormer model.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) for 2D or (B, C, D, H, W) for 3D,
                        where B is batch size, C is number of input channels,
                        H is height, W is width, and D is depth.

        Returns:
            List[Tensor] or Tensor: If deep supervision is enabled and training, returns a list of
            segmentation outputs for each layer with deep supervision. Otherwise, returns the final
            segmentation output tensor.
        """
        seg_outputs = []
        skips = self.model_down(x)

        neck = skips[-1]

        out = self.decoder(neck, skips)

        if self.deep_supervision and self.training:
            for i in range(len(out)):
                seg_outputs.append(self.final[-(i + 1)](out[i]))

            return seg_outputs[::-1]
        else:
            seg_outputs.append(self.final[0](out[-1]))
            return seg_outputs[-1]


if __name__ == "__main__":
    patch_size = [160, 160, 14]
    in_channels = 1
    num_classes = 4
    embedding_dim = 96
    depths = [2, 2, 2, 2]
    num_heads = [3, 6, 12, 24]
    mini_patch_size = [4, 4, 1]
    window_size = [[5, 5, 3], [5, 5, 3], [10, 10, 7], [5, 5, 3]]
    input_reduction = [[4, 4, 1], [8, 8, 1], [16, 16, 2], [32, 32, 4]]
    kernel = [[3, 3, 1], [3, 3, 3], [3, 3, 3], [2, 2, 2], [2, 2, 2], [2, 2, 1]]
    stride = [[2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]]
    padding = [[1, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]]
    deep_supervision = True

    """patch_size = [160, 160]
    in_channels = 1
    num_classes = 4
    embedding_dim = 96
    depths = [2, 2, 2, 2]
    num_heads = [3, 6, 12, 24]
    mini_patch_size = [4, 4]
    window_size = [[5, 5], [5, 5], [10, 10], [5, 5]]
    input_reduction = [[4, 4], [8, 8], [16, 16], [32, 32]]
    kernel = [[3, 3], [3, 3], [3, 3], [2, 2], [2, 2], [2, 2]]
    stride = [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
    padding = [[1, 1], [1, 1], [1, 1], [0, 0], [0, 0], [0, 0]]
    deep_supervision = True"""

    model = nnFormer(
        in_channels,
        num_classes,
        patch_size,
        embedding_dim,
        depths,
        num_heads,
        mini_patch_size,
        window_size,
        input_reduction,
        kernel,
        stride,
        padding,
        deep_supervision,
    )

    load_transforms = {
        "train": {
            "data_loading": LoadNpyd(keys=["data"]),
            "channel_first": EnsureChannelFirstd(keys=["image", "label"]),
            "ensure_typed": EnsureTyped(keys=["image", "label"]),
        },
        "test": {
            "data_loading": LoadNpyd(keys=["data"]),
            "channel_first": EnsureChannelFirstd(keys=["image", "label"]),
            "ensure_typed": EnsureTyped(keys=["image", "label"]),
            "pad": ResizeWithPadOrCropd(
                keys=["image", "label"],
                spatial_size=[160, 160, 14],
                mode="constant",
                method="symmetric",
            ),
        },
    }

    patch_transform = {
        "crop": RandSpatialCropd(
            keys=["image", "label"], roi_size=[160, 160, 14], random_center=True, random_size=False
        ),
        "pad": SpatialPadd(
            keys=["image", "label"],
            spatial_size=[160, 160, 14],
            mode="constant",
            method="symmetric",
        ),
        "intensity": {"scale_intensity": ScaleIntensityd(keys="image")},
        "rotate": {
            "rand_rotate": RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1])
        },
    }

    data_keys = {"image_key": "image", "label_key": "label", "all_keys": ["image", "label"]}

    dataloader = DataModule(
        data_dir="C:/Users/goujat/Documents/thesis/LDMforCardiacMRI/data/",
        dataset_name="ACDC",
        fold=0,
        batch_size=16,
        patch_size=patch_size,
        in_channels=1,
        do_dummy_2D_data_aug=True,
        num_workers=1,
        pin_memory=True,
        test_splits=False,
        data_keys=data_keys,
        augmentation=patch_transform,
        loading=load_transforms,
        seed=0,
    )

    dataloader.prepare_data()
    dataloader.setup()

    train_dataloader = dataloader.train_dataloader()

    for check_data in train_dataloader:
        image = check_data["image"]
        # model(image)
        exit()
