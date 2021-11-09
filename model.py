#!/usr/bin/env python3

import math
import torch
from torch import nn
from functions import MyConv, MyPool, MyDense

class ConvLayer(nn.Module):
    def __init__(self, in_maps: int, out_maps: int, k_size: int, stride: int = 1) -> None:
        super().__init__()
        self.k_size = k_size
        self.in_maps = in_maps
        self.out_maps = out_maps
        self.stride = stride
        self.receptive_field_size = k_size**2
        fan_in = in_maps * self.receptive_field_size
        fan_out = out_maps * self.receptive_field_size
        self.kernels = nn.Parameter(torch.randn(out_maps, in_maps, k_size, k_size) * (4 / (fan_in + fan_out))**0.5) #Kaiming initialization
        self.biases = nn.Parameter(torch.randn(out_maps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implements valid convolution."""
        out = MyConv.apply(x, self.kernels, self.biases, self.k_size, self.stride)
        return out


class FCLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.randn(out_features, in_features) * (4 / (in_features + out_features))**0.5) #Kaiming initialization
        self.biases = nn.Parameter(torch.randn(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of fully connected layer"""
        out = MyDense.apply(x, self.weights, self.biases)
        return out


class PoolingLayer(nn.Module):
    """Implementation of a 2D pooling layer"""
    def __init__(self, k_size: int, stride: int = 2) -> None:
        """2D pooling layer. Specify the following inputs
          : k_size -> pixel length of square kernel
          : stride -> number of pixels to skip when shifting the pooling kernel
        """
        super().__init__()
        self.k_size = k_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of pooling layer (valid)"""
        out = MyPool.apply(x, self.k_size, self.stride)
        return out
