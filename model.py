#!/usr/bin/env python3

import torch
from torch import nn
from functions import MyConv, MyPool, MyDense, MyReLU

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
        self.biases = nn.Parameter(torch.randn(out_maps) * (2 / out_maps)**0.5)

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
        self.biases = nn.Parameter(torch.randn(out_features) * (2 / out_features)**0.5)

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


class LeNet5(nn.Module):
    """Implementing the LeNet5 architecture using custom made layers"""
    def __init__(self, in_channels, out_features) -> None:
        """Initialize a LeNet5 model. `in_channels` is the number of channels
        in input image, and `out_features` is number of output classes."""
        super().__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.c1 = ConvLayer(in_channels, 6, 5)
        self.s2 = PoolingLayer(2, 2)
        self.c3 = ConvLayer(6, 16, 5)
        self.s4 = PoolingLayer(2, 2)
        self.c5 = ConvLayer(16, 120, 5)
        self.f6 = FCLayer(120,84)
        self.out = FCLayer(84,out_features)

    def forward(self, x):
        relu = MyReLU.apply
        x = relu(self.c1.forward(x))
        x = self.s2.forward(x)
        x = relu(self.c3.forward(x))
        x = self.s4.forward(x)
        x = relu(self.c5.forward(x)).flatten()
        x = relu(self.f6.forward(x))
        x = self.out.forward(x)
        return x
