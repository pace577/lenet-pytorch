#!/usr/bin/env python3

import math
import torch
from torch import nn

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
        out_dim = (self.out_maps, math.floor((x.shape[1]-self.k_size)/self.stride)+1, math.floor((x.shape[2]-self.k_size)/self.stride)+1)
        out = torch.Tensor(*out_dim)
        for k in range(out.shape[0]):
            for i in range(out.shape[1]):
                for j in range(out.shape[2]):
                    ii = self.stride*i #strided index
                    jj = self.stride*j #strided index
                    out[k,i,j] = torch.sum(x[:, ii:ii+self.k_size, jj:jj+self.k_size] * self.kernels[k,:,:,:])
            out[k,:,:] += self.biases[k]
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
        return (self.weights @ x) + self.biases


class PoolingLayer(nn.Module):
    """Implementation of a 2D pooling layer"""
    def __init__(self, k_size: int, stride: int = 1) -> None:
        """2D pooling layer. Specify the following inputs
          : k_size -> pixel length of square kernel
          : stride -> number of pixels to skip when shifting the pooling kernel
        """
        super().__init__()
        self.k_size = k_size
        self.stride = stride
        self.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of pooling layer (valid)"""
        out_shape = (x.shape[0], math.floor((x.shape[1] - self.k_size)/self.stride)+1, math.floor((x.shape[2] - self.k_size)/self.stride)+1)
        out = torch.Tensor(*out_shape)
        for k in range(out_shape[0]):
            for i in range(out_shape[1]):
                for j in range(out_shape[2]):
                    ii = self.stride*i #strided index
                    jj = self.stride*j #strided index
                    out[k,i,j] = torch.max(x[k, ii:ii+self.k_size, jj:jj+self.k_size])
        return out


class ReLUFunction(torch.autograd.Function):
    """Implementation of a ReLU activation function.
    ReLUFunction.apply() method will be used for forward pass."""
    def __init__(self) -> None:
        super().__init__()
        # self.inputs = torch.Tensor(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.inputs = torch.Tensor(*x.shape)
        self.inputs = self.inputs.copy_(x)
        x[x<0] = 0
        return x

    def backward(self, dl_dy: torch.Tensor) -> torch.Tensor:
        dl_dx = dl_dy
        try:
            dl_dx[self.inputs<0] = 0
        except NameError as e:
            print(e)
            print("Execute forward() before running backward()")
        return dl_dx
