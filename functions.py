#!/usr/bin/env python3

"""This file contains functions derived from `torch.autograd.Function`.
Using such functions, we can make use of PyTorch's inbuilt computational graph
for backpropagating weight updates, after which we can update the weights.
"""

import math
import torch
import torch.nn.functional as F

class MyReLU(torch.autograd.Function):
    """Implementation of a ReLU activation function.
    ReLUFunction.apply() method will be used for forward pass."""
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        # self.inputs = torch.Tensor(*x.shape)
        # self.inputs = self.inputs.copy_(x)
        x[x<0] = 0
        return x

    @staticmethod
    def backward(ctx, dl_dy: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors
        dl_dx = dl_dy.clone()
        try:
            dl_dx[x<0] = 0
        except NameError as e:
            print(e)
            print("Execute forward() before running backward()")
        return dl_dx


class MyConv(torch.autograd.Function):
    """Implementation of the convolution operation"""
    @staticmethod
    def forward(ctx, x: torch.Tensor, kernels: torch.Tensor, biases: torch.Tensor, k_size, stride):
        ctx.save_for_backward(x, kernels)
        ctx.k_size = k_size
        ctx.stride = stride
        out_maps = biases.shape[0]
        out_dim = (out_maps, math.floor((x.shape[1]-k_size)/stride)+1, math.floor((x.shape[2]-k_size)/stride)+1)
        out = torch.Tensor(*out_dim)

        for k in range(out.shape[0]):
            for i in range(out.shape[1]):
                for j in range(out.shape[2]):
                    ii = stride*i #strided index
                    jj = stride*j #strided index
                    out[k,i,j] = torch.sum(x[:, ii:ii+k_size, jj:jj+k_size] * kernels[k,:,:,:])
            out[k,:,:] += biases[k]
        return out

    @staticmethod
    def backward(ctx, dl_dy: torch.Tensor):
        x, w = ctx.saved_tensors
        k_size = ctx.k_size
        stride = ctx.stride
        out_channels = dl_dy.shape[0]
        in_channels = x.shape[0]

        if stride>1:
            temp = torch.zeros(out_channels, x.shape[1]-k_size+1, x.shape[2]-k_size+1)
            temp[:,::2,::2] = dl_dy
            dl_dy = temp
        dl_dy_pad = F.pad(dl_dy, (k_size-1, k_size-1, k_size-1, k_size-1))

        kernels = w.flip([2,3]) #flip along height and width

        # Input gradients
        dl_dx = torch.zeros(*x.shape)
        for k in range(out_channels):
            for i in range(dl_dx.shape[1]):
                for j in range(dl_dx.shape[2]):
                    grads = dl_dy_pad[k,:,:].repeat(in_channels,1,1)
                    dl_dx[:,i,j] += torch.sum(grads[:, i:i+k_size, j:j+k_size] * kernels[k,:,:,:], dim=(1,2))

        # Kernel gradients
        dl_dw = torch.zeros(*w.shape)
        for k in range(out_channels):
            for i in range(k_size):
                for j in range(k_size):
                    grads = dl_dy[k,:,:].repeat(in_channels,1,1)
                    dl_dw[k,:,i,j] += torch.sum(x[:, i:i+grads.shape[1], j:j+grads.shape[2]] * grads[:,:,:], dim=(1,2))

        # Bias gradients
        dl_db = dl_dy.mean(dim=(1,2))
        return dl_dx, dl_dw, dl_db, None, None


class MyPool(torch.autograd.Function):
    """Implementation of the pooling operation"""
    @staticmethod
    def forward(ctx, x:torch.Tensor, k_size, stride):
        ctx.save_for_backward(x)
        ctx.k_size = k_size
        ctx.stride = stride

        out_shape = (x.shape[0], math.floor((x.shape[1] - k_size)/stride)+1, math.floor((x.shape[2] - k_size)/stride)+1)
        out = torch.Tensor(*out_shape)
        for k in range(out_shape[0]):
            for i in range(out_shape[1]):
                for j in range(out_shape[2]):
                    ii = stride*i #strided index
                    jj = stride*j #strided index
                    out[k,i,j] = torch.max(x[k, ii:ii+k_size, jj:jj+k_size])
        return out

    @staticmethod
    def backward(ctx, dl_dy: torch.Tensor):
        x, = ctx.saved_tensors
        k_size = ctx.k_size
        stride = ctx.stride
        dl_dx = torch.zeros(*x.shape) #change this expression
        print(dl_dx)

        for k in range(dl_dy.shape[0]):
            for i in range(dl_dy.shape[1]):
                for j in range(dl_dy.shape[2]):
                    ii = stride*i #strided index
                    jj = stride*j #strided index
                    idx = x[k, ii:ii+k_size, jj:jj+k_size].argmax()
                    z = torch.zeros(k_size, k_size)
                    z[int(idx/2),int(idx%2)] = dl_dy[k,i,j]
                    dl_dx[k,ii:ii+k_size,jj:jj+k_size] += z
                    print(x[k, ii:ii+k_size, jj:jj+k_size], torch.argmax(x[k, ii:ii+k_size, jj:jj+k_size], dim=1), z)

        print(dl_dx)
        return dl_dx, None, None


class MyDense(torch.autograd.Function):
    """Implementation of the Dense layer operation (matrix multiplication)"""
    @staticmethod
    def forward(ctx, x, weights, biases):
        ctx.save_for_backward(x, weights)
        out = (weights @ x) + biases
        return out

    @staticmethod
    def backward(ctx, dl_dy: torch.Tensor):
        x, w = ctx.saved_tensors
        # dl_dx = dl_dy.clone() #change this expression
        dl_dx = torch.transpose(w, 0, 1) @ dl_dy
        dl_dw = dl_dy.reshape(dl_dy.shape+(1,)) @ x.reshape((1,)+x.shape)
        dl_db = dl_dy
        return dl_dx, dl_dw, dl_db
