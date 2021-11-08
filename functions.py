#!/usr/bin/env python3

"""This file contains functions derived from `torch.autograd.Function`.
Using such functions, we can make use of PyTorch's inbuilt computational graph
for backpropagating weight updates, after which we can update the weights.
"""

import torch

class ReLUFunction(torch.autograd.Function):
    """Implementation of a ReLU activation function.
    ReLUFunction.apply() method will be used for forward pass."""
    def __init__(self) -> None:
        super().__init__()
        # self.inputs = torch.Tensor(1)

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        # self.inputs = torch.Tensor(*x.shape)
        # self.inputs = self.inputs.copy_(x)
        x[x<0] = 0
        return x

    @staticmethod
    def backward(ctx, dl_dy: torch.Tensor) -> torch.Tensor:
        inputs, = ctx.saved_tensors
        dl_dx = dl_dy.clone()
        try:
            dl_dx[inputs<0] = 0
        except NameError as e:
            print(e)
            print("Execute forward() before running backward()")
        return dl_dx
