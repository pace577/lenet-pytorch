#!/usr/bin/env python3

import sys
import os

sys.path.insert(1, os.getcwd())

# import pytest
from model import ConvLayer, LeNet5, PoolingLayer, FCLayer
import torch


class TestConv():
    c = ConvLayer(3,6,3)
    x = torch.randn(3,12,12)
    def test_forward_dimensions(self):
        assert self.c.forward(self.x).shape == torch.Size([6,10,10]), "Output Dimensions not matching"

    def test_backward(self):
        out_tensor = self.c.forward(self.x)
        out = out_tensor.sum()
        out.backward()

class TestPool():
    p = PoolingLayer(2,2)
    t = torch.randn(3,12,12, requires_grad=True)
    def test_dimensions(self):
        assert self.p.forward(self.t).shape == torch.Size((3,6,6)), "Output dimensions not matching"

    def test_backward(self):
        out_tensor = self.p.forward(self.t)
        out = out_tensor.sum()
        out.backward()

class TestDense():
    f = FCLayer(12,20)
    t = torch.randn(12, requires_grad=True)
    def test_dimensions(self):
        assert self.f.forward(self.t).shape == torch.Size((20,)), "Output dimensions not matching"

    def test_backward(self):
        out_tensor = self.f.forward(self.t)
        out = out_tensor.sum()
        out.backward()

class TestLeNet5():
    f = LeNet5(3, 10)
    t = torch.randn(3, 32, 32 , requires_grad=True)
    def test_dimensions(self):
        assert self.f.forward(self.t).shape == torch.Size((10,)), "Output dimensions not matching"

    def test_backward(self):
        out_tensor = self.f.forward(self.t)
        out = out_tensor.mean()
        out.backward()
