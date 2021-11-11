#!/usr/bin/env pytorch-venv

"""Main code that trains and tests your custom made LeNet5 model"""

import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from model import LeNet5
from functions import MyCrossentropyLoss
from tqdm import tqdm

import numpy as np

system_weight = True
dtype = torch.float
device = torch.device("cpu")

def train(args: argparse.Namespace):
    # load the dataset
    print("Inside Train function")
    print("model path:", args.model_path)
    dataroot = './data'
    mb_size = 1
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0,), (1,))])
    trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True,transform=transform)
    trainloader = DataLoader(trainset, batch_size=mb_size, shuffle=True, num_workers=6)
    print(len(trainset))

    # Model and Hyperparameters
    model = LeNet5(3,10)
    loss_fn = MyCrossentropyLoss.apply
    learning_rate = 1e-3
    epochs = 1

    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Training loop
    for epoch in range(epochs):
        print(f"Starting Epoch {epoch+1}")
        epoch_loss = 0
        for i, (features, labels) in tqdm(enumerate(trainloader)):
            features = features.to(device)
            labels = F.one_hot(labels, num_classes=10).to(device)
            preds = model(features.squeeze())
            loss = loss_fn(preds, labels)
            loss.backward()
            with torch.no_grad():
                epoch_loss += float(loss.numpy())
                for param in model.parameters(recurse=True):
                    if param.grad is not None:
                        param -= learning_rate*param.grad
                        # print(param.mean(), param.grad.mean(), param.shape)
                        param.grad[param.grad!=0] = 0

            if i%20==0:
                print(features.squeeze().shape, labels.shape)
                print(f"Batch #{i}, Epoch Loss: {epoch_loss/(i+1):.6}, Batch Loss: {loss:.6}")

    # Save model
    torch.save(model.state_dict, args.model_path)


def test(args: argparse.Namespace):
    # Testing
    print("Inside Test function")
    # load the dataset
    dataroot = './data'
    mb_size = 1
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0,), (1,))])
    testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True,transform=transform)
    testloader = DataLoader(testset, batch_size=mb_size, shuffle=True, num_workers=6)
    print("Test dataset length:", len(testset))

    # Model and Hyperparameters
    model = LeNet5(3,10) #change model
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path))
    loss_fn = MyCrossentropyLoss.apply

    # Training loop
    print(f"Starting Testing")
    testing_loss = 0
    with torch.no_grad():
        for i, (features, labels) in tqdm(enumerate(testloader)):
            features = features.to(device)
            labels = F.one_hot(labels, num_classes=10).to(device)
            preds = model(features.squeeze())
            loss = loss_fn(preds, labels)
            testing_loss += float(loss.numpy())

            if i%20==0:
                print(features.squeeze().shape, labels.shape)
                print(f"Batch #{i}, Epoch Loss: {testing_loss/(i+1):.6}, Batch Loss: {loss:.6}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Main script used to train and test the custom built LeNet5 model with the CIFAR10 dataset.")
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train', help="Train and save a CIFAR10 model")
    train_parser.add_argument('--model-path',
                              help="Path to save the model after training",
                              default=f"./models/lenet5_{time.strftime('%Y%m%d-%H%M%S')}")
    train_parser.set_defaults(func=train)
    test_parser = subparsers.add_parser('test', help="Test a pretrained model (trains a model from scratch if model path is not specified)")
    test_parser.add_argument('--model-path',
                             help="Path to trained model used for testing (trains a model from scratch if model path is not specified)")
    test_parser.set_defaults(func=test)
    args = parser.parse_args()
    args.func(args)
