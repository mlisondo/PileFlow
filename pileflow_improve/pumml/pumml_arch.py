#!/usr/bin/env python3
"""
pumml_arch.py

PyTorch implementation of the nominal PUMML architecture from
Komiske, Metodiev, Nachman, Schwartz (arXiv:1707.08600).

Input:
    (batch, 3, 36, 36)

Channels:
    0 = total neutral pT
    1 = charged pileup pT
    2 = charged leading-vertex pT

Output:
    (batch, 1, 9, 9)

Architecture:
    ZeroPad2d(2)
    Conv2d(3 -> 10, kernel=6, stride=2, padding=0)
    ReLU
    ZeroPad2d(2)
    Conv2d(10 -> 10, kernel=6, stride=2, padding=0)
    ReLU
    Conv2d(10 -> 1, kernel=1, stride=1, padding=0)
    ReLU

The convolution weights use He/Kaiming-uniform initialization, matching
Keras' `he_uniform` initialization for ReLU networks.
"""

import torch
import torch.nn as nn


PBAR = 10.0


class PUMML(nn.Module):
    """Exact nominal PUMML CNN architecture in PyTorch."""

    def __init__(self):
        super().__init__()

        self.pad1 = nn.ZeroPad2d(2)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=10,
            kernel_size=6,
            stride=2,
            padding=0,
            bias=True,
        )
        self.relu1 = nn.ReLU()

        self.pad2 = nn.ZeroPad2d(2)
        self.conv2 = nn.Conv2d(
            in_channels=10,
            out_channels=10,
            kernel_size=6,
            stride=2,
            padding=0,
            bias=True,
        )
        self.relu2 = nn.ReLU()

        self.proj = nn.Conv2d(
            in_channels=10,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.relu_out = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        """
        Match Keras `kernel_initializer='he_uniform'`.

        Keras He-uniform:
            limit = sqrt(6 / fan_in)

        torch.nn.init.kaiming_uniform_ with a=0, mode='fan_in',
        nonlinearity='relu' gives the same weight bound.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    module.weight,
                    a=0.0,
                    mode="fan_in",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    # Keras Conv2D defaults to zero bias initialization.
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.pad1(x)       # 36 -> 40
        x = self.conv1(x)      # 40 -> 18
        x = self.relu1(x)

        x = self.pad2(x)       # 18 -> 22
        x = self.conv2(x)      # 22 -> 9
        x = self.relu2(x)

        x = self.proj(x)       # 10 channels -> 1
        x = self.relu_out(x)

        return x


class PUMMLLoss(nn.Module):
    """
    Modified per-pixel logarithmic squared loss used in PUMML:

        L = < [ log((pT_pred + pbar)/(pT_true + pbar)) ]^2 >

    with pbar = 10 GeV.

    PyTorch implementation averages over all batch elements and output pixels.
    """

    def __init__(self, pbar=PBAR, eps=1.0e-7):
        super().__init__()
        self.pbar = float(pbar)
        self.eps = float(eps)

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, min=self.eps)
        y_true = torch.clamp(y_true, min=self.eps)

        log_pred = torch.log(y_pred + self.pbar)
        log_true = torch.log(y_true + self.pbar)

        return torch.mean((log_pred - log_true) ** 2)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary():
    model = PUMML()

    x = torch.zeros(1, 3, 36, 36)

    print(model)
    print()
    print("Shape check:")
    print("input :", tuple(x.shape))

    with torch.no_grad():
        x1 = model.pad1(x)
        print("pad1  :", tuple(x1.shape))

        x1 = model.relu1(model.conv1(x1))
        print("conv1 :", tuple(x1.shape))

        x2 = model.pad2(x1)
        print("pad2  :", tuple(x2.shape))

        x2 = model.relu2(model.conv2(x2))
        print("conv2 :", tuple(x2.shape))

        y = model.relu_out(model.proj(x2))
        print("output:", tuple(y.shape))

    print()
    print("Trainable parameters:", count_parameters(model))


if __name__ == "__main__":
    print_model_summary()