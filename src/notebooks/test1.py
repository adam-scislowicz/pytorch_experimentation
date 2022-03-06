# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + jupyter={"outputs_hidden": true} tags=[]
"""Test notebook"""

# +
import typing

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
from torchsummary import summary
from torchviz import make_dot

import matplotlib.pyplot as plt

# %matplotlib inline

# +
import random


class InputManager:
    """Input generation and conversion routines."""

    def __init__(self) -> None:
        """ "constructor."""

        # IUPAC DNA nucleotide codes
        self.alphabet = [
            "a",
            "c",
            "g",
            "t",
            "r",
            "y",
            "s",
            "w",
            "k",
            "m",
            "b",
            "d",
            "h",
            "v",
            "n",
            "-",
        ]

        self.alphabet_to_onehoti = dict()
        self.onehoti_to_alphabet = dict()
        self.alphabet_len = len(self.alphabet)

        # populate map and reverse map
        for idx, cval in enumerate(self.alphabet):
            self.alphabet_to_onehoti[cval] = idx
            self.onehoti_to_alphabet[idx] = cval

    def gen_random_seqstr(self, len) -> str:
        """Generate a random sequence string."""
        off = 0
        seqstr = ""
        while off < len:
            seqstr += random.choice(self.alphabet[:-1])
            off += 1
        return seqstr

    def char_to_onhot(self, cval) -> "typing.list[float]":
        """Convert a nucleotide character to a one-hot vector."""
        onehot = np.zeros((self.alphabet_len), dtype="float32")
        idx = self.alphabet_to_onehoti[cval]
        onehot[idx] = 1
        return onehot

    def seqstr_to_onehot(self, seqstr) -> "typing.list[typing.list[float]]":
        """Convert a sequence string to a one-hot vector."""
        size = len(seqstr)
        onehot = np.zeros((size, self.alphabet_len), dtype="float32")
        for off, cval in enumerate(seqstr):
            idx = self.alphabet_to_onehoti[cval]
            onehot[off][idx] = 1
        return onehot

    def onehot_to_seqstr(self, onehot) -> str:
        """Convert a one-hot vector to a sequence string."""
        size = onehot.shape[0]
        seqstr = str()
        for pos in range(size):
            idx = next((idx for idx, val in np.ndenumerate(onehot[pos, :]) if val == 1))[0]
            seqstr += self.onehoti_to_alphabet[idx]
        return seqstr


# +
dna_input = InputManager()

seqstr = dna_input.gen_random_seqstr(300)
ohv = dna_input.seqstr_to_onehot(seqstr)
print(seqstr)
print(ohv)


# -


class PrintSize(nn.Module):
    def __init__(self) -> None:
        super(PrintSize, self).__init__()

    def forward(self, input):
        print(input.shape)
        return input


class FixedSeqEmbeddingNet(nn.Module):
    def __init__(self, input_shape) -> None:
        super(FixedSeqEmbeddingNet, self).__init__()

        input_size = input_shape[0] * input_shape[1]

        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv1d(1, 64, 3)),
                    ("ps", PrintSize()),
                    ("prelu1", nn.PReLU()),
                    ("maxpool1", nn.MaxPool1d(kernel_size=2)),
                    ("prelu2", nn.PReLU()),
                    ("conv2", nn.Conv1d(64, 32, 3)),
                    ("prelu3", nn.PReLU()),
                    ("maxpool2", nn.MaxPool1d(kernel_size=2)),
                    ("prelu4", nn.PReLU()),
                    ("conv3", nn.Conv1d(32, 16, 3)),
                    ("prelu5", nn.PReLU()),
                    ("maxpool3", nn.MaxPool1d(kernel_size=2)),
                    ("prelu6", nn.PReLU()),
                    ("fc1", nn.Linear(16 * 2, 8192, bias=False)),
                    ("prelu7", nn.PReLU()),
                ]
            )
        )

    def forward(self, input):
        return self.net(input)


model = FixedSeqEmbeddingNet((300, 16))
print(model)
# summary(model, (300, 16))

ivec = torch.randn(1, 1, 300, 16)
torch.Tensor.ndim = property(lambda self: len(self.shape))
print(ivec.shape)
print(ivec)
ivec2 = ivec[0, 0, :, :]
print(ivec2.shape)
print(ivec2)
# plt.plot()
plt.show()

torch.rand((2, 3))
# y = model(x)
# make_dot(y.mean(), params=dict(list(model.named_parameters()))).render("model", format="png")
