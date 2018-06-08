import torch
from torch import nn
from torch.nn import functional as F

from math import sqrt


class TiedSoftmax(nn.Module):
    def __init__(self, weight, cutoff):
        super().__init__()

        self.weights = nn.ModuleList()
        self.weights.append(weight[: cutoff[0]])
        self.biases = nn.ModueList()
        self.biases.append(nn.Parameter(torch.zeros(cutoff[0])))
        for i in range(len(cutoff) - 1):
            self.weights.append(weights[cutoff[i] : cutoff[i + 1]])
            self.biases.append(nn.Parameter(torch.zeros(cutoff[i + 1] - cutoff[i])))
        self.split = nn.Linear(weight.shape[1], len(cutoff) - 1)
        self.cutoff = cutoff

    def set_target(self, target):
        self.id = []

        for i in range(len(self.cutoff) - 1):
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))

            if mask.sum() > 0:
                self.id.append(mask.float().nonzero().squeeze(1))

            else:
                self.id.append(None)

    def forward(self, input, target):
        head = F.linear(input, self.weights[0], self.biases[0])
        split = self.split(input)
        output = [torch.cat([head, split], 1)]
        self.set_target(target)

        for i in range(len(self.id)):
            if self.id[i] is not None:
                output.append(
                    F.linear(
                        input.index_select(0, self.id[i]),
                        self.weights[i],
                        self.biases[i],
                    )
                )
            else:
                output.append(None)

        return output
