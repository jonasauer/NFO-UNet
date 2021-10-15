from abc import ABC

import torch
import torch.nn as nn


class LogisticLoss(nn.Module, ABC):

    def __init__(self):
        super(LogisticLoss, self).__init__()

    def forward(self, predictions, target):
        # flatten both
        predictions, target = torch.flatten(predictions), torch.flatten(target)
        # transform from [0, 1] into [-1, 1]
        target = torch.sub(torch.mul(target, 2), 1)

        # loss = mean(log(1+exp(-uv)))
        values = torch.mul(predictions, target)
        values = torch.exp(-values)
        values = torch.add(values, 1)
        values = torch.log(values)
        return torch.mean(values)
