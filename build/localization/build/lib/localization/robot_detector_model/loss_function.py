import numpy as np
import torch
import torch.nn as nn

class CenterLossFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.center_loss = nn.SmoothL1Loss()

    def forward(self, predictions, targets):
        return self.center_loss(predictions, targets)

class OrientationLossFunction(nn.Module):
    def __init__(self):
        super().__init__()
        A = torch.zeros(72, 72)
        for i in range(72):
            for j in range(72):
                A[i, j] = min(abs(i - j), 72 - abs(i - j))
        A = A / torch.max(A)
        self.register_buffer("A", A)

    def forward(self, predictions, targets):
        #[predictions]x[similarity matrix]x[target]
        #[1x72]x[72x72]x[72x1]
        probs = torch.softmax(predictions, dim=1)

        losses = []
        for i in range(len(targets)):
            #select column of similarity matrix corresponding to target heading
            cost = self.A[:, targets[i]]
            losses.append(torch.pow(torch.dot(probs[i], cost), 2))
        return torch.stack(losses).mean()
