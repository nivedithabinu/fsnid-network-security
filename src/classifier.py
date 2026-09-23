import torch
import torch.nn as nn


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class FeatureClassifier(nn.Module):

    def __init__(self, input_size, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size,250),
            nn.ReLU(),
            nn.Linear(250,250),
            nn.ReLU(),
            nn.Linear(250,num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)