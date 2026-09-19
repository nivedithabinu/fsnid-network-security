import torch
import torch.nn as nn


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class FeatureClassifier(nn.Module):

    def __init__(self, input_size, num_classes):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.log_softmax(x)

        return x