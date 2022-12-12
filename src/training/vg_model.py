import torch
import torch.nn as nn

class PredictionHead(nn.Module):
    def __init__(self, input_size, output_size):
      super(PredictionHead, self).__init__()
      self.fc1 = nn.Linear(input_size, input_size)
      self.relu = torch.nn.ReLU()
      self.fc2 = nn.Linear(input_size, output_size)

    def forward(self, x):
      x = self.fc1(x)
      x = self.relu(x)
      x = self.fc2(x)
      return x