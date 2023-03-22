import torch
import torch.nn as nn 
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(self, Encoder).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer = nn.Sequential(
            nn.Conv1d(self.in_channels, 32, kernel_size=1, stride=1),
            nn.MaxPool1d(kernel_size=1, stride=1),
            nn.Conv1d(32, 64, kernel_size=1, stride=1),
            nn.MaxPool1d(kernel_size=1, stride=1),
            nn.BatchNorm1d(num_features = 64),
            nn.Softmax()
        )
    
    def forward(self, x):
        x = self.layer(x)
        return x 