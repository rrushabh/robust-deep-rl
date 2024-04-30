import torch
from torch import nn

import utils

class Encoder(nn.Module):
    def __init__(self, input_channels, output_dim):
        super().__init__()
        
        self.output_dim = output_dim
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.flatten_size = 256 * 6 * 6
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_dim)
        )

        self.apply(utils.weight_init)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.flatten_size)
        x = self.fc_layers(x)
        return x

# Example usage
input_channels = 3  # RGB images
output_dim = 256  # Dimension of the output representation
encoder = Encoder(input_channels, output_dim)
