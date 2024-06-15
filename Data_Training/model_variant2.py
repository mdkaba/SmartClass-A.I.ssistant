"""
Sources used:
Lab Exercise #06: Introduction to Deep Learning
Image Classification with Convolutional Neural Networks, https://www.youtube.com/watch?v=d9QHNkD_Pos
"""

import torch
import torch.nn as nn


# Sigmoid as activation function
class MultiLayerFCNetVariant2(nn.Module):
    def __init__(self):
        super(MultiLayerFCNetVariant2, self).__init__()
        self.layer1 = nn.Conv2d(3, 32, 7, padding=3, stride=1)  # Larger kernel size
        self.B1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(32, 32, 5, padding=2, stride=1)  # Larger kernel size
        self.B2 = nn.BatchNorm2d(32)
        self.Maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)
        self.layer3 = nn.Conv2d(32, 64, 3, padding=1, stride=1)  # Default kernel size
        self.B3 = nn.BatchNorm2d(64)
        self.layer4 = nn.Conv2d(64, 64, 3, padding=1, stride=1)
        self.B4 = nn.BatchNorm2d(64)
        self.layer5 = nn.Conv2d(64, 128, 2, padding=1, stride=1)  # Smaller kernel size
        self.B5 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.B1(torch.sigmoid(self.layer1(x)))
        x = self.Maxpool(torch.sigmoid(self.layer2(x)))
        x = self.B2(x)
        x = self.B3(self.Maxpool(torch.sigmoid(self.layer3(x))))
        x = self.B4(self.Maxpool(torch.sigmoid(self.layer4(x))))
        x = self.B5(self.Maxpool(torch.sigmoid(self.layer5(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x
