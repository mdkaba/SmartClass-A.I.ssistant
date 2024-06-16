"""
Sources used:
Lab Exercise #06: Introduction to Deep Learning
Image Classification with Convolutional Neural Networks, https://www.youtube.com/watch?v=d9QHNkD_Pos
"""

import torch.nn as nn


# PReLU as activation function
class MultiLayerFCNetVariant1(nn.Module):
    def __init__(self):
        super(MultiLayerFCNetVariant1, self).__init__()
        self.layer1 = nn.Conv2d(3, 32, 3, padding=1, stride=1)
        self.B1 = nn.BatchNorm2d(32)
        self.P1 = nn.PReLU()
        self.layer2 = nn.Conv2d(32, 64, 3, padding=1, stride=1)
        self.B2 = nn.BatchNorm2d(64)
        self.P2 = nn.PReLU()
        self.Maxpool1 = nn.MaxPool2d(2)
        self.layer3 = nn.Conv2d(64, 128, 3, padding=1, stride=2)  # Stride 2 instead of a maxpool
        self.B3 = nn.BatchNorm2d(128)
        self.P3 = nn.PReLU()
        self.layer4 = nn.Conv2d(128, 128, 3, padding=1, stride=1)
        self.B4 = nn.BatchNorm2d(128)
        self.P4 = nn.PReLU()
        self.Maxpool2 = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 12 * 12, 256)
        self.P5 = nn.PReLU()
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.B1(self.P1(self.layer1(x)))
        x = self.B2(self.P2(self.layer2(x)))
        x = self.Maxpool1(x)
        x = self.B3(self.P3(self.layer3(x)))
        x = self.B4(self.P4(self.layer4(x)))
        x = self.Maxpool2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.P5(self.fc1(x))
        x = self.fc2(x)
        return x
