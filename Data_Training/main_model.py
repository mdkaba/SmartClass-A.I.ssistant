"""
Sources used:
Lab Exercise #06: Introduction to Deep Learning
Image Classification with Convolutional Neural Networks, https://www.youtube.com/watch?v=d9QHNkD_Pos
"""

import torch.nn as nn
import torch.nn.functional as F


# Leaky_relu as activation function
class MultiLayerFCNet(nn.Module):
    def __init__(self):
        super(MultiLayerFCNet, self).__init__()
        self.layer1 = nn.Conv2d(3, 32, 3, padding=1, stride=1)
        self.B1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(32, 32, 3, padding=1, stride=1)
        self.B2 = nn.BatchNorm2d(32)
        self.Maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)
        self.layer3 = nn.Conv2d(32, 64, 3, padding=1, stride=1)
        self.B3 = nn.BatchNorm2d(64)
        self.layer4 = nn.Conv2d(64, 64, 3, padding=1, stride=1)
        self.B4 = nn.BatchNorm2d(64)
        self.layer5 = nn.Conv2d(64, 128, 3, padding=1, stride=1)
        self.B5 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.B1(F.leaky_relu(self.layer1(x)))
        x = self.Maxpool(F.leaky_relu(self.layer2(x)))
        x = self.B2(x)
        x = self.B3(self.Maxpool(F.leaky_relu(self.layer3(x))))
        x = self.B4(self.Maxpool(F.leaky_relu(self.layer4(x))))
        x = self.B5(self.Maxpool(F.leaky_relu(self.layer5(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
