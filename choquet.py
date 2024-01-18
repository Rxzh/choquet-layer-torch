import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseDilation2D(nn.Module):
    def __init__(self, in_channels, kernel_size, depth_multiplier=1, stride=1, padding=0, dilation=1):
        super(DepthwiseDilation2D, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels * depth_multiplier, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels)

    def forward(self, x):
        return self.depthwise(x)

class Choquet(nn.Module):
    def __init__(self, num_layers, num_filters, ksize, shrink, subspace, channels):
        super(Choquet, self).__init__()
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.ksize = ksize
        self.shrink = shrink
        self.subspace = subspace

        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(channels if i == 0 else num_filters, num_filters, kernel_size=3, padding=1))
            layers.append(nn.ReLU())

        if num_layers > 0:
            layers.append(nn.Conv2d(num_filters, num_filters // shrink, kernel_size=1))
            layers.append(nn.ReLU())
            layers.append(DepthwiseDilation2D(num_filters // shrink, kernel_size=ksize, padding=ksize // 2, depth_multiplier=shrink))
        else:
            layers.append(DepthwiseDilation2D(channels, kernel_size=ksize, padding=ksize // 2, depth_multiplier=num_filters))

        self.layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(num_filters // shrink if num_layers > 0 else num_filters, subspace)
        self.fc2 = nn.Linear(subspace, subspace)
        self.fc3 = nn.Linear(subspace, 1)

    def forward(self, x):
        x = self.layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(self.fc3(x))
        return x

# Example instantiation of the model
model = Choquet(num_layers=1, num_filters=48, ksize=13, shrink=4, subspace=12, channels=1)
print(model)
