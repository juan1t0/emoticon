import torch
import torch.nn as nn

class DepthNet(nn.Module):
  def __init__(self, inchanels=3, outchanels=26):
    super(DepthNet, self).__init__()
    self.inchanels = inchanels
    self.outchanels = outchanels

    self.conv_layers = nn.Sequential(
        nn.Conv2d(self.inchanels, 32, kernel_size=5, stride=1),        
        nn.MaxPool2d(kernel_size=2,stride=2,padding=1),

        nn.Conv2d(32, 64, kernel_size=3, stride=1),
        nn.MaxPool2d(kernel_size=2,stride=2,padding=1),

        nn.Conv2d(64, 128, kernel_size=3, stride=1),
        nn.MaxPool2d(kernel_size=2,stride=2,padding=1),

        nn.Conv2d(128, 256, kernel_size=3, stride=1),
        nn.MaxPool2d(kernel_size=2,stride=2, padding=1),

        nn.Conv2d(256, 512, kernel_size=3, stride=1),
        nn.MaxPool2d(kernel_size=5,stride=1)
    )
    self.lner_layers = nn.Sequential(
        nn.Linear(512*7*7, 1000),
        nn.Linear(1000, self.outchanels)
    )
  
  def forward(self, x):
    # print(x.shape)
    x = self.conv_layers(x)
    # print(x.shape)
    x = x.view(x.size(0),-1)
    x = self.lner_layers(x)
    return x