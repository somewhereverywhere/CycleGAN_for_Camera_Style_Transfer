
import torch
import torch.nn as nn
from torch.nn import Module, ReflectionPad2d


class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super(ResidualBlock,self).__init__()
        self.block=nn.Sequential(
            nn.ReflectionPad2d(1),nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=0),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=0),
            nn.InstanceNorm2d(channels)

        )
    def forward(self,x):
        return x+self.block(x)

class Generator(nn.Module):
    def __init__(self,in_channels=3,out_channels=3,n_residual=9):
        super(Generator,self).__init__()

        model=[
            ReflectionPad2d(3),
            nn.Conv2d(in_channels,64,kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        in_features=64
        out_features=in_features*2
        for _ in range(2):
            model+=[
            nn.Conv2d(in_features,out_features,kernel_size=3,padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)]

            in_features=out_features
            out_features=in_features*2

        for _ in range(n_residual):
            model+=[ResidualBlock]


        out_features=in_features//2
        for _ in range(2):
            model+=[
                nn.ConvTranspose2d(in_features,out_features,kernel_size=3,padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)

            ]

        model+=[
            nn.ReflectionPad2d(3),
            nn.Conv2d(64,out_channels,kernel_size=7),
            nn.Tanh
        ]

        self.model=nn.Sequential(*model)

    def forward(self,x):
        return self.model(x)

