import torch
from torch import nn


class BasicDecoder(nn.Module):

    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def _build_branch1(self):
        self.conv1 = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size)
        )
        self.conv2 = nn.Sequential(
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size)
        )
        self.conv3 = nn.Sequential(
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size)
        )
        self.conv4 = self._conv2d(self.hidden_size * 2, self.data_depth)

        return nn.ModuleList([self.conv1, self.conv2, self.conv3, self.conv4])


    def __init__(self, data_depth, hidden_size):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self._branch1 = self._build_branch1()


    def forward(self, x):
        x = self._branch1[0](x)
        x_list = []
        
        x = self._branch1[1](x)#torch.cat(x_list, dim=1))
        x_list.append(x)

        x = self._branch1[2](torch.cat(x_list, dim=1))
        x_list.append(x)
        
        x = self._branch1[3](torch.cat(x_list, dim=1))

        return x

