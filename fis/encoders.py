import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss

from .update import BasicUpdateBlock
from .wavelet import DWT, IWT
dwt = DWT()
iwt = IWT()


class ContextEncoder(nn.Module):
    def __init__(self, data_depth, hidden_size):
        super(ContextEncoder, self).__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size

        self.features = nn.Sequential(
            self._conv2d(3, self.hidden_size - self.data_depth),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(self.hidden_size - self.data_depth),
        )
        self.layer1 = nn.Sequential(
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(self.hidden_size)
        )
        self.layer2 = nn.Sequential(
            self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(self.hidden_size)
        )
        self.layer3 = nn.Sequential(
            self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
            nn.Tanh(),
        )

    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, padding=1)

    def forward(self, image, data):
        image = self.features(image)
        x = self.layer1(torch.cat([image] + [data], dim=1))
        x = self.layer2(torch.cat([x] + [data], dim=1))
        x = self.layer3(torch.cat([x] + [data], dim=1))

        return x


class BasicEncoder(nn.Module):

    def __init__(self, data_depth, hidden_size, iters=15):
        super(BasicEncoder, self).__init__()
        self.criterion = BCEWithLogitsLoss(reduction="sum")
        self.iters = iters

        assert hidden_size % 2 == 0
        self.hdim = self.cdim = hidden_size // 2

        self.cnet = ContextEncoder(data_depth, hidden_size)
        self.update_block = BasicUpdateBlock(self.hdim)

    def corr_fn(self, x, data):
        with torch.enable_grad():
            x.requires_grad = True
            loss = self.criterion(self.decoder(x), data)
            loss.backward()
            grad = x.grad.clone().detach()
            x.requires_grad = False
        return grad

    def forward(self, image, data, epoch, verbose=False):
        cnet = self.cnet(image, data)
        net, inp = torch.split(cnet, [self.hdim, self.cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        perturb = image.clone()
        predictions = []

        step_size = 1.0
        for itr in range(self.iters):
            perturb = perturb.detach()
            corr = self.corr_fn(perturb, data) 
            noise = perturb - image
            net, delta_noise = self.update_block(net, inp, corr, noise)

            if epoch<-1:
                delta_noise = iwt(dwt(delta_noise))
            perturb = perturb + delta_noise * step_size

            perturb = torch.clamp(perturb, -1, 1)
            if self.constraint is not None:
                perturb = torch.clamp(perturb, image - self.constraint, image + self.constraint)
            predictions.append(perturb)

        return predictions
