import torch
import torch.nn as nn
import torch.nn.functional as F

from .lCBAM import CBAM


class FlowHead(nn.Module):
    """
    Input: (N, input_dim, H, W)
    Output: (N, hidden_dim, H, W)
    """
    def __init__(self, input_dim=64, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 3, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class SepConvGRU(nn.Module):
    """
    Input:
        h: (N, hidden_dim, H, W)
        x: (N, input_dim, H, W)
    Output: (N, hidden_dim, H, W)
    """
    def __init__(self, hidden_size=32, L=4):
        super(SepConvGRU, self).__init__()
        self.convloc = nn.Sequential(*[nn.Conv2d(hidden_size * 6, hidden_size, kernel_size=5, stride=1, padding=2, dilation=1),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(hidden_size, L * 2, kernel_size=5, stride=1, padding=2, dilation=1)])
        self.convz = nn.Conv2d(hidden_size * (2*L + 1), hidden_size, 3, padding=1)
        self.convr = nn.Conv2d(hidden_size * (2*L + 1), hidden_size, 3, padding=1)
        self.convq = nn.Conv2d(hidden_size * 6, hidden_size, 3, padding=1)

    def wrap(self, input, flow):
        B, C, H, W = input.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1).cuda()
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W).cuda()
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        vgrid = grid + flow
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = torch.nn.functional.grid_sample(input, vgrid)
        return output
    
    def forward(self, h, x):
        flows = self.convloc(torch.cat([x, h], dim=1)) # 96 -> 8
        flows = torch.split(flows, 2, dim=1)
        wrapped_data = []
        for j in range(len(flows)):
            flow = flows[j]
            wrapped_data.append(self.wrap(h, -flow))
        wrapped_data = torch.cat(wrapped_data, dim=1) # 64
        # print(wrapped_data.shape)
        hx = torch.cat([wrapped_data, x], dim=1) # 64 + 80
        z = torch.sigmoid(self.convz(hx))  # 64
        r = torch.sigmoid(self.convr(hx))  # 64
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h


class BasicMotionEncoder(nn.Module):
    """
    Input:
        noise: (N, channels, H, W)
        corr: (N, channels, H, W)
    Output: (N, output_dim, H, W)
    """
    def __init__(self, channels=3, hidden_dim=32, output_dim=64):
        super(BasicMotionEncoder, self).__init__()
        self.convc1 = nn.Conv2d(channels, hidden_dim, 3, padding=1)
        self.convc2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.convf1 = nn.Conv2d(channels, hidden_dim, 3, padding=1)
        self.convf2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv = nn.Conv2d(output_dim, output_dim-channels, 3, padding=1)

    def forward(self, noise, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(noise))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, noise], dim=1)


class BasicUpdateBlock(nn.Module):
    """
    Input:
        net: (N, hidden_dim, H, W)
        inp: (N, hidden_dim, H, W)
        corr: (N, 3, H, W)
        noise: (N, 3, H, W)
    Output:
        net: (N, hidden_dim, H, W)
        delta_flow: (N, 3, H, W)
    """
    def __init__(self, hidden_dim):
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder()
        self.gru = SepConvGRU(hidden_size=hidden_dim)
        self.attn = CBAM(hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=64)

    def forward(self, net, inp, corr, noise):
        motion_features = self.encoder(noise, corr)  # 3, 3 -> 64
        inp = torch.cat([inp, motion_features], dim=1)  # 16 + 64 = 80

        net = self.gru(net, inp)  # 16, 80
        net = self.attn(net)
        delta_flow = self.flow_head(net)

        return net, delta_flow
