import torch
import torch.nn as nn
import torchvision
from thop import profile


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=1):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.fusion = nn.Conv2d(channel * 2, channel, 1, groups=channel, bias=False)
        self.fusion.weight.data.fill_(1.)

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        out = torch.cat([avgout, maxout], dim=1)
        for i in range(avgout.shape[1]):
           out[:, i*2,:,:] = avgout[:, i,:,:]
           out[:, i*2+1,:,:] = maxout[:, i,:,:]
        return self.sigmoid(self.fusion(out))


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        #self.conv1 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1)
        #self.conv2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, dilation=2, padding=4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)

        return self.sigmoid(self.conv(out))


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x



if __name__=='__main__':
    model = CBAM(3)
    input = torch.randn(1, 3, 360, 360)
    model.eval()
    macs, params = profile(model, inputs=(input, ))
    print(f'参数量 (Params): {params}')
    print(f'计算量 (MACs): {macs}')
    #out = model(input)




