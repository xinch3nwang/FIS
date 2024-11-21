import torch
import torch.nn as nn
#from utils import calc_psnr, calc_ssim, to_np_img
#from imageio import imread, imwrite
#import numpy as np

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = True#False
        
    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = (x1 + x2 + x3 + x4)*0.8
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = True#False

    def forward(self, x):
        in_batch, in_channel, in_height, in_width = x.size()
        #print([in_batch, in_channel, in_height, in_width])
        out_batch, out_channel, out_height, out_width = in_batch, int(in_channel/4), 2*in_height, 2*in_width
        x1 = x[:, 0:out_channel, :, :] / 2
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

        h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return h


if __name__ == '__main__':
    #x = torch.randn(2, 3, 360, 360)
    x = torch.tensor(imread('0801.jpg', pilmode="RGB").astype(np.float32)).permute(2,0,1).unsqueeze(0)
    #print(x.shape)
    iwt = IWT()
    dwt = DWT()
    h = dwt(x)
    h = iwt(h)
    print(calc_psnr(np.array(x.cpu()), np.array(h.cpu())))
