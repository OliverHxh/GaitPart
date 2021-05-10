import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.leaky_relu(x, inplace=True)

class FCONV_4(nn.Module):
    def __init__(self, in_channels, out_channels, p=4):
        super(FCONV_4, self).__init__()
        self.p = p
        self.out_channels = out_channels
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU()
            )

    def forward(self, x, batch_frame=None):        
        batch, c, h, w = x.size()
        out_list = []
        x_part = x.view(batch, c, self.p, -1, w).permute(0, 2, 1, 3, 4).contiguous()
        batch, p, c, h, w = x_part.size()
        x_part = x_part.view(-1, c, h, w)
        output = self.conv(x_part).view(batch, p, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
        batch, c, p, h, w = output.size()
        output = output.view(batch, c, -1, w)
        return output

class FCONV_8(nn.Module):
    def __init__(self, in_channels, out_channels, p=4):
        super(FCONV_8, self).__init__()
        self.p = p
        self.out_channels = out_channels
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU()
            )

    def forward(self, x, batch_frame=None):        
        batch, c, h, w = x.size()
        out_list = []
        x_part = x.view(batch, c, self.p, -1, w).permute(0, 2, 1, 3, 4).contiguous()
        batch, p, c, h, w = x_part.size()
        x_part = x_part.view(-1, c, h, w)
        output = self.conv(x_part).view(batch, p, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
        batch, c, p, h, w = output.size()
        output = output.view(batch, c, -1, w)
        return output

class MCM(nn.Module):
    def __init__(self, in_channels, out_channels, p=8, div=4):
        super(MCM, self).__init__()
        self.p = p
        self.div = div
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels*self.p, in_channels*self.p//self.div, kernel_size=3, padding=1, bias=False,groups=self.p),
            nn.ReLU(),
            nn.Conv1d(in_channels*self.p//self.div, out_channels*self.p, kernel_size=1, padding=0, bias=False,groups=self.p),
            nn.Sigmoid())
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels*self.p, in_channels*self.p//self.div, kernel_size=3, padding=1, bias=False,groups=self.p),
            nn.ReLU(),
            nn.Conv1d(in_channels*self.p//self.div, out_channels*self.p, kernel_size=3, padding=1, bias=False,groups=self.p),
            nn.Sigmoid())
        
        self.leaky_relu = nn.LeakyReLU()
        
        self.max_pool1d_3 = nn.MaxPool1d(kernel_size=3, padding=1, stride=1)
        self.avg_pool1d_3 = nn.AvgPool1d(kernel_size=3, padding=1, stride=1)
        self.max_pool1d_5 = nn.MaxPool1d(kernel_size=5, padding=2, stride=1)
        self.avg_pool1d_5 = nn.AvgPool1d(kernel_size=5, padding=2, stride=1)

    def forward(self, x):
        n, s, c, p = x.size()
        x_in = x.permute(0, 3, 2, 1).contiguous().view(n, p*c, s)
        
        mtb1_attention = self.conv1(x_in)
        tmp_out_1 = self.max_pool1d_3(x_in) + self.avg_pool1d_3(x_in)
        out_1 = tmp_out_1 * mtb1_attention
        
        mtb2_attention = self.conv2(x_in)
        tmp_out_2 = self.max_pool1d_5(x_in) + self.avg_pool1d_5(x_in)
        out_2 = tmp_out_2 * mtb2_attention
        
        out = self.leaky_relu(out_1 + out_2 + x_in).max(-1)[0].view(n, p, c).contiguous()
        
        return out.permute(1, 0, 2).contiguous()

class SetBlock(nn.Module):
    def __init__(self, forward_block, pooling=False):
        super(SetBlock, self).__init__()
        self.forward_block = forward_block
        self.pooling = pooling
        if pooling:
            self.pool2d = nn.MaxPool2d(2)
    def forward(self, x):
        n, s, c, h, w = x.size()
        x = self.forward_block(x.view(-1,c,h,w))
        if self.pooling:
            x = self.pool2d(x)
        _, c, h, w = x.size()
        return x.view(n, s, c, h ,w)
