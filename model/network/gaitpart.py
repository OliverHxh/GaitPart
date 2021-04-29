import torch
import torch.nn as nn
import numpy as np

from .basic_blocks import SetBlock, BasicConv2d, MCM, FCONV_4, FCONV_8


class PartNet(nn.Module):
    def __init__(self, hidden_dim):
        super(PartNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_frame = None

        _set_in_channels = 1
        _set_channels = [32,64,128]
        self.set_layer1 = SetBlock(BasicConv2d(_set_in_channels, _set_channels[0], 5, padding=2))
        self.set_layer2 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[0], 3, padding=1), True)
        self.set_layer3 = SetBlock(FCONV_4(_set_channels[0], _set_channels[1], p=4))
        self.set_layer4 = SetBlock(FCONV_4(_set_channels[1], _set_channels[1], p=4), True)
        self.set_layer5 = SetBlock(FCONV_8(_set_channels[1], _set_channels[2], p=8))
        self.set_layer6 = SetBlock(FCONV_8(_set_channels[2], _set_channels[2], p=8))

        self.MCM = MCM(_set_channels[2], _set_channels[2], p=16, div=16)

        self.bin_num = [16]
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(sum(self.bin_num), 128, hidden_dim)))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal(m.weight.data, 1.0, 0.02)
                nn.init.constant(m.bias.data, 0.0)

    def frame_max(self, x):
        if self.batch_frame is None:
            return torch.max(x, 1)
        else:
            _tmp = [
                torch.max(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
                for i in range(len(self.batch_frame) - 1)
                ]
            max_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
            arg_max_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
            return max_list, arg_max_list

    def frame_median(self, x):
        if self.batch_frame is None:
            return torch.median(x, 1)
        else:
            _tmp = [
                torch.median(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
                for i in range(len(self.batch_frame) - 1)
                ]
            median_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
            arg_median_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
            return median_list, arg_median_list

    def forward(self, silho, batch_frame=None):
        # n: batch_size, s: frame_num, k: keypoints_num, c: channel
        if batch_frame is not None:
            batch_frame = batch_frame[0].data.cpu().numpy().tolist()
            _ = len(batch_frame)
            for i in range(len(batch_frame)):
                if batch_frame[-(i + 1)] != 0:
                    break
                else:
                    _ -= 1
            batch_frame = batch_frame[:_]
            frame_sum = np.sum(batch_frame)
            if frame_sum < silho.size(1):
                silho = silho[:, :frame_sum, :, :]
            self.batch_frame = [0] + np.cumsum(batch_frame).tolist()
        n = silho.size(0)
        x = silho.unsqueeze(2)
        del silho

        x = self.set_layer1(x)
        x = self.set_layer2(x)

        x = self.set_layer3(x)
        x = self.set_layer4(x)

        x = self.set_layer5(x)
        x = self.set_layer6(x)

        feature = list()
        n, s, c, h, w = x.size()
        for num_bin in self.bin_num:
            z = x.view(n, s, c, num_bin, -1)
            z = z.mean(4) + z.max(4)[0]
            feature.append(z)
            
        feature = self.MCM(torch.cat(feature, 3))
        feature = feature.matmul(self.fc_bin)
        feature = feature.permute(1, 0, 2).contiguous()

        return feature, None
