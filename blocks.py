import torch
import torch.nn as nn
import torch.nn.functional as F


class idenUnit(nn.Module):
    def __init__(self, input_channel, g):
        super(idenUnit, self).__init__()

        # bottle neck channel = input channel / 4, as the paper did
        neck_channel = int(input_channel / 4)

        # conv layers, GConv - (shuffle) -> DWConv -> Gconv
        #               bn, relu             bn        bn
        self.gconv1 = nn.Conv2d(input_channel, neck_channel, groups = g, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(neck_channel)

        self.dwconv = nn.Conv2d(neck_channel, neck_channel, groups = neck_channel, kernel_size = 3, 
            padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(neck_channel)

        self.gconv2 = nn.Conv2d(neck_channel, input_channel, groups = g, kernel_size = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(input_channel)

        # for channel shuffle operation 
        self.g, self.n = g, neck_channel/g
        assert self.n == int(self.n), "wrong shape to shuffle"


    def forward(self, inputs):
        x = F.relu(self.bn1(self.gconv1(inputs)))
        
        # channel shuffle
        n, c, w, h = x.shape
        x = x.view(n, self.g, self.n, w, h)
        x = x.transpose_(1, 2).contiguous()
        x = x.view(n, c, w, h)

        x = self.bn(self.dwconv(x))
        x = self.bn2(self.gconv2(x))

        return F.relu(x + inputs)


class poolUnit(nn.Module):
    def __init__(self, input_channel, output_channel, g, first_group = True, downsample = True):
        super(poolUnit, self).__init__()
        self.downsample = downsample

        # bottle neck channel = input channel / 4, as the paper did
        neck_channel = int(output_channel / 4)

        # conv layers, GConv - (shuffle) -> DWConv -> Gconv
        #              bn,relu              bn        bn
        if first_group:
            self.gconv1 = nn.Conv2d(input_channel, neck_channel, groups = g, kernel_size = 1, bias = False)
        else:
            self.gconv1 = nn.Conv2d(input_channel, neck_channel, kernel_size = 1, bias = False)
        
        self.bn1 = nn.BatchNorm2d(neck_channel)

        stride = 2 if downsample else 1
        self.dwconv = nn.Conv2d(neck_channel, neck_channel, groups = neck_channel, stride = stride, kernel_size = 3, 
            padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(neck_channel)

        self.gconv2 = nn.Conv2d(neck_channel, output_channel - input_channel, groups = g, kernel_size = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(output_channel - input_channel)

        # for channel shuffle operation 
        self.g, self.n = g, neck_channel/g
        assert self.n == int(self.n), "error shape to shuffle"


    def forward(self, inputs):
        x = F.relu(self.bn1(self.gconv1(inputs)))
        
        # channel shuffle
        n, c, w, h = x.shape
        x = x.view(n, self.g, self.n, w, h)
        x = x.transpose_(1, 2).contiguous()
        x = x.view(n, c, w, h)

        x = self.bn(self.dwconv(x))
        x = self.bn2(self.gconv2(x))

        shortcut = F.avg_pool2d(inputs, 2) if self.downsample else inputs
        return F.relu(torch.cat((x, shortcut), dim = 1))



if __name__ == "__main__":
    import numpy as np
    x = np.random.randn(3, 16, 32, 32).astype(np.float32)
    x = torch.from_numpy(x)

    print(x.shape)
    b = poolUnit(16, 32, 4)
    y = b(x)
    print(b)
    print(y.shape, y.max(), y.min())