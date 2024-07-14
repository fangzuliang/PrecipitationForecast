import torch
import torch.nn as nn


class SELayer(nn.Module):
    '''
    func: SE channel attention.
    '''
    def __init__(self, in_dim, reduction=2, batch_first=True):
        super(SELayer, self).__init__()

        self.batch_first = batch_first
        self.in_dim = in_dim

        out_dim = 1 if self.in_dim <= reduction else self.in_dim // reduction

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.in_dim, out_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, self.in_dim, bias=False),
            nn.Sigmoid()
            )

    def forward(self, x):

        if not self.batch_first:
            x = x.permute(1, 0, 2, 3)

        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # size = (batch, channel)

        y = self.fc(y).view(b, c, 1, 1)  # size = (batch, channel, 1, 1)
        out = x * y.expand_as(x)    # size = (batch, channel, w, h)

        if not self.batch_first:
            out = out.permute(1, 0, 2, 3)  # size = (channel, batch, w, h)

        return out


class ChannelAttention(nn.Module):
    '''
    func: Channel-attention part of CBAM.
    '''
    def __init__(self, in_channels, reduction=2, batch_first=True):
        super(ChannelAttention, self).__init__()

        self.batch_first = batch_first
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        out_channels = 1 if in_channels <= reduction else in_channels // reduction
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, in_channels, kernel_size=1, bias=False),
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if not self.batch_first:
            x = x.permute(1, 0, 2, 3)

        avgout = self.sharedMLP(self.avg_pool(x))  # size = (batch,in_channels,1,1)
        maxout = self.sharedMLP(self.max_pool(x))  # size = (batch,in_channels,1,1)

        w = self.sigmoid(avgout + maxout)  # channel weight. size = (batch,in_channels,1,1)
        out = x * w.expand_as(x)  # size = (batch,in_channels,w,h)

        if not self.batch_first:
            out = out.permute(1, 0, 2, 3)  # size = (channel,batch,w,h)

        return out


class SpatialAttention(nn.Module):
    '''
    func: Spatial-attention part of CBAM.
    '''
    def __init__(self, kernel_size=3, batch_first=True):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 5, 7)
        padding = kernel_size // 2

        self.batch_first = batch_first
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if not self.batch_first:
            x = x.permute(1, 0, 2, 3)  # size = (batch,channels,w,h)
        avgout = torch.mean(x, dim=1, keepdim=True)  # size = (batch,1,w,h)
        maxout, _ = torch.max(x, dim=1, keepdim=True)  # size = (batch,1,w,h)
        x1 = torch.cat([avgout, maxout], dim=1)  # size = (batch,2,w,h)
        x1 = self.conv(x1)    # size = (batch, 1, w, h)
        w = self.sigmoid(x1)   # size = (batch,1, w, h)
        out = x * w            # size = (batch, channels,w,h)

        if not self.batch_first:
            out = out.permute(1, 0, 2, 3)  # size = (channels,batch,w,h)

        return out


class CBAM(nn.Module):
    '''
    func: CBAM attention composed with channel-attention + spatial-attention
    '''
    def __init__(self, in_channels,
                 out_channels=None, kernel_size=3,
                 stride=1, reduction=2, batch_first=True):

        super(CBAM, self).__init__()

        self.batch_first = batch_first
        self.reduction = reduction
        self.padding = kernel_size // 2

        if out_channels is None:
            out_channels = in_channels

        self.max_pool = nn.MaxPool2d(3, stride=stride, padding=self.padding)
        self.conv_res = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1, stride=1, bias=True)

        # h/2, w/2
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=self.padding,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_channels, reduction=self.reduction,
                                   batch_first=self.batch_first)

        self.sa = SpatialAttention(kernel_size=kernel_size,
                                   batch_first=self.batch_first)

    def forward(self, x):

        if not self.batch_first:
            x = x.permute(1, 0, 2, 3)  # size = (batch,in_channels,w,h)
        out = self.conv1(x)   # size = (batch,out_channels,w/stride,h/stride)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.ca(out)
        out = self.sa(out)  # size = (batch,out_channels,w/stride,h/stride)
        out = self.relu(out)  # size = (batch,out_channels,w/stride,h/stride)

        if not self.batch_first:
            out = out.permute(1, 0, 2, 3)  # size = (out_channels,batch,w/stride,h/stride)

        return out


class ECA(nn.Module):
    '''
    func: ECA attention.
    '''
    def __init__(self, channel, k_size=3):

        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        x: 4D-Tensor
            size = [batch,channel,height,width]
        '''
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class RecoNet(torch.nn.Module):
    '''
    func: RecoNet Attention.
    '''
    def __init__(self, in_dim, r=64):
        super(RecoNet, self).__init__()

        self.in_dim = in_dim
        self.r = r
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv = None
        self.sigmoid = torch.nn.Sigmoid()

        self.parameter_r = torch.nn.Parameter(torch.ones(r), requires_grad=True)

    def forward(self, X):

        self.parameter_r = self.parameter_r.to(X.device)

        assert len(X.size()) == 4
        batch, channel, height, width = X.size()

        for i in torch.arange(self.r):
            if i == 0:
                y = self.TGM_All(X) * self.parameter_r[i]
            else:
                y += self.TGM_All(X) * self.parameter_r[i]
        return y * X

    def TGM_All(self, X):
        '''
        func: Use C, H, and W as channels for channel attention calculations
        '''
        assert len(X.size()) == 4
        batch, channel, height, width = X.size()
        C_weight = self.TGM_C(self, X).to(X.device)
        H_weight = self.TGM_C(self, X.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).to(X.device)
        W_weight = self.TGM_C(self, X.permute(0, 3, 2, 1)).permute(0, 3, 2, 1).to(X.device)
        A = C_weight * H_weight * W_weight
        return A

    @staticmethod
    def TGM_C(self, X):
        '''
        channel attention.
        '''
        assert len(X.size()) == 4
        batch, channel, height, width = X.size()
        self.conv = torch.nn.Conv2d(channel, channel, kernel_size=1).to(X.device)
        y = self.avg_pool(X)
        y = self.conv(y)
        y = self.sigmoid(y)
        return y
