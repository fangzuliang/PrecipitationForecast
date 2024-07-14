import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as sn


class Discriminator(nn.Module):
    """
    Source: DCGAN(Deep Convolutional Generative Adversarial Network)
    Example:
    x = torch.rand(2, 1, 192, 192)
    model = Discriminator(base_dim=32, layer=4, input_size=(192, 192))
    y = model(x)
    y.shape  (2, 1)
    Args:
        nn (_type_): _description_
    """
    def __init__(self, base_dim=64,
                 layer=4,
                 input_size=(192, 192)
                 ):
        super(Discriminator, self).__init__()

        self.base_dim = base_dim
        self.layer = layer
        self.input_size = input_size

        self.conv_layers = nn.ModuleList()
        # self.sn = nn.utils.spectral_norm  # 添加spectral norm

        # 添加卷积层和实例归一化层
        for i in range(layer):
            in_channels = 1 if i == 0 else base_dim * (2 ** (i - 1))
            out_channels = base_dim if i == 0 else base_dim * (2 ** i)
            self.conv_layers.append(sn(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)))
            # self.instance_norm_layers.append(nn.InstanceNorm2d(out_channels))

        dst_h, dst_w = int(self.input_size[0] / (2 ** layer)), int(self.input_size[1] / (2 ** layer))
        self.fc = nn.Linear(base_dim * (2 ** (layer - 1)) * dst_h * dst_w, 1)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i in range(self.layer):
            # x = self.sn(self.conv_layers[i])(x)
            x = self.conv_layers[i](x)
            x = self.leaky_relu(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x
