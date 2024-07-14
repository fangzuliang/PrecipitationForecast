import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop
from gridforecast_v2.model.unet_2d import attention_assemble


class ResBlock(torch.nn.Module):
    '''
    func: ResNet module. out = x + conv2(relu(conv1(x))
    '''
    def __init__(self,
                 in_dim, out_dim,
                 stride=1,
                 kernel_size=3):
        super(ResBlock, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = self.kernel_size // 2
        self.conv1 = torch.nn.Conv2d(self.in_dim,
                                     self.out_dim,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding,
                                     )
        self.lrelu = torch.nn.LeakyReLU(0.2, True)
        self.conv2 = torch.nn.Conv2d(self.out_dim,
                                     self.in_dim,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding,
                                     )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = out + residual
        return out


class ResAttenBlock(nn.Module):
    '''
    func: Res + Atten.
    attention_mode: str
        default: None. choices: ['SENet', 'CBAM', 'ECA', 'RecoNet', None]
    '''
    def __init__(self,
                 in_dim,
                 attention_mode='ECA'):
        super(ResAttenBlock, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim

        self.attention_mode = attention_mode
        assert self.attention_mode in ['SENet', 'CBAM', 'ECA', 'RecoNet', None]
        if self.attention_mode == 'SENet':
            self.atten = attention_assemble.SELayer(in_dim=self.out_dim)
        elif self.attention_mode == 'CBAM':
            self.atten = attention_assemble.CBAM(self.out_dim)
        elif self.attention_mode == 'ECA':
            self.atten = attention_assemble.ECA(self.out_dim)
        elif self.attention_mode == 'RecoNet':
            self.atten = attention_assemble.RecoNet(in_dim=self.out_dim, r=16)

    def forward(self, x):
        '''
        Parameter
        --------
        x: 4D-Tensor  ----> (batch,channel,height,width)
        '''
        x1 = x
        if self.attention_mode is not None:
            x1 = self.atten(x1)
            y = x1 + x
        else:
            y = x1
        return y


class ConvBlock(torch.nn.Module):
    '''
    func: Conv module. activation + conv + bn + dropout + res + atten
    '''
    def __init__(self,
                 in_dim, out_dim,
                 kernel_size=3,
                 stride=2,
                 activation=True,
                 batch_norm=True,
                 dropout=False,
                 res=1,
                 atten_mode='ECA'):
        super(ConvBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.res = res
        self.atten_mode = atten_mode
        self.padding = kernel_size // 2
        self.conv = torch.nn.Conv2d(self.in_dim,
                                    self.out_dim,
                                    kernel_size=self.kernel_size,
                                    stride=self.stride,
                                    padding=self.padding,
                                    )
        if self.activation:
            self.lrelu = torch.nn.LeakyReLU(0.2, True)
        if self.batch_norm:
            self.bn = torch.nn.BatchNorm2d(self.out_dim, affine=True)
        if self.dropout:
            self.drop = torch.nn.Dropout(0.5)
        if self.res:
            BS = []
            for i in range(self.res):
                BS.append(ResBlock(self.out_dim, self.out_dim))
            self.Res = torch.nn.Sequential(*BS)
        if self.atten_mode:
            self.ResAtten = ResAttenBlock(in_dim=self.out_dim, attention_mode=self.atten_mode)

    def forward(self, x):
        '''
        x: 4D-Tensor ---> [batch, in_dim, height, width]
        '''
        if self.activation:
            x = self.lrelu(x)
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        if self.dropout:
            x = self.drop(x)
        if self.res:
            x = self.Res(x)
        if self.atten_mode:
            x = self.ResAtten(x)
        return x


class ASPP_ConvBlock(torch.nn.Module):
    '''
    func: Conv module using ASPP(Atrous Spatial Pyramid Pooling).
    '''
    def __init__(self,
                 in_dim, out_dim,
                 stride=2,
                 activation=True,
                 batch_norm=True,
                 dropout=False,
                 res=1,
                 atten_mode='SENet'):
        super(ASPP_ConvBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = 3
        self.stride = stride
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.res = res
        self.atten_mode = atten_mode

        # 6 cascades, [conv(3,3), d=6, d=12, d=18, maxpool, meanpool],stride = 2, kernel_size = 3
        self.conv1 = torch.nn.Conv2d(self.in_dim,
                                     self.out_dim,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.kernel_size // 2
                                     )
        self.bn1 = torch.nn.BatchNorm2d(self.out_dim, affine=True)

        self.conv2 = torch.nn.Conv2d(self.in_dim,
                                     self.out_dim,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     dilation=6,
                                     padding=(6 + 6) // 2
                                     )
        self.bn2 = torch.nn.BatchNorm2d(self.out_dim, affine=True)

        self.conv3 = torch.nn.Conv2d(self.in_dim,
                                     self.out_dim,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     dilation=12,
                                     padding=(12 + 12) // 2
                                     )
        self.bn3 = torch.nn.BatchNorm2d(self.out_dim, affine=True)

        self.conv4 = torch.nn.Conv2d(self.in_dim,
                                     self.out_dim,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     dilation=18,
                                     padding=(18 + 18) // 2
                                     )
        self.bn4 = torch.nn.BatchNorm2d(self.out_dim, affine=True)

        self.max_pool = torch.nn.MaxPool2d(kernel_size=self.kernel_size,
                                           stride=self.stride,
                                           padding=self.kernel_size // 2)
        self.mean_pool = torch.nn.AvgPool2d(kernel_size=self.kernel_size,
                                            stride=self.stride,
                                            padding=self.kernel_size // 2)
        self.weights = torch.nn.Parameter(torch.ones((6)), requires_grad=True)

        if self.in_dim != self.out_dim:
            self.conv5 = torch.nn.Conv2d(self.in_dim,
                                         self.out_dim,
                                         kernel_size=1)
            self.bn5 = torch.nn.BatchNorm2d(self.out_dim, affine=True)
            self.conv6 = torch.nn.Conv2d(self.in_dim,
                                         self.out_dim,
                                         kernel_size=1)
            self.bn6 = torch.nn.BatchNorm2d(self.out_dim, affine=True)

        # concat 6 branches of ASPP. 6*self.out_dim ---> self.out_dim
        self.conv_cat = torch.nn.Conv2d(6*self.out_dim,
                                        self.out_dim,
                                        kernel_size=1)
        if self.activation:
            self.lrelu = torch.nn.LeakyReLU(0.2, True)
        if self.batch_norm:
            self.bn = torch.nn.BatchNorm2d(self.out_dim, affine=True)
        if self.dropout:
            self.drop = torch.nn.Dropout(0.5)
        if self.res:
            BS = []
            for i in range(self.res):
                BS.append(ResBlock(self.out_dim, self.out_dim))
            self.Res = torch.nn.Sequential(*BS)
        if self.atten_mode:
            self.ResAtten = ResAttenBlock(in_dim=self.out_dim, attention_mode=self.atten_mode)

    def forward(self, x):
        '''
        x: 4D-Tensor ---> [batch, in_dim, height, width]
        '''
        if self.activation:
            x = self.lrelu(x)
        # Give 6 branches learnable weights
        self.weights = self.weights.to(x.device)

        x1 = self.bn1(self.conv1(x))*self.weights[0]
        x2 = self.bn2(self.conv2(x))*self.weights[1]
        x3 = self.bn3(self.conv3(x))*self.weights[2]
        x4 = self.bn4(self.conv4(x))*self.weights[3]
        x5 = self.max_pool(x)
        x6 = self.mean_pool(x)

        if self.in_dim != self.out_dim:
            x5 = self.bn5(self.conv5(x5))
            x6 = self.bn6(self.conv6(x6))

        x5 = x5*self.weights[4]
        x6 = x6*self.weights[5]

        # out.size = (batch,6*self.out_dim,h,w)
        out = torch.cat([x1, x2, x3, x4, x5, x6], axis=1)
        out = self.conv_cat(out)

        if self.batch_norm:
            out = self.bn(out)
        if self.dropout:
            out = self.drop(out)
        if self.res:
            out = self.Res(out)
        if self.atten_mode:
            out = self.ResAtten(out)
        return out


class DeconvBlock(nn.Module):
    '''
    func: size upscale. activation + upsample(or deconv) + bn + dropout + res + atten.
    up_mode: str
        choices:['upsample','devonv']
        when 'upsample': nn.UpsamplingBilinear2d.
        when 'deconv': nn.ConvTranspose2d.
    '''
    def __init__(self,
                 in_dim, out_dim,
                 kernel_size=3,
                 stride=2,
                 up_mode='upsample',
                 activation=True,
                 batch_norm=True,
                 dropout=False,
                 res=1,
                 atten_mode='ECA'):
        super(DeconvBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.res = res
        self.atten_mode = atten_mode
        self.up_mode = up_mode

        # kernel_size should be odd.
        assert kernel_size % 2 == 1
        assert up_mode in ['upsample', 'deconv']
        assert self.stride in [1, 2]

        if up_mode == 'upsample':
            self.up = torch.nn.Upsample(scale_factor=self.stride, mode='bilinear', align_corners=True)
        else:
            if self.stride == 1:
                self.up = nn.ConvTranspose2d(self.in_dim, self.in_dim,
                                             kernel_size=self.kernel_size,
                                             stride=1,
                                             padding=self.padding)
            # when stride==2, kernel_size must == 4, then size can double.
            elif self.stride == 2:
                self.up = nn.ConvTranspose2d(self.in_dim, self.in_dim,
                                             kernel_size=4,
                                             stride=2,
                                             padding=1)
            self.bn0 = torch.nn.BatchNorm2d(self.in_dim, affine=True)
            self.lrelu0 = torch.nn.LeakyReLU(0.2, True)

        self.conv1 = nn.Conv2d(self.in_dim, self.out_dim,
                               kernel_size=self.kernel_size, stride=1,
                               padding=self.padding)

        if self.activation:
            self.lrelu = torch.nn.LeakyReLU(0.2, True)

        if self.batch_norm:
            self.bn = torch.nn.BatchNorm2d(self.out_dim, affine=True)

        if self.dropout:
            self.drop = torch.nn.Dropout(0.5)

        if self.res:
            BS = []
            for i in range(self.res):
                BS.append(ResBlock(self.out_dim, self.out_dim))
            self.Res = torch.nn.Sequential(*BS)

        if self.atten_mode:
            self.ResAtten = ResAttenBlock(in_dim=self.out_dim, attention_mode=self.atten_mode)

    def forward(self, x):
        '''
        x: 4D-Tensor ---> [batch, in_dim, height, width]
        '''
        if self.activation:
            x = self.lrelu(x)
        out = self.up(x)
        if self.up_mode == 'deconv':
            out = self.bn0(out)
            out = self.lrelu0(out)
        out = self.conv1(out)
        if self.batch_norm:
            out = self.bn(out)
        if self.res:
            out = self.Res(out)
        if self.atten_mode:
            out = self.ResAtten(out)
        if self.dropout:
            out = self.drop(out)
        return out


class UnetEncode(torch.nn.Module):
    '''
    func: Unet Encode part of Unet.
    Parameter
    ---------
    in_dim: int.
        channels of input.
    base_dim: int
        base channels of each layers.default 32. It should be integer multiple of 32.
    layer: int
        vertical layers of Unet, default 4.
    atten_mode: str or None
        choices: ['SENet', 'CBAM', 'ECA', 'RecoNet', None]
    ASPP: int.
        whether add the ASPP module on the output of each layer's encode.
        when 0. whichs means no ASPP module.
    Returns
    -------
    out: list
        output of each layer.
    '''
    def __init__(self, in_dim,
                 base_dim=32,
                 layer=4,
                 atten_mode='SENet',
                 ASPP=2,
                 ):
        super(UnetEncode, self).__init__()
        self.in_dim = in_dim
        self.base_dim = base_dim
        self.atten_mode = atten_mode
        self.layer = layer
        self.ASPP = ASPP
        self.conv0 = ConvBlock(in_dim=self.in_dim,
                               out_dim=self.base_dim,
                               stride=1,
                               activation=False,
                               batch_norm=True,
                               res=0,
                               atten_mode=None)

        self.conv_layers = nn.ModuleList()
        for i in range(self.layer):
            if i == 0:
                in_dim = self.base_dim
                out_dim = in_dim
            else:
                in_dim = out_dim
                out_dim = in_dim * 2
            self.conv = ConvBlock(in_dim=in_dim,
                                  out_dim=out_dim,
                                  stride=2,
                                  activation=True,
                                  batch_norm=True,
                                  res=1,
                                  atten_mode=self.atten_mode)
            self.conv_layers.append(self.conv)

        if self.ASPP != 0:
            self.ASPP_all_layer = torch.nn.ModuleList()
            for i in range(self.layer):
                ASPP_layer = torch.nn.ModuleList()
                if i == 0:
                    in_dim = self.base_dim
                    out_dim = in_dim
                else:
                    in_dim = out_dim*2
                    out_dim = in_dim
                for j in range(self.ASPP):
                    dropout = False if j == 0 else True
                    activation = False if j == 0 else True
                    A_layer = ASPP_ConvBlock(in_dim=in_dim,
                                             out_dim=out_dim,
                                             stride=1,
                                             activation=activation,
                                             batch_norm=True,
                                             res=0,
                                             dropout=dropout,
                                             atten_mode=None,
                                             )
                    ASPP_layer.append(A_layer)
                self.ASPP_all_layer.append(ASPP_layer)

    def forward(self, x):
        '''
        x: 4D-Tensor ---> [batch,in_dim, height, width]
        '''
        out = []
        # size = (batch, base_dim, height, width)
        x = self.conv0(x)
        for conv in self.conv_layers:
            x = conv(x)
            out.append(x)
        if self.ASPP != 0:
            final_out = []
            for x, ASPP_layer in zip(out, self.ASPP_all_layer):
                for A_layer in ASPP_layer:
                    x_aspp = A_layer(x)
                    x = x_aspp + x
                final_out.append(x)
        return out if self.ASPP == 0 else final_out


class ComposeMoreEncode(torch.nn.Module):
    '''
    func: compose and process the outputs of several encode module.
    Parameter
    ---------
    in_dim_list: list of int
        The list composed of the input_dim of each encode.
    encode_num: int
        how many encode.
    base_dim: int
        base channels of each layers.default 32. It should be integer multiple of 32.
    layer: int
        vertical layers of Unet, default 4.
    atten_mode: str or None
        choices: ['SENet', 'CBAM', 'ECA', 'RecoNet', None]
    compose_type: str
        choices: ['cat','add']
        when 'add': Add the output of the same layer corresponding to multiple encode stages directly.
        when 'cat': concate the outputs of the same layer by channel corresponding to multiple encode stages first.
                    Then conv + res + atten to get the right output dim.
        default: cat
    ASPP: int.
        whether add the ASPP module on the output of each layer's encode.
        when 0. whichs means no ASPP module.
    Returns
    -------
    out: list
        output of each layer.
    '''
    def __init__(self,
                 in_dim_list,
                 encode_num,
                 base_dim=32,
                 layer=4,
                 atten_mode='ECA',
                 compose_type='cat',
                 ASPP=2,
                 ):
        super(ComposeMoreEncode, self).__init__()

        self.in_dim_list = in_dim_list
        self.encode_num = encode_num
        self.base_dim = base_dim
        self.layer = layer
        self.atten_mode = atten_mode
        self.compose_type = compose_type
        self.ASPP = ASPP

        assert isinstance(self.in_dim_list, list)
        assert len(self.in_dim_list) == encode_num
        assert compose_type in ['cat', 'add']

        self.encode_layers = nn.ModuleList()
        for in_dim in in_dim_list:
            self.encode = UnetEncode(in_dim=in_dim,
                                     base_dim=self.base_dim,
                                     layer=self.layer,
                                     atten_mode=self.atten_mode,
                                     ASPP=self.ASPP,
                                     )
            self.encode_layers.append(self.encode)

        if self.compose_type == 'cat':
            self.conv_layers = torch.nn.ModuleList()
            for i in range(self.layer):
                in_dim = self.encode_num * self.base_dim * 2 ** (i)
                out_dim = self.base_dim * 2 ** (i)
                conv = ConvBlock(in_dim=in_dim,
                                 out_dim=out_dim,
                                 stride=1,
                                 activation=True,
                                 batch_norm=True,
                                 dropout=False,
                                 res=2,
                                 atten_mode=self.atten_mode)
                self.conv_layers.append(conv)

    def forward(self, X_list):
        '''
        X_list: list or Tensor
            composed with the input of each encode.
        '''

        assert len(X_list) == self.encode_num
        for i in range(self.encode_num):
            assert X_list[i].size()[1] == self.in_dim_list[i]

        # get the output of each encode. Them should have same size.
        all_encode_out = []
        for i in range(self.encode_num):
            x = X_list[i]
            out = self.encode_layers[i](x)
            all_encode_out.append(out)

        single_out = []
        if self.compose_type == 'add':
            for i in range(self.layer):
                out = 0
                for j in range(self.encode_num):
                    out = out + all_encode_out[j][i]
                single_out.append(out)
        else:
            new_single_out = []
            for i in range(self.layer):
                out = []
                for j in range(self.encode_num):
                    out.append(all_encode_out[j][i])
                out = torch.cat(out, axis=1)
                new_single_out.append(out)

            # after concate, using conv + res + atten to reduce channel.
            for i, conv in enumerate(self.conv_layers):
                out = conv(new_single_out[i])
                single_out.append(out)

        return single_out


class UnetDecode(torch.nn.Module):
    '''
    func: Decode module of Unet.
    Parameter
    ---------
    in_dim_list: list of int
        The list composed of the input_dim of each encode.
    out_dim: int
        channels of output.In radar prediction job, it means the predict frames.
    base_dim: int
        base channels of each layers.default 32. It should be integer multiple of 32.
    layer: int
        vertical layers of Unet, default 4.
    up_mode: str
        choices:['upsample','devonv']
        when 'upsample': nn.UpsamplingBilinear2d.
        when 'deconv': nn.ConvTranspose2d.
    atten_mode: str or None
        choices: ['SENet', 'CBAM', 'ECA', 'RecoNet', None]
    compose_type: str
        choices: ['cat','add']
        when 'add': Add the output of the same layer corresponding to multiple encode stages directly.
        when 'cat': concate the outputs of the same layer by channel corresponding to multiple encode stages first.
                    Then conv + res + atten to get the right output dim.
        default: cat
    ASPP: int.
        whether add the ASPP module on the output of each layer's encode.
        when 0. whichs means no ASPP module.
    Returns
    -------
    out: 4D-Tensor ---> [batch, out_dim, height, width]
    Example
    -------
    example1: two encode
        unet = UnetDecode(in_dim_list=[5,10], out_dim=12, base_dim=32, layer=4)
        x1 = torch.rand((2, 5, 256, 256))
        x2 = torch.rand((2, 10, 256, 256))
        x = [x1, x2]
        y = unet(x)
        print(y.shape) --> (2, 10, 256, 256)
    example2: one encode
        unet = UnetDecode(in_dim_list=13, out_dim=12, base_dim=32, layer=4)
        x = torch.rand((2, 13, 256, 256))
        y = unet(x)
        print(y.shape) --> (2, 10, 256, 256)
    act_final: None or str
        when None. no activation on output.
        when str, choices: ['Tanh', 'ReLU', 'LeakyReLU', 'Sigmoid']
    act_last: bool
        default True. When True: out = activate(out+decoder_input) if decoder_input is not None.
        When False. out = decoder_input + activate(out) if decoder_input is not None.
    in_out_size_ratio: list
        The in_out_size_ratio = input_size / output_size.
        Eg: input_size=[192, 192], output_size=[64, 64]. then the in_out_size_ratio = [3, 3]
        if the in_out_size_ratio > 1: then use center_crop to crop the model's final output to target_size.
        default [1, 1].
    '''
    def __init__(self,
                 in_dim_list,
                 out_dim,
                 base_dim=32,
                 layer=4,
                 up_mode='upsample',
                 atten_mode='ECA',
                 compose_type='cat',
                 ASPP=2,
                 act_final='Tanh',
                 act_last=True,
                 in_out_size_ratio=[1, 1]
                 ):
        super(UnetDecode, self).__init__()
        self.in_dim_list = in_dim_list
        self.out_dim = out_dim
        self.base_dim = base_dim
        self.layer = layer
        self.up_mode = up_mode
        self.atten_mode = atten_mode
        self.compose_type = compose_type
        self.ASPP = ASPP
        self.act_final = act_final
        self.act_last = act_last
        self.in_out_size_ratio = in_out_size_ratio

        encode_in_dim = [self.in_dim_list] if isinstance(self.in_dim_list, int) else in_dim_list

        self.encode = ComposeMoreEncode(in_dim_list=encode_in_dim,
                                        encode_num=len(encode_in_dim),
                                        base_dim=self.base_dim,
                                        layer=self.layer,
                                        atten_mode=self.atten_mode,
                                        compose_type=self.compose_type,
                                        ASPP=self.ASPP
                                        )
        self.deconv_layers = nn.ModuleList()
        for i in range(self.layer):
            if i == 0:
                in_dim = self.base_dim * 2 ** (self.layer - 1)
                out_dim = in_dim // 2
                dropout = True
            else:
                in_dim = out_dim * 2
                out_dim = in_dim // 4
                dropout = False
            self.deconv = DeconvBlock(in_dim=in_dim,
                                      out_dim=out_dim,
                                      stride=2,
                                      up_mode=self.up_mode,
                                      activation=True,
                                      batch_norm=True,
                                      dropout=dropout,
                                      atten_mode=self.atten_mode
                                      )
            self.deconv_layers.append(self.deconv)
        self.conv1 = ConvBlock(in_dim=out_dim,
                               out_dim=self.out_dim,
                               stride=1,
                               activation=True,
                               # batch_norm = False,
                               res=0,
                               atten_mode=None
                               )

        if self.act_final is not None:
            if self.act_final == 'Tanh':
                self.act = torch.nn.Tanh()
            elif self.act_final == 'ReLU':
                self.act = torch.nn.ReLU(inplace=True)
            elif self.act_final == 'LeakyReLU':
                self.act = torch.nn.LeakyReLU(0.2, True)
            elif self.act_final == 'ReLU6':
                self.act = torch.nn.ReLU6(inplace=True)
            elif self.act_final == 'Sigmoid':
                self.act = torch.nn.Sigmoid()
            else:
                self.act_final = None

    def forward(self, x_list, decoder_inputs=None):
        '''
        x_list: list or Tensor
            when Tensor. 4D-Tensor or 5D-Tensor
                when 4D-Tensor, size = (batch, in_dim, height, width)
                when 5D-Tensor, size = (batch,encode_num,in_dim,height,width)
            when list. composed of inputs of several(encode_num) encode.
        decoder_inputs: Tensor
            when None. pass
            when Tensor. size = (batch, out_dim, height, width)
        '''
        if isinstance(x_list, torch.Tensor) and len(x_list.size()) == 5:
            batch, nums, channel, height, width = x_list.size()
            assert nums == len(self.in_dim_list)
            x = []
            for i in range(nums):
                x.append(x_list[:, i, :, :, :])
            x_list = x
        if not isinstance(x_list, list):
            x_list = [x_list]
        out = self.encode(x_list)
        for i, deconv in enumerate(self.deconv_layers):
            if i == 0:
                x = deconv(out[-1])
            else:
                x = torch.cat([x, out[-(i + 1)]], axis=1)
                x = deconv(x)
        out = self.conv1(x)

        if self.act_last:
            if decoder_inputs is not None:
                out = out + decoder_inputs
            # act the output
            if self.act_final is not None:
                # print(self.act_final)
                out = self.act(out) / 6 if self.act_final == 'ReLU6' else self.act(out)
        else:
            # act the output
            if self.act_final is not None:
                # print(self.act_final)
                out = self.act(out) / 6 if self.act_final == 'ReLU6' else self.act(out)
            if decoder_inputs is not None:
                out = out + decoder_inputs

        w_ratio, h_ratio = self.in_out_size_ratio
        if w_ratio != 1 or h_ratio != 1:
            ori_w, ori_h = out.shape[-2], out.shape[-1]
            dst_w, dst_h = int(ori_w / w_ratio), int(ori_h / h_ratio)
            out = CenterCrop((dst_w, dst_h))(out)

        return out
