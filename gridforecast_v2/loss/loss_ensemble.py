import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MSAELoss(torch.nn.Module):
    '''
    func: 构建mse + mae损失权重和
    Parameter
    ---------
    mse_w: float
        default 1. mse损失的权重系数
    mae_w: float
        default 1. mae损失的权重系数
    '''
    def __init__(self, mse_w=1, mae_w=1):

        super(MSAELoss, self).__init__()
        self.mse_w = mse_w
        self.mae_w = mae_w

    def forward(self, y_pre, y_true):
        loss_1 = torch.nn.MSELoss()
        loss_2 = torch.nn.L1Loss()
        loss = self.mse_w * loss_1(y_pre, y_true) + self.mae_w * loss_2(y_pre, y_true)
        return loss


class BMSELoss(torch.nn.Module):
    '''
    func: MSE损失中给强降水处的误差更大的权重
    Parameter
    ---------
    weights: list
        default [1,2,5,10,15,20,25,30].权重列表,给不同的降水强度处对应的像素点的损失不同的权重
    thresholds: list
        阈值列表，即将降水强度按照阈值范围分为若干段，不同阈值给与不同的损失权重
        default [1,5,10,20,30,40,50,90].
    max_value: int or float
        降水强度最大值,默认90, max_value == thresholds[-1]
    if_log: bool
        default True. 输入的y(包括y_pre和y_true)是否做了log(1 + y)的变换
        when True.则同时需要对thresholds做 log(1 + thresholds) / log(1 + max_value)的变换
        when False. thresholds只需要做 thresholds/max_value的变换
    '''
    def __init__(self,
                 weights=[1, 2, 5, 10, 15, 20, 25, 30],
                 thresholds=[1, 5, 10, 20, 30, 40, 50, 90],
                 max_value=90,
                 if_log=True):

        super(BMSELoss, self).__init__()

        assert len(weights) == len(thresholds)
        assert np.min(weights) >= 1

        # 断定阈值的最大值为max_value
        assert thresholds[-1] == max_value

        self.weights = weights
        self.thresholds = thresholds
        self.max_value = max_value
        self.if_log = if_log

        # 如果对y做了np.log(1 + y)的变换
        # 则输入的y_pre和y_true都是
        if self.if_log:
            log_max = np.log(1 + self.max_value)
            self.thresholds = [np.log(1 + threshold)/log_max for threshold in self.thresholds]

        # 如果没有对y做log变换，则默认输入进来的y为[0~1之间]
        else:
            self.thresholds = [threshold / max_value for threshold in self.thresholds]

    def forward(self, y_pre, y_true):

        # 确保y_true在[0,1]之间
        assert y_true.min() >= 0
        assert y_true.max() <= 1

        w_true = y_true.clone()
        for i in range(len(self.weights)):
            w_true[w_true < self.thresholds[i]] = self.weights[i]  # 获取权重矩阵

        return torch.mean(w_true * (y_pre - y_true) ** 2)


class BMAELoss(torch.nn.Module):
    '''
    func: MAE损失中给强降水处的误差更大的权重
    Parameter
    ---------
    weights: list
        default [1,2,5,10,15,20,25,30].权重列表,给不同的降水强度处对应的像素点的损失不同的权重
    thresholds: list
        阈值列表，即将降水强度按照阈值范围分为若干段，不同阈值给与不同的损失权重
        default [1,5,10,20,30,40,50,90].
    max_value: int or float
        降水强度最大值,默认90, max_value == thresholds[-1]
    if_log: bool
        default True. 输入的y(包括y_pre和y_true)是否做了log(1 + y)的变换
        when True.则同时需要对thresholds做 log(1 + thresholds) / log(1 + max_value)的变换
        when False. thresholds只需要做 thresholds/max_value的变换
    '''
    def __init__(self,
                 weights=[1, 2, 5, 10, 15, 20, 25, 30],
                 thresholds=[1, 5, 10, 20, 30, 40, 50, 90],
                 max_value=90,
                 if_log=True):

        super(BMAELoss, self).__init__()

        assert len(weights) == len(thresholds)
        assert np.min(weights) >= 1

        # 断定阈值的最大值为max_value
        assert thresholds[-1] == max_value

        self.weights = weights
        self.thresholds = thresholds
        self.max_value = max_value
        self.if_log = if_log

        # 如果对y做了np.log(1 + y)的变换
        # 则输入的y_pre和y_true都是
        if self.if_log:
            log_max = np.log(1 + self.max_value)
            self.thresholds = [np.log(1 + threshold)/log_max for threshold in self.thresholds]

        # 如果没有对y做log变换，则默认输入进来的y为[0~1之间]
        else:
            self.thresholds = [threshold / max_value for threshold in self.thresholds]

    def forward(self, y_pre, y_true):

        # 确保y_true在[0,1]之间
        assert y_true.min() >= 0
        assert y_true.max() <= 1

        w_true = y_true.clone()
        for i in range(len(self.weights)):
            w_true[w_true < self.thresholds[i]] = self.weights[i]  # 获取权重矩阵

        return torch.mean(w_true * abs(y_pre - y_true))


class BMSAELoss(torch.nn.Module):
    '''
    func: MSE损失中给强降水处的误差更大的权重
    Parameter
    ---------
    weights: list
        default [1,2,5,10,15,20,25,30].权重列表,给不同的降水强度处对应的像素点的损失不同的权重
    thresholds: list
        阈值列表，即将降水强度按照阈值范围分为若干段，不同阈值给与不同的损失权重
        default [1,5,10,20,30,40,50,90].
    max_value: int or float
        降水强度最大值,默认90, max_value == thresholds[-1]
    if_log: bool
        default False. 输入的y(包括y_pre和y_true)是否做了log(1 + y)的变换
        when True.则同时需要对thresholds做 log(1 + thresholds) / log(1 + max_value)的变换
        when False. thresholds只需要做 thresholds/max_value的变换
    mse_w: float
        default 1. mse损失的权重系数
    mae_w: float
        default 1. mae损失的权重系数
    '''
    def __init__(self,
                 weights=[1, 2, 5, 10, 15, 20, 25, 30],
                 thresholds=[1, 5, 10, 20, 30, 40, 50, 90],
                 max_value=90,
                 if_log=False,
                 mse_w=1,
                 mae_w=1):

        super(BMSAELoss, self).__init__()
        assert len(weights) == len(thresholds)
        assert np.min(weights) >= 1

        # 断定阈值的最大值为max_value
        assert thresholds[-1] == max_value
        self.weights = weights
        self.thresholds = thresholds
        self.max_value = max_value
        self.if_log = if_log

        self.mse_w = mse_w
        self.mae_w = mae_w

        # 如果对y做了np.log(1 + y)的变换
        # 则输入的y_pre和y_true都是
        if self.if_log:
            log_max = np.log(1 + self.max_value)
            self.thresholds = [np.log(1 + threshold)/log_max for threshold in self.thresholds]

        # 如果没有对y做log变换，则默认输入进来的y为[0~1之间]
        else:
            self.thresholds = [threshold / max_value for threshold in self.thresholds]

    def forward(self, y_pre, y_true):

        # 确保y_true在[0,1]之间
        y_true[y_true < 0] = 0
        y_true[y_true > 1] = 1

        w_true = y_true.clone()
        for i in range(len(self.weights)):
            w_true[w_true < self.thresholds[i]] = self.weights[i]  # 获取权重矩阵

        loss1 = torch.mean(w_true * (y_pre - y_true)**2)
        loss2 = torch.mean(w_true * abs(y_pre - y_true))

        return self.mse_w * loss1 + self.mae_w * loss2


class ExpWeightedMse(nn.Module):
    """
    Implements exponential weighting on the true labels per Hilburn et al. (2021)
    https://doi.org/10.1175/JAMC-D-20-0084.1

    The weighting follows mean(W(y_true) * (y_pred - y_true) ** 2) where W = exp(b * t_true ** c).

    The authors found for their problem that with radar dBZ scaled 0-1 for 0-60 dBZ and MSE loss, the optimal
    values of the constants are b=5 and c=4. We use the same defaults here. The option `is_minus_1_scaled` should be
    enabled if the dBZ values are scaled -1 to 1 instead of 0 to 1.
    """
    def __init__(self, b=5.0, c=4.0, is_minus_1_scaled=True):
        super().__init__()
        self.b = b
        self.c = c
        self.is_minus_1_scaled = is_minus_1_scaled

    def forward(self, inputs, target):
        if self.is_minus_1_scaled:
            scaled_target = torch.mul(torch.add(target, 1.), 0.5)
        else:
            scaled_target = target
        true_weights = torch.exp(torch.mul(torch.pow(scaled_target, self.c), self.b))
        mse = torch.mean(true_weights * (torch.pow(inputs - target, 2.)))
        return mse


class ExpWeightedMSAELoss(torch.nn.Module):
    """
    Implements exponential weighting on the true labels per Hilburn et al. (2021)
    https://doi.org/10.1175/JAMC-D-20-0084.1

    The weighting follows mean(W(y_true) * (y_pred - y_true) ** 2) where W = exp(b * t_true ** c).

    The authors found for their problem that with radar dBZ scaled 0-1 for 0-60 dBZ and MSE loss, the optimal
    values of the constants are b=5 and c=4. We use the same defaults here.
    """
    def __init__(self,
                 b=5, c=4,
                 mse_w=1, mae_w=1):
        super(ExpWeightedMSAELoss, self).__init__()
        self.b = b
        self.c = c

        self.mse_w = mse_w
        self.mae_w = mae_w

    def forward(self, y_pre, y_true):
        '''
        Parameter
        ---------
        y_pre: 4D or 5D Tensor
            predict by model.
        y_true: 4D or 5D Tensor
            real value.
        ---------
        '''  
        # 确保真实值的范围在 0-1之间
        y_true[y_true < 0] = 0
        y_true[y_true > 1] = 1

        weight = y_true.clone()
        weight = torch.exp(self.b * (weight ** self.c))

        loss_mse = torch.mean(((y_pre - y_true)**2)*(weight))
        loss_mae = torch.mean(abs((y_pre - y_true))*(weight))

        return loss_mse * self.mse_w + loss_mae * self.mae_w


class BEXPMSAELoss(torch.nn.Module):
    '''
    func: MSE + MAE损失在空间中给强回波处的误差更大的权重, 其中权重与真实回波值呈指数正比
          即将MSE + MAE结合使用
    Parameter
    ---------
    a: float or int
        default 6, 即权重是真实回波的指数系数
    b: float
        default 0.8, 即 weight = exp(y_true * a) - b
    '''
    def __init__(self,
                 a=4.3, b=0.8,
                 mse_w=1, mae_w=1):
        super(BEXPMSAELoss, self).__init__()
        self.a = a
        self.b = b

        self.mse_w = mse_w
        self.mae_w = mae_w

    def forward(self, y_pre, y_true):
        '''
        Parameter
        ---------
        y_pre: 4D or 5D Tensor
            predict by model.
        y_true: 4D or 5D Tensor
            real value.
        ---------
        '''  
        # 确保真实值的范围在 0-1之间
        y_true[y_true < 0] = 0
        y_true[y_true > 1] = 1

        weight = y_true.clone()
        weight = torch.exp(weight * self.a) - self.b

        loss_mse = torch.mean(((y_pre - y_true)**2)*(weight))
        loss_mae = torch.mean(abs((y_pre - y_true))*(weight))

        return loss_mse * self.mse_w + loss_mae * self.mae_w


def HingeLoss(prob, flag='fake'):

    assert flag in ['fake', 'real']
    if flag == 'fake':
        return F.relu(prob).mean()
    else:
        return F.relu(1 - prob).mean()


def DgmrDisLoss(dis_fake, dis_real):

    l_real = F.relu(1. - dis_real).mean()
    l_fake = F.relu(1. + dis_fake).mean()
    return l_real + l_fake


def DgmrGenLoss(dis_fake):
    loss = -dis_fake.mean()
    return loss


def DGMRLoss(dis_fake, dis_real, flag='dis'):
    assert flag in ['dis', 'gen']
    if flag == 'dis':
        return DgmrDisLoss(dis_fake, dis_real)
    elif flag == 'gen':
        return DgmrGenLoss(dis_fake)


def WGANLoss(x):
    return torch.mean(x)
