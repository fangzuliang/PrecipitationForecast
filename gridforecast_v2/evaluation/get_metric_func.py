from gridforecast_v2.evaluation.metric_method import (binary_method, spatial_method,
                                                      continuous_method, cv_method
                                                      )


def get_binary_metric(name):
    if name == 'hfmc':
        return binary_method.hfmc
    elif name == 'ts_hfmc':
        return binary_method.ts_hfmc
    elif name == 'ets_hfmc':
        return binary_method.ets_hfmc
    elif name == 'far_hfmc':
        return binary_method.far_hfmc
    elif name == 'mar_hfmc':
        return binary_method.mar_hfmc
    elif name == 'pod_hfmc':
        return binary_method.pod_hfmc
    elif name == 'sr_hfmc':
        return binary_method.sr_hfmc
    elif name == 'bias_hfmc':
        return binary_method.bias_hfmc
    elif name == 'f1_hfmc':
        return binary_method.f1_hfmc
    elif name == 'acc_hfmc':
        return binary_method.acc_hfmc
    elif name == 'hss_hfmc':
        return binary_method.hss_hfmc
    elif name == 'obs1_ratio_hfmc':
        return binary_method.obs1_ratio_hfmc
    elif name == 'pre1_ratio_hfmc':
        return binary_method.pre1_ratio_hfmc
    else:
        raise ValueError(f'No such method: {name}')


def get_spatial_metric(name):
    if name == 'fbs_pobfo':
        return spatial_method.fbs_pobfo
    if name == 'fss_fbs_pobfo':
        return spatial_method.fss_fbs_pobfo
    if name == 'fss':
        return spatial_method.fss
    else:
        raise ValueError(f'No such method: {name}')


def get_continuous_metric(name):
    if name == 'mse':
        return continuous_method.mse
    if name == 'mae':
        return continuous_method.mae
    if name == 'rmse':
        return continuous_method.rmse
    else:
        raise ValueError(f'No such method: {name}')


def get_cv_metric(name):
    if name == 'ssim':
        return cv_method.ssim
    if name == 'psnr':
        return cv_method.psnr
    else:
        raise ValueError(f'No such method: {name}')
