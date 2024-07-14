"""
Func: some common metrics for compute vision.
Method including
    - ssim, psnr
"""
from skimage import measure, metrics


def ssim(obs, pre, **kwargs):
    '''
    get ssim score.
    '''
    # score = measure.compare_ssim(X=obs, Y=pre, **kwargs)
    score = metrics.structural_similarity(obs, pre, multichannel=True, **kwargs)
    return score


def psnr(obs, pre, **kwargs):
    '''
    get ssim score.
    '''
    obs[obs <= 0] = 0
    pre[pre <= 0] = 0
    # score = measure.compare_psnr(obs, pre, **kwargs)
    score = metrics.peak_signal_noise_ratio(obs, pre, **kwargs)
    return score
