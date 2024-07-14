"""
Func:
FSS:
    - https://github.com/nmcdev/meteva/blob/master/meteva/method/space/fss/fss.py
    - https://www.showdoc.com.cn/meteva/8023545421481661
"""
import numpy as np
from scipy.ndimage.filters import uniform_filter


def fbs_pobfo(ob_xy, fo_xy, grade_list=[1e-30], half_window_size_list=[5], compare=">=", masker_xy=None):
    '''
    :param Ob: 实况数据 2维的numpy
    :param Fo: 实况数据 2维的numpy
    :param window_sizes_list: 卷积窗口宽度的列表，以格点数为单位
    :param threshold_list:  事件发生的阈值
    :param Masker:  2维的numpy检验的关注区域，在Masker网格值取值为0或1，函数只对网格值等于1的区域的数据进行计算。
    :return:
    返回3维数组，shape = (窗口种类数,等级数,3） pob,pfo,fbs
    pob = result[…,0]: 概率观测值的平方的均值.
    pfo = result[…,1]: 窗口概率预报值的平方的均值，
    fbs = result[…,2]: 窗口概率误差（预报-观测）的平方的均值
    若需计算某种时效下，所有时间数据的总体的fss，不能直接将fss取平均，
    需要先对 pob、 pfo、 fbs列先求和，再利用fss = 1 - fbs/(pob+pfo)来计算
    '''
    def moving_ave(dat_xy, half_window_size):
        size = half_window_size * 2 + 1
        dat1 = uniform_filter(dat_xy, size=size)
        dat1 = np.round(dat1[:, :], 10)
        return dat1

    if compare not in [">=", ">", "<", "<="]:
        print("compare 参数只能是 >=   >  <  <=  中的一种")
        return
    shape = ob_xy.shape
    nw = len(half_window_size_list)
    nt = len(grade_list)
    result = np.zeros((nw, nt, 3))
    if masker_xy is None:
        count = ob_xy.size
    else:
        count = np.sum(masker_xy)

    for j in range(nt):
        ob_01 = np.zeros(shape)
        fo_01 = np.zeros(shape)
        if compare == ">=":
            ob_01[ob_xy >= grade_list[j]] = 1
            fo_01[fo_xy >= grade_list[j]] = 1
        elif compare == "<=":
            ob_01[ob_xy <= grade_list[j]] = 1
            fo_01[fo_xy <= grade_list[j]] = 1
        elif compare == ">":
            ob_01[ob_xy > grade_list[j]] = 1
            fo_01[fo_xy > grade_list[j]] = 1
        else:
            ob_01[ob_xy < grade_list[j]] = 1
            fo_01[fo_xy < grade_list[j]] = 1
        for i in range(nw):
            ob_01_smooth = moving_ave(ob_01, half_window_size_list[i])
            fo_01_smooth = moving_ave(fo_01, half_window_size_list[i])
            if masker_xy is not None:
                ob_01_smooth *= masker_xy
                fo_01_smooth *= masker_xy
            result[i, j, 2] = np.sum(np.square(ob_01_smooth - fo_01_smooth))
            result[i, j, 0] = np.sum(np.square(ob_01_smooth))
            result[i, j, 1] = np.sum(np.square(fo_01_smooth))

    result /= count
    return result


def fss(grd_ob, grd_fo,
        half_window_size_list=[5],
        grade_list=[1],
        compare=">=",
        masker=None):
    '''
    params:
        ob_xy: 实况数据 2维的numpy
        fo_xy: 实况数据 2维的numpy
        half_window_sizes_list: 卷积窗口宽度的列表，以格点数为单位
        grade_list:  事件发生的阈值
        masker:  2维的numpy检验的关注区域，在Masker网格值取值为0或1，函数只对网格值等于1的区域的数据进行计算.
    return:
        fss score, array with shape: (M, N). which M = len(half_window_size_list). N = len(grade_list)
        dim 0: number of half_window_size, dim 1: number os grade_list.
        result. 返回3维数组，shape = (窗口种类数,等级数, 4） pob,pfo,fbs, fss
        pob = result[…,0]: 概率观测值的平方的均值.
        pfo = result[…,1]: 窗口概率预报值的平方的均值，
        fbs = result[…,2]: 窗口概率误差（预报-观测）的平方的均值
        fss = result[…,3]: fss值
    '''
    probs = fbs_pobfo(ob_xy=grd_ob, fo_xy=grd_fo,
                      half_window_size_list=half_window_size_list,
                      grade_list=grade_list,
                      compare=compare,
                      masker_xy=masker
                      )
    score = fss_fbs_pobfo(probs)
    score = np.expand_dims(score, axis=-1)  # (M, N, 1)
    results = np.concatenate([probs, score], axis=-1)

    return results


def fss_fbs_pobfo(result):
    """
    result.shape = [..., 3] pob,pfo,fbs
    pob = result[…, 0]: 概率观测值的平方的均值.
    pfo = result[…, 1]: 窗口概率预报值的平方的均值
    fbs = result[…, 2]: 窗口概率误差（预报-观测）的平方的均值
    """
    score = 1 - result[..., 2] / (result[..., 0] + result[..., 1])
    return score
