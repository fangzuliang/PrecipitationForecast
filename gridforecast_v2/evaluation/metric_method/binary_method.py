"""
Reference:
meteva github: https://github.com/nmcdev/meteva/blob/master/meteva/method/yes_or_no/score.py
meteva doc: https://www.showdoc.com.cn/meteva/3975610088390562
including
    1.hfmc, ts_hfmc, ets_hfmc, far_hfmc, mar_hfmc,
      pod_hfmc, precision_hfmc, sr_hfmc, bias_hfmc,
      f1_hfmc, acc_hfmc, hss_hfmc, obs1_ratio_hfmc,
      pre1_ratio_hfmc, odds_ratio_hfmc, orss_hfmc, hk_hfmc
    2.ts, ets, far, mar, pod, precision, sr, bias,
      f1, acc, hss, obs1_ratio, pre1_ratio, odds_ratio, orss, hk
"""
import numpy as np
from sklearn.metrics import confusion_matrix


def hfmc(obs, pre,
         grade_list=[0.1, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100],
         compare=">="):
    """_summary_
    Args:
        :param obs: 实况数据  任意维numpy数组
        :param pre: 预测数据 任意维numpy数组,pre.shape 和obs.shape一致.
            M = 1 if obs.shape == pre.shape
            M = pre.shape[0] if pre.shape[1: ] = obs.shape
        :param grade_list: 多个阈值同时检验时的等级参数
        :compare (str, optional): _description_. Defaults to ">=". choices: [">=", ">", "<", "<="].
    Return:
        hfmc_score with shape: [M, N, 4]. N = len(grade_list). 4 = [tp, fp, fn, tn].
        [tp, fp, fn, tn] --> [hits, falsealarms, misses, correctnegatives]
    """
    assert compare in [">=", ">", "<", "<="]

    if pre.shape == obs.shape:
        pre = np.expand_dims(pre, axis=0)
    assert pre.shape[1:] == obs.shape

    M = pre.shape[0]
    all_hfmc = []
    for i in range(M):
        hfmc_list = []
        for threshold in grade_list:
            if compare == '>=':
                obs_binary = np.where(obs >= threshold, 1, 0)
                pre_binary = np.where(pre[i] >= threshold, 1, 0)
            if compare == '>':
                obs_binary = np.where(obs > threshold, 1, 0)
                pre_binary = np.where(pre[i] > threshold, 1, 0)
            if compare == '<=':
                obs_binary = np.where(obs <= threshold, 1, 0)
                pre_binary = np.where(pre[i] <= threshold, 1, 0)
            if compare == '<':
                obs_binary = np.where(obs < threshold, 1, 0)
                pre_binary = np.where(pre[i] < threshold, 1, 0)

            # correctnegatives, falsealarms, misses, hits
            tn, fp, fn, tp = confusion_matrix(obs_binary.ravel(), pre_binary.ravel(), labels=[0, 1]).ravel()
            hfmc_list.append([tp, fp, fn, tn])
        # hfmc_threshold.shape: [N, 4]. N = len(grade_list)
        hfmc_threshold = np.asarray(hfmc_list)
        all_hfmc.append(hfmc_threshold)
    # shape: [M, N, 4]. N = len(grade_list)
    hfmc_score = np.asarray(all_hfmc)

    return hfmc_score


def ts_hfmc(hfmc_array):
    """
    func: get ts score by hfmc. 1 is perfect. 0 is no skill.
    """
    assert hfmc_array.shape[-1] == 4
    # hit, fal, mis, cn
    tp, fp, fn, tn = hfmc_array[..., 0], hfmc_array[..., 1], hfmc_array[..., 2], hfmc_array[..., 3]
    su = tp + fn + fp
    su[su == 0] = -1
    ts = tp / su
    ts[su == -1] = 0
    return ts


def ts(obs, pre, grade_list=[1, 5, 10, 20], compare=">="):
    hfmc_array = hfmc(obs, pre, grade_list=grade_list, compare=compare)
    return ts_hfmc(hfmc_array)


def ets_hfmc(hfmc_array):
    """
    func: get ets score by hfmc.
    -1/3 到1 的实数，完美值为1, 0代表没有技巧
    """
    assert hfmc_array.shape[-1] == 4
    # hit, fal, mis, cn
    tp, fp, fn, tn = hfmc_array[..., 0], hfmc_array[..., 1], hfmc_array[..., 2], hfmc_array[..., 3]
    total = tn + fp + fn + tp
    hit_random = (tp + fn) * (tp + fp) / total
    su = tp + fn + fp - hit_random
    de = tp - hit_random
    su[su == 0] = -1
    score = de / su
    score[su == -1] = 0
    return score


def ets(obs, pre, grade_list=[1, 5, 10, 20], compare=">="):
    """
    func: get ets score by hfmc.
    -1/3 到1 的实数，完美值为1, 0代表没有技巧
    """
    hfmc_array = hfmc(obs, pre, grade_list=grade_list, compare=compare)
    return ets_hfmc(hfmc_array)


def far_hfmc(hfmc_array):
    """
    func: get far score by hfmc
    """
    assert hfmc_array.shape[-1] == 4
    # hit, fal, mis, cn
    tp, fp, fn, tn = hfmc_array[..., 0], hfmc_array[..., 1], hfmc_array[..., 2], hfmc_array[..., 3]
    su = tp + fp
    su[su == 0] = -1
    score = fp / su
    score[su == -1] = 0
    return score


def far(obs, pre, grade_list=[1, 5, 10, 20], compare=">="):
    hfmc_array = hfmc(obs, pre, grade_list=grade_list, compare=compare)
    return far_hfmc(hfmc_array)


def mar_hfmc(hfmc_array):
    """
    func: get mar score by hfmc
    """
    assert hfmc_array.shape[-1] == 4
    # hit, fal, mis, cn
    tp, fp, fn, tn = hfmc_array[..., 0], hfmc_array[..., 1], hfmc_array[..., 2], hfmc_array[..., 3]
    su = tp + fn
    su[su == 0] = -1
    score = fn / su
    score[su == -1] = 0
    return score


def mar(obs, pre, grade_list=[1, 5, 10, 20], compare=">="):
    hfmc_array = hfmc(obs, pre, grade_list=grade_list, compare=compare)
    return mar_hfmc(hfmc_array)


def pod_hfmc(hfmc_array):
    """
    func: get pod score by hfmc
    """
    assert hfmc_array.shape[-1] == 4
    # hit, fal, mis, cn
    tp, fp, fn, tn = hfmc_array[..., 0], hfmc_array[..., 1], hfmc_array[..., 2], hfmc_array[..., 3]
    su = tp + fn
    su[su == 0] = -1
    score = tp / su
    score[su == -1] = 0
    return score


def pod(obs, pre, grade_list=[1, 5, 10, 20], compare=">="):
    hfmc_array = hfmc(obs, pre, grade_list=grade_list, compare=compare)
    return pod_hfmc(hfmc_array)


def precision_hfmc(hfmc_array):
    '''
    precision = 1 - far
    '''
    assert hfmc_array.shape[-1] == 4
    # hit, fal, mis, cn
    tp, fp, fn, tn = hfmc_array[..., 0], hfmc_array[..., 1], hfmc_array[..., 2], hfmc_array[..., 3]
    su = tp + fp
    su[su == 0] = -1
    score = tp / su
    score[su == -1] = 0
    return score


def precision(obs, pre, grade_list=[1, 5, 10, 20], compare=">="):
    hfmc_array = hfmc(obs, pre, grade_list=grade_list, compare=compare)
    return precision_hfmc(hfmc_array)


def sr_hfmc(hfmc_array):
    '''
    报中率，反映预报的正样本中实际发生的比例. sr = precision = 1 - far
    '''
    assert hfmc_array.shape[-1] == 4
    # hit, fal, mis, cn
    tp, fp, fn, tn = hfmc_array[..., 0], hfmc_array[..., 1], hfmc_array[..., 2], hfmc_array[..., 3]
    su = tp + fp
    su[su == 0] = -1
    score = tp / su
    score[su == -1] = 0
    return score


def sr(obs, pre, grade_list=[1, 5, 10, 20], compare=">="):
    hfmc_array = hfmc(obs, pre, grade_list=grade_list, compare=compare)
    return sr_hfmc(hfmc_array)


def bias_hfmc(hfmc_array):
    """
    func: get bias score by hfmc
    0到正无穷的实数，完美值为1
    """
    assert hfmc_array.shape[-1] == 4
    # hit, fal, mis, cn
    tp, fp, fn, tn = hfmc_array[..., 0], hfmc_array[..., 1], hfmc_array[..., 2], hfmc_array[..., 3]
    su = tp + fn
    su[su == 0] = 1e-10
    score = (tp + fp) / su
    delta = fp - fn
    score[delta == 0] = 1
    score[score > 1e9] = 999999
    return score


def bias(obs, pre, grade_list=[1, 5, 10, 20], compare=">="):
    hfmc_array = hfmc(obs, pre, grade_list=grade_list, compare=compare)
    return bias_hfmc(hfmc_array)


def f1_hfmc(hfmc_array, belta=1):
    """
    #precision = sr_hfmc(hfmc_array)
    #recall = pod_hfmc(hfmc_array)
    #f_score = (1 + belta * belta) * (precision * recall) / (belta * belta * precision + recall)
    """
    hit = hfmc_array[...,0]
    fal = hfmc_array[...,1]
    mis = hfmc_array[...,2]

    su = (1 + belta * belta) * hit + belta * belta * mis + fal
    su[su == 0] = -1
    fscore_array = (1 + belta * belta) * hit / su
    fscore_array[su == -1] = 0
    return fscore_array


def f1(obs, pre, grade_list=[1, 5, 10, 20], compare=">=", belta=1):
    hfmc_array = hfmc(obs, pre, grade_list=grade_list, compare=compare)
    return f1_hfmc(hfmc_array, belta=belta)


def acc_hfmc(hfmc_array):
    """
    func: get acc score by hfmc.
    准确率，反映被正确预报的样本占比
    """
    assert hfmc_array.shape[-1] == 4
    # hit, fal, mis, cn
    tp, fp, fn, tn = hfmc_array[..., 0], hfmc_array[..., 1], hfmc_array[..., 2], hfmc_array[..., 3]
    su = tn + fp + fn + tp
    su[su == 0] = -1
    score = (tp + tn) / su
    score[su == 0] = 0
    return score


def acc(obs, pre, grade_list=[1, 5, 10, 20], compare=">="):
    hfmc_array = hfmc(obs, pre, grade_list=grade_list, compare=compare)
    return acc_hfmc(hfmc_array)


def hss_hfmc(hfmc_array):
    """
    Heidke skill score，统计准确率相对于随机预报的技巧
    """
    assert hfmc_array.shape[-1] == 4
    # hit, fal, mis, cn
    tp, fp, fn, tn = hfmc_array[..., 0], hfmc_array[..., 1], hfmc_array[..., 2], hfmc_array[..., 3]
    su = tp + fp + fn + tn
    correct_random = ((tp + fn) * (tp + fp) + (tn + fn) * (fn + fp)) / su
    sum_rc = su - correct_random
    sum_rc[sum_rc == 0] = -1
    hss = (tp + tn - correct_random) / sum_rc
    hss[sum_rc == -1] = 0
    return hss


def hss(obs, pre, grade_list=[1, 5, 10, 20], compare=">="):
    """
    Heidke skill score，统计准确率相对于随机预报的技巧
    """
    hfmc_array = hfmc(obs, pre, grade_list=grade_list, compare=compare)
    return hss_hfmc(hfmc_array)


def obs1_ratio_hfmc(hfmc_array):
    '''
    观测发生率，观测的正样本占总样本的比例
    '''
    assert hfmc_array.shape[-1] == 4
    # hit, fal, mis, cn
    tp, fp, fn, tn = hfmc_array[..., 0], hfmc_array[..., 1], hfmc_array[..., 2], hfmc_array[..., 3]
    return (tp + fn) / (tp + fp + fn + tn)


def obs1_ratio(obs, pre, grade_list=[1, 5, 10, 20], compare=">="):
    '''
    观测发生率，观测的正样本占总样本的比例
    '''
    hfmc_array = hfmc(obs, pre, grade_list=grade_list, compare=compare)
    return obs1_ratio_hfmc(hfmc_array)


def pre1_ratio_hfmc(hfmc_array):
    '''
    预测发生率，预测的正样本占总样本的比例
    '''
    assert hfmc_array.shape[-1] == 4
    # hit, fal, mis, cn
    tp, fp, fn, tn = hfmc_array[..., 0], hfmc_array[..., 1], hfmc_array[..., 2], hfmc_array[..., 3]
    return (tp + fp) / (tp + fp + fn + tn)


def pre1_ratio(obs, pre, grade_list=[1, 5, 10, 20], compare=">="):
    '''
    预测发生率，预测的正样本占总样本的比例
    '''
    hfmc_array = hfmc(obs, pre, grade_list=grade_list, compare=compare)
    return pre1_ratio_hfmc(hfmc_array)


def odds_ratio_hfmc(hfmc_array):
    '''
    The odds ratio (or评分) gives the ratio of the odds of making a hit to the odds of making a false alarm,
    and takes prior probability into account.
    return: 0 到无穷大的实数，完美值为无穷大, 0代表没有技巧
    '''
    assert hfmc_array.shape[-1] == 4
    # hit, fal, mis, cn
    tp, fp, fn, tn = hfmc_array[..., 0], hfmc_array[..., 1], hfmc_array[..., 2], hfmc_array[..., 3]
    ors = tp * tn / (fn + fp + 1e-8)
    return ors


def odds_ratio(obs, pre, grade_list=[1, 5, 10, 20], compare=">="):
    hfmc_array = hfmc(obs, pre, grade_list=grade_list, compare=compare)
    return odds_ratio_hfmc(hfmc_array)


def orss_hfmc(hfmc_array):
    '''
    The odds ratio (or评分) gives the ratio of the odds of making a hit to the odds of making a false alarm,
    and takes prior probability into account.
    :return: 0 到无穷大的实数，完美值为无穷大, 0代表没有技巧
    '''
    assert hfmc_array.shape[-1] == 4
    # hit, fal, mis, cn
    tp, fp, fn, tn = hfmc_array[..., 0], hfmc_array[..., 1], hfmc_array[..., 2], hfmc_array[..., 3]
    ors = (tp * tn - fn * fp) / (tp * tn + fp * fn)
    return ors


def orss(obs, pre, grade_list=[1, 5, 10, 20], compare=">="):
    hfmc_array = hfmc(obs, pre, grade_list=grade_list, compare=compare)
    return orss_hfmc(hfmc_array)


def hk_hfmc(hfmc_array):
    '''
    Hanssen and Kuipers discriminant，统计准确率相对于随机预报的技巧
    '''
    assert hfmc_array.shape[-1] == 4
    # hit, fal, mis, cn
    tp, fp, fn, tn = hfmc_array[..., 0], hfmc_array[..., 1], hfmc_array[..., 2], hfmc_array[..., 3]
    sum_hm = tp + fn
    sum_fc = fp + tn
    score = tp / sum_hm - fp / sum_fc
    return score


def hk(obs, pre, grade_list=[1, 5, 10, 20], compare=">="):
    '''
    Hanssen and Kuipers discriminant，统计准确率相对于随机预报的技巧
    '''
    hfmc_array = hfmc(obs, pre, grade_list=grade_list, compare=compare)
    return hk_hfmc(hfmc_array)
