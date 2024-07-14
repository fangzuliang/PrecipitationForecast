'''
Reference:
    - github: https://github.com/nmcdev/meteva/blob/master/meteva/method/continuous/score.py
    - doc: https://www.showdoc.com.cn/meteva/3975613933093256
Method including
    - mae, mse, rmse
'''
import numpy as np


def mae(Ob, Fo):
    '''
    me 求两组数据的误差平均值
    -----------------------------
    :param Ob: 实况数据  任意维numpy数组
    :param Fo: 预测数据 任意维numpy数组,Fo.shape 和Ob.shape一致
    :return: 负无穷到正无穷的实数，最优值为0
    '''
    mae_list = []
    Fo_shape = Fo.shape
    Ob_shape = Ob.shape

    Ob_shpe_list = list(Ob_shape)
    size = len(Ob_shpe_list)
    ind = -size
    Fo_Ob_index = list(Fo_shape[ind:])
    if Fo_Ob_index != Ob_shpe_list:
        print('预报数据和观测数据维度不匹配')
        return
    if len(Fo_shape) == len(Ob_shape):
        mean_abs_error = np.mean(np.abs(Fo - Ob))
        return mean_abs_error
    else:
        Ob_shpe_list.insert(0, -1)
        new_Fo_shape = tuple(Ob_shpe_list)
        new_Fo = Fo.reshape(new_Fo_shape)
        new_Fo_shape = new_Fo.shape
        for line in range(new_Fo_shape[0]):
            mean_abs_error = np.mean(np.abs(new_Fo[line, :] - Ob))
            mae_list.append(mean_abs_error)
        mean_error_array = np.array(mae_list)
        shape = list(Fo_shape[:ind])
        mean_abs_error_array = mean_error_array.reshape(shape)
        return mean_abs_error_array


def mse(Ob, Fo):
    '''
    mean_sqrt_error, 求两组数据的均方误差
    ----------------------------------
    :param Ob: 实况数据  任意维numpy数组
    :param Fo: 预测数据 任意维numpy数组,Fo.shape 和Ob.shape一致
    :return: 0到无穷大，最优值为0
    '''

    mse_list = []
    Fo_shape = Fo.shape
    Ob_shape = Ob.shape

    Ob_shpe_list = list(Ob_shape)
    size = len(Ob_shpe_list)
    ind = -size
    Fo_Ob_index = list(Fo_shape[ind:])
    if Fo_Ob_index != Ob_shpe_list:
        print('预报数据和观测数据维度不匹配')
        return
    if len(Fo_shape) == len(Ob_shape):
        mean_square_error = np.mean(np.square(Fo - Ob))
        return mean_square_error
    else:
        Ob_shpe_list.insert(0, -1)
        new_Fo_shape = tuple(Ob_shpe_list)
        new_Fo = Fo.reshape(new_Fo_shape)
        new_Fo_shape = new_Fo.shape
        for line in range(new_Fo_shape[0]):
            mean_square_error = np.mean(np.square(new_Fo[line, :] - Ob))
            mse_list.append(mean_square_error)
        mean_sqrt_array = np.array(mse_list)
        shape = list(Fo_shape[:ind])
        mean_sqrt_error_array = mean_sqrt_array.reshape(shape)
        return mean_sqrt_error_array


def rmse(Ob, Fo):
    '''
    root_mean_square_error 求两组数据的均方根误差
    ------------------------------
    :param Ob: 实况数据  任意维numpy数组
    :param Fo: 预测数据 任意维numpy数组,Fo.shape 和Ob.shape一致
    :return: 0到无穷大，最优值为0
    '''
    rmse_list = []
    Fo_shape = Fo.shape
    Ob_shape = Ob.shape

    Ob_shpe_list = list(Ob_shape)
    size = len(Ob_shpe_list)
    ind = -size
    Fo_Ob_index = list(Fo_shape[ind:])
    if Fo_Ob_index != Ob_shpe_list:
        print('预报数据和观测数据维度不匹配')
        return
    if len(Fo_shape) == len(Ob_shape):
        mean_square_error = np.sqrt(np.mean(np.square(Fo - Ob)))
        return mean_square_error
    else:
        Ob_shpe_list.insert(0, -1)
        new_Fo_shape = tuple(Ob_shpe_list)
        new_Fo = Fo.reshape(new_Fo_shape)
        new_Fo_shape = new_Fo.shape
        for line in range(new_Fo_shape[0]):
            root_mean_sqrt_error = np.sqrt(np.mean(np.square(new_Fo[line, :] - Ob)))
            rmse_list.append(root_mean_sqrt_error)
        root_mean_sqrt_array = np.array(rmse_list)
        shape = list(Fo_shape[:ind])
        root_mean_sqrt_error_array = root_mean_sqrt_array.reshape(shape)
        return root_mean_sqrt_error_array
