import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gridforecast_v2.src.utils import configure_logging, read_yml_json_file
from gridforecast_v2.evaluation.get_metric_func import (get_binary_metric, get_spatial_metric,
                                                        get_continuous_metric, get_cv_metric
                                                        )
from gridforecast_v2.evaluation.metric_visualization import performance


def get_method_list(all_method_params):
    '''
    get the method_name_list | method_func_list | method_parameter_list
    '''
    # binary_method_params = all_method_params('binary_method', None)
    continuous_method_params = all_method_params.get('continuous_method', None)
    spatial_method_params = all_method_params.get('spatial_method', None)
    cv_method_params = all_method_params.get('cv_method', None)

    method_name_list = []
    method_func_list = []
    method_parameter_list = []
    if continuous_method_params is not None:
        for name in continuous_method_params:
            params = continuous_method_params[name]
            func = get_continuous_metric(params['method'])
            parameter = params.get('parameter', None)

            method_name_list.append(name)
            method_func_list.append(func)
            method_parameter_list.append(parameter)

    if spatial_method_params is not None:
        for name in spatial_method_params:
            params = spatial_method_params[name]
            func = get_spatial_metric(params['method'])
            parameter = params.get('parameter', None)

            method_name_list.append(name)
            method_func_list.append(func)
            method_parameter_list.append(parameter)

    if cv_method_params is not None:
        for name in cv_method_params:
            params = cv_method_params[name]
            func = get_cv_metric(params['method'])
            parameter = params.get('parameter', None)

            method_name_list.append(name)
            method_func_list.append(func)
            method_parameter_list.append(parameter)

    return method_name_list, method_func_list, method_parameter_list


def calculate_metrics(
    obs, pre,
    metric_setting_file,
    save_path=None
):

    all_method_params = read_yml_json_file(metric_setting_file)
    binary_method_params = all_method_params.get('binary_method', None)
    method_name_list, method_func_list, method_parameter_list = get_method_list(all_method_params)

    if binary_method_params is not None:
        if 'HFMC' not in binary_method_params:
            raise ValueError('HFMC method must exist!')
        grade_list = binary_method_params['HFMC']['parameter']['grade_list']
        hfmc_method = get_binary_metric(binary_method_params['HFMC']['method'])

        # shape: [1, N, 4]. N = len(grade_list). 4 = [tp, fp, fn, tn].
        hfmc_array = hfmc_method(obs, pre, grade_list=grade_list)

        # plot 综合评分图
        performance(hfmc_array,
                    grade_list=grade_list,
                    member_list=['DL'],
                    x_y="sr_pod", show=False,
                    dpi=300, sup_fontsize=10,
                    width=None, height=None,
                    bias_list=[0.2, 0.4, 0.6, 0.8, 1, 1.25, 1.67, 2.5, 5],
                    ts_list=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    title="Performance",
                    save_path=os.path.join(save_path, 'Performance.png'),
                    )

        df_hfmc = pd.DataFrame(columns=['TP', 'FP', 'FN', 'TN'], index=grade_list)
        df_hfmc.loc[:, :] = hfmc_array[0, :, :]
        df_hfmc.to_csv(os.path.join(save_path, 'hfmc.csv'))

        for name in binary_method_params:
            if name == 'HFMC':
                continue
            params = binary_method_params[name]
            func = get_binary_metric(params['method'])
            score = func(hfmc_array[0, :, :])  # shape=[N]
            df = pd.DataFrame()
            df[name] = score.ravel()
            df.index = grade_list
            df.to_csv(os.path.join(save_path, f'{name}.csv'))
            plt.figure()
            df.plot(title=name, kind='bar',
                    xlabel='Threshold', ylabel='Score'
                    )
            plt.savefig(os.path.join(save_path, f'{name}.png'), bbox_inches='tight', dpi=200)
            plt.close()

    H, W = obs.shape[-2], obs.shape[-1]
    obs_c = obs.reshape(-1, H, W)
    pre_c = pre.reshape(-1, H, W)

    for name, func, pa in zip(method_name_list, method_func_list, method_parameter_list):
        if name != 'FSS':
            score = func(obs_c, pre_c) if pa is None else func(obs_c, pre_c, **pa)
            df = pd.DataFrame()
            df[name] = [score]
            df.to_csv(os.path.join(save_path, f'{name}.csv'))
            plt.figure()
            df.plot(title=name, kind='line', marker='*',
                    ylabel='Score'
                    )
            plt.savefig(os.path.join(save_path, f'{name}.png'), bbox_inches='tight', dpi=200)
            plt.close()
        else:
            half_window_size_list = pa['half_window_size_list']
            grade_list = pa['grade_list']
            L = pre_c.shape[0]
            for i in range(L):
                if i == 0:
                    score = func(obs_c[i], pre_c[i]) if pa is None else func(obs_c[i], pre_c[i], **pa)
                else:
                    score += func(obs_c[i], pre_c[i]) if pa is None else func(obs_c[i], pre_c[i], **pa)
            # shape=[M, N, 4]  M = len(half_window_size_list). N = len(grade_list)
            score = score / L
            pob, pfo, fbs = score[:, :, 0], score[:, :, 1], score[:, :, 2]
            fss = 1 - fbs / (pob + pfo)
            for j, window_size in enumerate(half_window_size_list):
                df = pd.DataFrame()
                df['FSS'] = fss[j, :]
                df.index = grade_list
                df.to_csv(os.path.join(save_path, f'{name}_window_size_{window_size}.csv'))
                plt.figure()
                df.plot(title=name, kind='bar',
                        xlabel='Threshold', ylabel='Score'
                        )
                plt.savefig(os.path.join(save_path, f'{name}_window_size_{window_size}.png'),
                            bbox_inches='tight', dpi=200)
                plt.close()

    return

# if __name__ == "__main__":
    
#     obs_file = '/THL8/home/zhq/fzl/hydra_experiment/unet_2d_2022-07-08_train/22-21-05/npy/obs.npy'
#     pre_file = '/THL8/home/zhq/fzl/hydra_experiment/unet_2d_2022-07-08_train/22-21-05/npy/pre.npy'
#     metric_setting_file = '/THL8/home/zhq/fzl/branch/forecast/gridforecast_v2/evaluation/eval_setting_v1.yaml'
#     save_path = '/THL8/home/zhq/fzl/branch/forecast/gridforecast_v2/evaluation/test'
#     obs = np.load(obs_file)
#     pre = np.load(pre_file)

#     calculate_metrics(
#                     obs, pre,
#                     metric_setting_file=metric_setting_file,
#                     save_path=save_path
#                 )
