import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from gridforecast_v2.evaluation.get_metric_func import get_binary_metric
from gridforecast_v2.evaluation.metric_visualization import performance


def post_process_binary_metric(score_pd_info,
                               sum_columns,
                               binary_method_params,
                               forecast_model_names=['DL', 'EC', 'SMS', 'GRAPES'],
                               groupby_coords=['year', 'month', 'day', 'leadtime', 'cycle', 'issue_time', 'valid_day'],
                               save_path=None,
                               ):

    if 'HFMC' not in binary_method_params:
        raise ValueError('HFMC method must exist!')
    grade_list = binary_method_params['HFMC']['parameter']['grade_list']

    binary_method_func_list = []
    binary_method_name_list = []
    for name in binary_method_params:
        if name == 'HFMC':
            continue
        params = binary_method_params[name]
        func = get_binary_metric(params['method'])
        binary_method_func_list.append(func)
        binary_method_name_list.append(name)

    # groupby_coords = ['year', 'month', 'day', 'leadtime', 'cycle']
    for coord in groupby_coords:
        hfmc_array_dict = defaultdict(dict)
        binary_metric_score = defaultdict(dict)
        df = score_pd_info.groupby(coord)[sum_columns].sum()
        for model in forecast_model_names:
            model_save_path = os.path.join(save_path, coord, model, 'binary')
            os.makedirs(model_save_path, exist_ok=True)
            tp_columns = [f'{model}-{grade}-tp' for grade in grade_list]
            fp_columns = [f'{model}-{grade}-fp' for grade in grade_list]
            fn_columns = [f'{model}-{grade}-fn' for grade in grade_list]
            tn_columns = [f'{model}-{grade}-tn' for grade in grade_list]
            df_tp = df[tp_columns]
            df_fp = df[fp_columns]
            df_fn = df[fn_columns]
            df_tn = df[tn_columns]
            df_tp.columns = grade_list
            df_fp.columns = grade_list
            df_fn.columns = grade_list
            df_tn.columns = grade_list
            df_tp.to_csv(os.path.join(model_save_path, 'tp.csv'))
            df_fp.to_csv(os.path.join(model_save_path, 'fp.csv'))
            df_fn.to_csv(os.path.join(model_save_path, 'fn.csv'))
            df_tn.to_csv(os.path.join(model_save_path, 'tn.csv'))
            hfmc_array_dict[model]['tp'] = df_tp
            hfmc_array_dict[model]['fp'] = df_fp
            hfmc_array_dict[model]['fn'] = df_fn
            hfmc_array_dict[model]['tn'] = df_tn

            # tp.values.shape = [M, N]. M = len(leadtime). N = len(grade_list)
            tp_values = df_tp.values
            fp_values = df_fp.values
            fn_values = df_fn.values
            tn_values = df_tn.values
            hfmc_array = np.stack([tp_values, fp_values, fn_values, tn_values], axis=2)  # [M, N, 4]

            for metric_name, func in zip(binary_method_name_list, binary_method_func_list):
                score = func(hfmc_array)  # shape = [M, N]
                df_score = df_fp.copy()
                df_score.loc[:, :] = score
                binary_metric_score[model][metric_name] = df_score
                df_score.to_csv(os.path.join(model_save_path, f'{metric_name}.csv'))
                plt.figure()
                df_score.plot(title=metric_name, kind='line', marker='*',
                              xlabel=coord, ylabel='Score'
                              )
                plt.savefig(os.path.join(model_save_path, f'{metric_name}.png'), bbox_inches='tight', dpi=200)
                plt.close()

        # one directory per grade.
        for grade in grade_list:
            compare_save_path = os.path.join(save_path, coord, 'compare', 'binary', 'grade', str(grade))
            os.makedirs(compare_save_path, exist_ok=True)
            for name in ['tp', 'fp', 'fn', 'tp']:
                df_binary_list = [hfmc_array_dict[model][name][grade] for model in forecast_model_names]
                df_binary = pd.concat(df_binary_list, axis=1)
                df_binary.columns = forecast_model_names
                df_binary.to_csv(os.path.join(compare_save_path, f'{name}.csv'))

            for metric_name in binary_method_name_list:
                df_score_list = [binary_metric_score[model][metric_name][grade] for model in forecast_model_names]
                df_score = pd.concat(df_score_list, axis=1)
                df_score.columns = forecast_model_names
                df_score.to_csv(os.path.join(compare_save_path, f'{metric_name}.csv'))
                plt.figure()
                df_score.plot(title=metric_name, kind='bar',
                              xlabel=coord, ylabel='Score', rot=300)
                plt.savefig(os.path.join(compare_save_path, f'{metric_name}1.png'), bbox_inches='tight', dpi=200)
                plt.close()

                plt.figure()
                df_score.plot(title=metric_name, kind='line', marker='*',
                              xlabel=coord, ylabel='Score')
                plt.savefig(os.path.join(compare_save_path, f'{metric_name}2.png'), bbox_inches='tight', dpi=200)
                plt.close()

        # one directory per coord(eg: leadtime).
        for i in df.index:
            df_tp = pd.DataFrame(index=grade_list, columns=forecast_model_names)
            df_fp = pd.DataFrame(index=grade_list, columns=forecast_model_names)
            df_fn = pd.DataFrame(index=grade_list, columns=forecast_model_names)
            df_tn = pd.DataFrame(index=grade_list, columns=forecast_model_names)
            compare_save_path = os.path.join(save_path, coord, 'compare', 'binary', coord, str(i))
            os.makedirs(compare_save_path, exist_ok=True)
            for grade in grade_list:
                tp_columns = [f'{model}-{grade}-tp' for model in forecast_model_names]
                fp_columns = [f'{model}-{grade}-fp' for model in forecast_model_names]
                fn_columns = [f'{model}-{grade}-fn' for model in forecast_model_names]
                tn_columns = [f'{model}-{grade}-tn' for model in forecast_model_names]
                df_tp.loc[grade, :] = df.loc[i, :][tp_columns].values
                df_fp.loc[grade, :] = df.loc[i, :][fp_columns].values
                df_fn.loc[grade, :] = df.loc[i, :][fn_columns].values
                df_tn.loc[grade, :] = df.loc[i, :][tn_columns].values

            df_tp.to_csv(os.path.join(compare_save_path, 'tp.csv'))
            df_fp.to_csv(os.path.join(compare_save_path, 'fp.csv'))
            df_fn.to_csv(os.path.join(compare_save_path, 'fn.csv'))
            df_tn.to_csv(os.path.join(compare_save_path, 'tn.csv'))

            # tp.values.shape = [M, N]. M = len(grade_list). N = len(forecast_model_names)
            tp_values = df_tp.values
            fp_values = df_fp.values
            fn_values = df_fn.values
            tn_values = df_tn.values
            hfmc_array = np.stack([tp_values, fp_values, fn_values, tn_values], axis=2)  # [M, N, 4]

            binary_metric_score = {}
            for metric_name, func in zip(binary_method_name_list, binary_method_func_list):
                score = func(hfmc_array)  # shape = [M, N]
                binary_metric_score[metric_name] = score
                df_score = df_fp.copy()
                df_score.loc[:, :] = score
                df_score.to_csv(os.path.join(compare_save_path, f'{metric_name}.csv'))
                plt.figure()
                df_score.plot(title=metric_name, kind='bar',
                              xlabel='Threshold', ylabel='Score',
                              rot=330)
                plt.savefig(os.path.join(compare_save_path, f'{metric_name}.png'), bbox_inches='tight', dpi=200)
                plt.close()

            # plot 综合评分图
            performance(hfmc_array.transpose(1, 0, 2),
                        grade_list=grade_list, member_list=forecast_model_names,
                        x_y="sr_pod", show=False,
                        dpi=300, sup_fontsize=10,
                        width=None, height=None,
                        bias_list=[0.2, 0.4, 0.6, 0.8, 1, 1.25, 1.67, 2.5, 5],
                        ts_list=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        title="Performance",
                        save_path=os.path.join(compare_save_path, 'Performance.png'),
                        )

    return None


def post_process_fss_metric(score_pd_info,
                            spatial_method_params,
                            forecast_model_names=['DL', 'EC', 'SMS', 'GRAPES'],
                            groupby_coords=['year', 'month', 'day', 'leadtime', 'cycle', 'issue_time', 'valid_day'],
                            save_path=None
                            ):

    half_window_size_list = spatial_method_params['FSS']['parameter']['half_window_size_list']
    grade_list = spatial_method_params['FSS']['parameter']['grade_list']

    fss_columns = []
    for model in forecast_model_names:
        for grade in grade_list:
            for half_window_size in half_window_size_list:
                fss_columns.append(f'{model}-{grade}-{half_window_size}-pob')
                fss_columns.append(f'{model}-{grade}-{half_window_size}-pfo')
                fss_columns.append(f'{model}-{grade}-{half_window_size}-fbs')

    for coord in groupby_coords:
        df = score_pd_info.groupby(coord)[fss_columns].sum()
        for half_window_size in half_window_size_list:
            fss_metric_score = defaultdict(dict)
            for model in forecast_model_names:
                model_save_path = os.path.join(save_path, coord, model, 'spatial',
                                               f'fss_half_window_size{half_window_size}')
                os.makedirs(model_save_path, exist_ok=True)
                pob_columns = [f'{model}-{grade}-{half_window_size}-pob' for grade in grade_list]
                pfo_columns = [f'{model}-{grade}-{half_window_size}-pfo' for grade in grade_list]
                fbs_columns = [f'{model}-{grade}-{half_window_size}-fbs' for grade in grade_list]
                df_pob = df[pob_columns]
                df_pfo = df[pfo_columns]
                df_fbs = df[fbs_columns]
                df_pob.columns = grade_list
                df_pfo.columns = grade_list
                df_fbs.columns = grade_list
                df_pob.to_csv(os.path.join(model_save_path, 'pob.csv'))
                df_pfo.to_csv(os.path.join(model_save_path, 'pfo.csv'))
                df_fbs.to_csv(os.path.join(model_save_path, 'fbs.csv'))

                # shape = [T, N] T=len(df_pob), N=len(grade_list).
                pob_values = df_pob.values
                pfo_values = df_pfo.values
                fbs_values = df_fbs.values
                fss_values = 1 - fbs_values / (pob_values + pfo_values)
                df_fss = df_pfo.copy()
                df_fss.loc[:, :] = fss_values
                df_fss.to_csv(os.path.join(model_save_path, 'fss.csv'))

                fss_metric_score[model]['pob'] = df_pob
                fss_metric_score[model]['pfo'] = df_pfo
                fss_metric_score[model]['fbs'] = df_fbs
                fss_metric_score[model]['fss'] = df_fss

                plt.figure()
                df_fss.plot(title='FSS', kind='line', marker='*',
                            xlabel=coord,  ylabel='Score'
                            )
                plt.savefig(os.path.join(model_save_path, 'fss.png'), bbox_inches='tight', dpi=200)
                plt.close()

            # Compare more models.
            # 1.one directory per grade.
            for grade in grade_list:
                compare_save_path = os.path.join(save_path, coord, 'compare', 'spatial', 'grade', f'{grade}',
                                                 f'fss_half_window_size{half_window_size}')
                os.makedirs(compare_save_path, exist_ok=True)
                for n in ['pob', 'pfo', 'fbs', 'fss']:
                    df_fss_list = [fss_metric_score[model][n][grade] for model in forecast_model_names]
                    df_fss = pd.concat(df_fss_list, axis=1)
                    df_fss.columns = forecast_model_names
                    df_fss.to_csv(os.path.join(compare_save_path, f'{n}.csv'))
                    if n == 'fss':
                        plt.figure()
                        df_fss.plot(title='FSS', kind='bar',
                                    xlabel=coord, ylabel='Score',
                                    rot=300)
                        plt.savefig(os.path.join(compare_save_path, 'fss1.png'), bbox_inches='tight', dpi=200)
                        plt.close()

                        plt.figure()
                        df_fss.plot(title='FSS', kind='line', marker='*',
                                    xlabel=coord, ylabel='Score',
                                    )
                        plt.savefig(os.path.join(compare_save_path, 'fss2.png'), bbox_inches='tight', dpi=200)
                        plt.close()

            # 2.one directory per coord(eg: leadtime).
            for i in df.index:
                compare_save_path = os.path.join(save_path, coord, 'compare', 'spatial', coord, str(i),
                                                 f'fss_half_window_size{half_window_size}')
                os.makedirs(compare_save_path, exist_ok=True)
                for n in ['pob', 'pfo', 'fbs', 'fss']:
                    df_fss_list = [fss_metric_score[model][n].loc[i] for model in forecast_model_names]
                    df_fss = pd.concat(df_fss_list, axis=1)
                    df_fss.columns = forecast_model_names
                    df_fss.to_csv(os.path.join(compare_save_path, f'{n}.csv'))
                    if n == 'fss':
                        plt.figure()
                        df_fss.plot(title='FSS', kind='bar',
                                    xlabel='Threshold', ylabel='Score',
                                    rot=300)
                        plt.savefig(os.path.join(compare_save_path, 'fss.png'), bbox_inches='tight', dpi=200)
                        plt.close()

    return None


def post_process_continuous_cv_metric(score_pd_info,
                                      method_name_list,
                                      forecast_model_names=['DL', 'EC', 'SMS', 'GRAPES'],
                                      groupby_coords=['year', 'month', 'day', 'leadtime', 'cycle', 'issue_time', 'valid_day'],
                                      save_path=None
                                      ):
    method_columns = []
    for method in method_name_list:
        for model in forecast_model_names:
            method_columns.append(f'{model}-{method}')

    for coord in groupby_coords:
        df = score_pd_info.groupby(coord)[method_columns].mean()
        for method in method_name_list:
            method_score = []
            for model in forecast_model_names:
                model_save_path = os.path.join(save_path, coord, model, 'continuous_cv')
                os.makedirs(model_save_path, exist_ok=True)
                df_model = df[f'{model}-{method}']
                df_model = df_model.reset_index().set_index(coord)
                df_model.columns = [model]
                method_score.append(df_model)
                df_model.to_csv(os.path.join(model_save_path, f'{method}.csv'))
                plt.figure()
                df_model.plot(title=method, kind='line', marker='*',
                              xlabel=coord,  ylabel='Score'
                              )
                plt.savefig(os.path.join(model_save_path, f'{method}.png'), bbox_inches='tight', dpi=200)
                plt.close()

            # Compare all models
            all_df_model = pd.concat(method_score, axis=1)
            compare_save_path = os.path.join(save_path, coord, 'compare', 'continuous_cv', method)
            os.makedirs(compare_save_path, exist_ok=True)
            all_df_model.to_csv(os.path.join(compare_save_path, f'{method}.csv'))

            plt.figure()
            all_df_model.plot(title=method, kind='bar',
                              xlabel=coord, ylabel='Score',
                              rot=300)
            plt.savefig(os.path.join(compare_save_path, f'{method}1.png'), bbox_inches='tight', dpi=200)
            plt.close()

            plt.figure()
            all_df_model.plot(title=method, kind='line', marker='*',
                              xlabel=coord, ylabel='Score'
                              )
            plt.savefig(os.path.join(compare_save_path, f'{method}2.png'), bbox_inches='tight', dpi=200)
            plt.close()

    return None
