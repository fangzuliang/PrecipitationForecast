"""
Example:
PYTHONPATH=`pwd` \
    python gridforecast_v2/evaluation/metric_multi_source.py \
    --metric_setting_file /fs1/home/zhq/users/fzl/branch/forecast/gridforecast_v2/evaluation/eval_setting_4grades.yaml \
    --save_path /fs1/home/zhq/users/fzl/experiments/Unet/Ablation/All/metric/new_name/4_grades/metric_test_20210701-20210831 \
    --obs_path /fs1/home/zhq/data/Fusion_data/CMPA/CMPA5_V3_nc/HOR/NRT_0.05_15-60_70-140_UTC_rain \
    --obs_var 'r3' \
    --forecast_paths "/fs1/home/zhq/users/fzl/experiments/Unet/baseline/ensemble/inference_nc,
                     /fs1/home/zhq/users/fzl/experiments/Unet/Ablation/Exp1/ensemble/inference_nc,
                     /fs1/home/zhq/users/fzl/experiments/Unet/Ablation/Exp2/ensemble/inference_nc,
                     /fs1/home/zhq/users/fzl/experiments/Unet/Ablation/Exp3/ensemble/inference_nc,
                     /fs1/home/zhq/users/fzl/experiments/Unet/Ablation/Exp4/ensemble/inference_nc,
                     /fs1/home/zhq/users/fzl/experiments/Unet/Ablation/Exp5/ensemble/inference_nc,
                     /fs1/home/zhq/users/fzl/experiments/Unet/Ablation/Exp6/ensemble/inference_nc,
                     /fs1/home/zhq/users/fzl/experiments/Unet/Ablation/Exp7/ensemble/inference_nc, 
                     /fs1/home/zhq/users/fzl/experiments/Unet/Ablation/Exp8/ensemble/inference_nc, 
                     /fs1/home/zhq/data/NWP_data/EC/EC_V3_nc/ecmwf_nc_0.05_31-45_108-124_rain,
                     /fs1/home/zhq/data/NWP_data/SMS/SMS_V3_nc/SMS_nc_0.05_31-45_108-124_rain,
                     /fs1/home/zhq/data/NWP_data/GRAPES/GRAPES_V3_nc/GRAPES_nc_0.05_31-45_108-124_rain" \
    --forecast_variables 'pre_r3,pre_r3,pre_r3,pre_r3,pre_r3,pre_r3,pre_r3,pre_r3,pre_r3,r3,r3,r3' \
    --forecast_model_names 'DL,DL,DL,DL,DL,DL,DL,DL,DL,EC,SMS,GRAPES' \
    --forecast_show_model_names 'Baseline,Exp1,Exp2,Exp3,Exp4,Exp5,Exp6,Exp7,Exp8,ECMWF,CMA-SH9,CMA-MESO' \
    --start_day '2021-07-01' \
    --end_day '2021-08-31' \
    --cycle_list '0,12' \
    --leadtime_list '3,6,9,12,15,18,21,24' \
    --groupby_coords 'year,month,cycle,leadtime'
"""
import os
import numpy as np
import xarray as xr
import pandas as pd
import logging
import argparse
from collections import defaultdict
from gridforecast_v2.src.utils import configure_logging, read_yml_json_file
from gridforecast_v2.evaluation.get_metric_func import (get_binary_metric, get_spatial_metric,
                                                        get_continuous_metric, get_cv_metric
                                                        )
from gridforecast_v2.evaluation.post_process_metric_df import (post_process_binary_metric,
                                                               post_process_fss_metric,
                                                               post_process_continuous_cv_metric
                                                               )

configure_logging()
logger = logging.getLogger(__name__)

FILE_FORMAT_LIST = {
    # NRT_0.05_15-60_70-140_UTC_rain/2021/06/07/CMPA5_0.05_20210607_f010000_HOR_NRT_PRE.nc
    'OBS': '%Y/%m/%d/CMPA5_0.05_%Y%m%d_f%H0000_HOR_NRT_PRE.nc',
    # 2021/03/18/12/20210318_T12_f12.nc
    'DL': '{year}/{month}/{day}/{cycle}/{year}{month}{day}_T{cycle}_f{leadtime}.nc',
    'FRNet': '{year}/{month}/{day}/{cycle}/{year}{month}{day}_T{cycle}_f{leadtime}.nc',
    # ecmwf_nc_0.05_31-45_108-124_rain/2021/07/02/00/EC_surface_0.05_20210702_T00_f012.nc
    'EC': '{year}/{month}/{day}/{cycle}/EC_surface_0.05_{year}{month}{day}_T{cycle}_f0{leadtime}.nc',
    'ECMWF': '{year}/{month}/{day}/{cycle}/EC_surface_0.05_{year}{month}{day}_T{cycle}_f0{leadtime}.nc',
    # SMS_nc_0.05_31-45_108-124_rain/2021/07/03/00/SMS_surface_0.05_20210703_T00_f03.nc
    'SMS': '{year}/{month}/{day}/{cycle}/SMS_surface_0.05_{year}{month}{day}_T{cycle}_f{leadtime}.nc',
    'CMA-SH9': '{year}/{month}/{day}/{cycle}/SMS_surface_0.05_{year}{month}{day}_T{cycle}_f{leadtime}.nc',
    # GRAPES_nc_0.05_31-45_108-124_rain/2021/06/02/12/GRAPES_surface_0.05_20210602_T12_f004.nc
    'GRAPES': '{year}/{month}/{day}/{cycle}/GRAPES_surface_0.05_{year}{month}{day}_T{cycle}_f0{leadtime}.nc',
    'CMA-MESO': '{year}/{month}/{day}/{cycle}/GRAPES_surface_0.05_{year}{month}{day}_T{cycle}_f0{leadtime}.nc',
    'CMA-3KM': '{year}/{month}/{day}/{cycle}/GRAPES_surface_0.05_{year}{month}{day}_T{cycle}_f0{leadtime}.nc',
}


def get_obs_models_index(
    obs_path,
    forecast_path_list,
    forecast_model_list=['DL', 'EC', 'SMS', 'GRAPES'],
    forecast_show_model_list=['FRNet', 'ECMWF', 'CMA-SH9', 'CMA-MESO'],
    start_day='2021-06-01',
    end_day='2021-07-01',
    cycle_list=['00', '12'],
    leadtime_list=['03', '06', '09', '12', '15', '18', '21', '24'],
    save_file=None,
):
    global year, month, day, cycle, leadtime

    nums = len(forecast_model_list)

    all_year = []
    all_month = []
    all_day = []
    all_cycle = []
    all_issue_time = []
    all_leadtime = []
    all_valid_time = []
    all_valid_day = []
    all_date = []
    all_obs_file = []
    all_model_file = defaultdict(list)
    all_days = pd.date_range(start_day, end_day, freq='1d')
    for t in all_days:
        for cycle in cycle_list:
            for leadtime in leadtime_list:
                date = t.strftime('%Y%m%d')
                year, month, day = date[0:4], date[4:6], date[6:8]
                issue_time = t + pd.Timedelta(f'{int(cycle)}h')
                # issue_time = issue_time.strftime('%Y%m%dT%H')
                valid_time = t + pd.Timedelta(f'{int(cycle)}h') + pd.Timedelta(f'{int(leadtime)}h')
                valid_day = valid_time.strftime('%d')
                obs_filename = valid_time.strftime(FILE_FORMAT_LIST['OBS'])
                obs_file = os.path.join(obs_path, obs_filename)
                if not os.path.exists(obs_file):
                    print(f'{obs_file} does not exist!')
                    continue
                valid_time = valid_time.strftime('%Y%m%dT%H%M%S')
                flag = True
                model_file_list = []
                for model, show_model, model_path in zip(forecast_model_list, forecast_show_model_list, forecast_path_list):
                    model_format = "f'" + FILE_FORMAT_LIST[model] + "'"
                    model_filename = eval(model_format)
                    model_file = os.path.join(model_path, model_filename)
                    if not os.path.exists(model_file):
                        flag = False
                        print(f'{model} {model_file} does not exist!')
                        break
                    model_file_list.append(model_file)

                if flag:
                    all_year.append(year)
                    all_month.append(month)
                    all_day.append(day)
                    all_cycle.append(cycle)
                    all_issue_time.append(issue_time)
                    all_leadtime.append(leadtime)
                    all_valid_time.append(valid_time)
                    all_valid_day.append(valid_day)
                    all_date.append(f'{date}T{cycle}_f{leadtime}')
                    all_obs_file.append(obs_file)

                    for i in range(nums):
                        all_model_file[forecast_show_model_list[i]].append(model_file_list[i])

    pd_info = pd.DataFrame()
    pd_info['year'] = all_year
    pd_info['month'] = all_month
    pd_info['day'] = all_day
    pd_info['cycle'] = all_cycle
    pd_info['issue_time'] = all_issue_time
    pd_info['leadtime'] = all_leadtime
    pd_info['valid_time'] = all_valid_time
    pd_info['valid_day'] = all_valid_day
    pd_info['date'] = all_date
    pd_info['OBS'] = all_obs_file
    for i in range(nums):
        pd_info[forecast_show_model_list[i]] = all_model_file[forecast_show_model_list[i]]

    if save_file is not None:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        pd_info.to_csv(save_file, index=None)
        logger.info(f'{save_file} down!')

    return pd_info


def get_spatial_selection(pd_info, forecast_model_names):
    '''
    func: get the common spatial area of OBS and all models.
    '''
    info = pd_info.iloc[0, :]
    ds_obs = xr.open_dataset(info['OBS'])
    lon_min, lon_max = ds_obs['lon'].min().values, ds_obs['lon'].max().values
    lat_min, lat_max = ds_obs['lat'].min().values, ds_obs['lat'].max().values
    lon = ds_obs['lon'].values
    lat = ds_obs['lat'].values
    lat_res = abs(lat[2] - lat[1])
    lon_res = abs(lon[2] - lon[1])

    for name in forecast_model_names:
        ds_model = xr.open_dataset(info[name])
        lon_min_m, lon_max_m = ds_model['lon'].min().values, ds_model['lon'].max().values
        lat_min_m, lat_max_m = ds_model['lat'].min().values, ds_model['lat'].max().values

        # get the intersection of lat_min, lat_max, lon_min, lon_max
        lon_min = max(lon_min, lon_min_m)
        lon_max = min(lon_max, lon_max_m)
        lat_min = max(lat_min, lat_min_m)
        lat_max = min(lat_max, lat_max_m)

    return [lat_min - lat_res * 0.1, lat_max + lat_res * 0.1,
            lon_min - lon_res * 0.1, lon_max + lon_res * 0.1]


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


def calculate_all_metrics(pd_info,
                          spatial_selection,
                          all_method_params,
                          obs_var='r3',
                          forecast_variables=['pre_r3', 'r3', 'r3', 'r3'],
                          forecast_model_names=['DL', 'EC', 'SMS', 'GRAPES'],
                          groupby_coords=['year', 'leadtime'],
                          save_path=None
                          ):

    binary_method_params = all_method_params.get('binary_method', None)
    method_name_list, method_func_list, method_parameter_list = get_method_list(all_method_params)
    logger.info(f'all_method_params: {all_method_params}')
    logger.info(f'method_name_list: {method_name_list}')

    hfmc_flag = False
    if binary_method_params is not None:
        if 'HFMC' not in binary_method_params:
            raise ValueError('HFMC method must exist!')
        hfmc_flag = True
        hfmc_parameter = binary_method_params['HFMC']['parameter']
        grade_list = binary_method_params['HFMC']['parameter']['grade_list']
        hfmc_method = get_binary_metric(binary_method_params['HFMC']['method'])
        all_hfmc_array = []

        binary_method_func_list = []
        binary_method_name_list = []
        for name in binary_method_params:
            if name == 'HFMC':
                continue
            params = binary_method_params[name]
            func = get_binary_metric(params['method'])
            binary_method_func_list.append(func)
            binary_method_name_list.append(name)

    logger.info(f'Calculate hfmc: {hfmc_flag}')
    logger.info(f'method_name_list: {method_name_list}')

    M = len(forecast_model_names)
    assert len(forecast_variables) == M
    nums = len(pd_info)
    valid_index = []
    all_method_scores = defaultdict(list)

    for i in range(nums):
        infos = pd_info.iloc[i, :]
        date = infos['date']
        obs_file = infos['OBS']
        logger.info(f'{i}/{nums} {date} obs_file: {obs_file}')
        try:
            obs_value = xr.open_dataset(obs_file).sel(**spatial_selection)[obs_var].values
            obs_value = np.squeeze(obs_value)
            H, W = obs_value.shape[0], obs_value.shape[1]
            all_model_value = []
            for name, var in zip(forecast_model_names, forecast_variables):
                model_file = infos[name]
                model_value = xr.open_dataset(model_file).sel(**spatial_selection)[var].values
                model_value = np.squeeze(model_value)[0: H, 0: W]
                all_model_value.append(model_value)
            # shape [M, lat, lon]. M = len(forecast_model_names)
            all_model_value = np.asarray(all_model_value)
            valid_index.append(i)
        except Exception as e:
            logger.info(f'{i}/{nums}: {date} failed due to {e}')
            continue

        for name, fun, pa in zip(method_name_list, method_func_list, method_parameter_list):
            model_score_list = []
            for j, model_name in enumerate(forecast_model_names):
                score = fun(obs_value, all_model_value[j]) if pa is None else fun(obs_value, all_model_value[j], **pa)
                model_score_list.append(score)
            if name != 'FSS':
                all_method_scores[name].append(model_score_list)
            else:
                # np.asarray(model_score_list).shape=[M, H, N, 4]. H=len(half_window_size_list), N = len(grade_list).
                all_method_scores[name].append(np.asarray(model_score_list))

        if hfmc_flag:
            # hfmc_array.shape = [M, N, 4]. N = len(grade_list). 4=[TP, FP, FN, TN]
            hfmc_array = hfmc_method(obs=obs_value, pre=all_model_value, **hfmc_parameter)
            assert hfmc_array.shape[1] == len(grade_list)
            all_hfmc_array.append(hfmc_array)

    keep_columns = ['year', 'month', 'day', 'cycle', 'issue_time', 'leadtime', 'valid_time', 'valid_day', 'date']
    score_pd_info = pd_info.iloc[valid_index, :][keep_columns]
    score_pd_info.index = range(len(score_pd_info))
    logger.info(f'keep_columns: {keep_columns}')
    logger.info(f'valid sample shape: {score_pd_info.shape}')

    all_score_columns = []
    sum_score_columns = []
    mean_score_columns = []

    # STEP1: Calculate binary metric
    logger.info('Calculate binary metric!')
    if hfmc_flag:
        binary_df = score_pd_info.copy()
        # all_hfmc_array.shape = [T, M, N, 4]. T = len(valid_index).
        all_hfmc_array = np.asarray(all_hfmc_array)
        T, _, N, _ = all_hfmc_array.shape
        assert T == len(valid_index)

        binary_metric_score = {}
        for metric_name, func in zip(binary_method_name_list, binary_method_func_list):
            logger.info(f'Calculate {metric_name}')
            # score.shape = [T, M, N]
            score = func(all_hfmc_array)
            binary_metric_score[metric_name] = score

        for i in range(M):
            for j in range(N):
                name = forecast_model_names[i]
                grade = grade_list[j]
                hfmc_array = all_hfmc_array[:, i, j, :]  # shape = [T, 4]
                tp = all_hfmc_array[:, i, j, 0]
                fp = all_hfmc_array[:, i, j, 1]
                fn = all_hfmc_array[:, i, j, 2]
                tn = all_hfmc_array[:, i, j, 3]
                binary_df[f'{name}-{grade}-tp'] = tp
                binary_df[f'{name}-{grade}-fp'] = fp
                binary_df[f'{name}-{grade}-fn'] = fn
                binary_df[f'{name}-{grade}-tn'] = tn
                all_score_columns.append(f'{name}-{grade}-tp')
                all_score_columns.append(f'{name}-{grade}-fp')
                all_score_columns.append(f'{name}-{grade}-fn')
                all_score_columns.append(f'{name}-{grade}-tn')

                sum_score_columns.append(f'{name}-{grade}-tp')
                sum_score_columns.append(f'{name}-{grade}-fp')
                sum_score_columns.append(f'{name}-{grade}-fn')
                sum_score_columns.append(f'{name}-{grade}-tn')

                for metric_name in binary_metric_score:
                    score = binary_metric_score[metric_name]
                    binary_df[f'{name}-{grade}-{metric_name}'] = score[:, i, j]
                    all_score_columns.append(f'{name}-{grade}-{metric_name}')
        save_file = os.path.join(save_path, 'binary_metric.csv')
        binary_df.to_csv(save_file, index=None)
        logger.info(f'Binary_df: {save_file} save down!')
        logger.info('Post-process binary metric!')
        post_process_binary_metric(
            score_pd_info=binary_df,
            sum_columns=sum_score_columns,
            binary_method_params=binary_method_params,
            forecast_model_names=forecast_model_names,
            groupby_coords=groupby_coords,
            save_path=save_path
        )

    # STEP2: Calculate fss metric
    if 'FSS' in method_name_list:
        fss_df = score_pd_info.copy()
        index = method_name_list.index('FSS')
        pa = method_parameter_list[index]
        F_grade_list = pa['grade_list']
        F_half_window_size_list = pa['half_window_size_list']

        score_array = all_method_scores['FSS']
        score_array = np.asarray(score_array)   # score_array.shape = [T, M, H, N, 4]

        for g, grade in enumerate(F_grade_list):
            for h, half_window_size in enumerate(F_half_window_size_list):
                for i in range(M):
                    name = forecast_model_names[i]
                    fss_df[f'{name}-{grade}-{half_window_size}-pob'] = score_array[:, i, h, g, 0]
                    fss_df[f'{name}-{grade}-{half_window_size}-pfo'] = score_array[:, i, h, g, 1]
                    fss_df[f'{name}-{grade}-{half_window_size}-fbs'] = score_array[:, i, h, g, 2]
                    fss_df[f'{name}-{grade}-{half_window_size}-fss'] = score_array[:, i, h, g, 3]
        save_file = os.path.join(save_path, 'fss_metric.csv')
        fss_df.to_csv(save_file, index=None)
        logger.info(f'fss_df: {save_file} save down!')
        logger.info('Post-process fss metric!')
        post_process_fss_metric(
            score_pd_info=fss_df,
            spatial_method_params=all_method_params['spatial_method'],
            forecast_model_names=forecast_model_names,
            groupby_coords=groupby_coords,
            save_path=save_path
        )

    # STEP3: Calculate continuous_cv metric
    if len(method_name_list) > 0:
        if 'FSS' in method_name_list:
            method_name_list.remove('FSS')
        continuous_cv_df = score_pd_info.copy()
        for method_name in method_name_list:
            if method_name in ['FSS']:
                continue
            score_array = all_method_scores[method_name]
            score_array = np.asarray(score_array)
            # score_array.shape = [T, M]
            for i in range(M):
                name = forecast_model_names[i]
                continuous_cv_df[f'{name}-{method_name}'] = score_array[:, i]
                all_score_columns.append(f'{name}-{method_name}')
                mean_score_columns.append(f'{name}-{method_name}')
        save_file = os.path.join(save_path, 'continuous_cv_metric.csv')
        continuous_cv_df.to_csv(save_file, index=None)
        logger.info(f'continuous_cv_df: {save_file} save down!')
        logger.info('Post-process continuous_cv metric!')
        post_process_continuous_cv_metric(
            score_pd_info=continuous_cv_df,
            method_name_list=method_name_list,
            forecast_model_names=forecast_model_names,
            groupby_coords=groupby_coords,
            save_path=save_path
            )

    return None


def main(metric_setting_file,
         save_path,
         index_save_path,
         obs_path='/THL8/home/zhq/data/Fusion_data/CMPA/CMPA5_V3_nc/HOR/NRT_0.05_15-60_70-140_UTC_rain',
         obs_var='r3',
         forecast_paths="/THL8/home/zhq/fzl/hydra_experiment/unet_2d_2022-05-29_train/14-27-59/test/10-41-46/inference_nc,\
                        /THL8/home/zhq/data/NWP_data/EC/EC_V3_nc/ecmwf_nc_0.05_31-45_108-124_rain,\
                        /THL8/home/zhq/data/NWP_data/SMS/SMS_V3_nc/SMS_nc_0.05_31-45_108-124_rain,\
                        /THL8/home/zhq/data/NWP_data/GRAPES/GRAPES_V3_nc/GRAPES_nc_0.05_31-45_108-124_rain",
         forecast_variables='pre_r3,r3,r3,r3',
         forecast_model_names='DL,EC,SMS,GRAES',
         forecast_show_model_names='FRNet,ECMWF,CMA-SH9,CMA-MESO',
         start_day='2021-06-01',
         end_day='2021-07-01',
         leadtime_list='3,6,9,12,15,18,21,24',
         cycle_list='00,12',
         groupby_coords='year,month,cycle,leadtime,issue_time',
         loc_range=None,
         ):

    logger.info('Start metric!')
    logger.info(f'obs_var: {obs_var}. obs_path: {obs_path}.')
    forecast_paths = forecast_paths.split(',')
    forecast_paths = [fp.strip() for fp in forecast_paths]

    forecast_variables = forecast_variables.split(',')
    forecast_variables = [var.strip(' ') for var in forecast_variables]

    forecast_model_names = forecast_model_names.split(',')
    forecast_model_names = [name.strip(' ') for name in forecast_model_names]

    forecast_show_model_names = forecast_show_model_names.split(',')
    forecast_show_model_names = [name.strip(' ') for name in forecast_show_model_names]

    groupby_coords = groupby_coords.split(',')
    groupby_coords = [coord.strip(' ') for coord in groupby_coords]

    assert len(forecast_paths) == len(forecast_variables) == len(forecast_model_names) == len(forecast_show_model_names)
    logger.info(f'forecast_paths: {forecast_paths}')
    logger.info(f'forecast_variables: {forecast_variables}')
    logger.info(f'forecast_model_names: {forecast_model_names}')

    cycle_list = cycle_list.split(',')
    cycle_list = [cycle.strip(' ') for cycle in cycle_list]
    cycle_list = [str(cycle).rjust(2, '0') for cycle in cycle_list]

    leadtime_list = leadtime_list.split(',')
    leadtime_list = [leadtime.strip(' ') for leadtime in leadtime_list]
    leadtime_list = [str(leadtime).rjust(2, '0') for leadtime in leadtime_list]
    logger.info(f'cycle_list: {cycle_list}')
    logger.info(f'leadtime_list: {leadtime_list}')

    if len(index_save_path.strip(' ')) == 0 or not os.path.exists(index_save_path):
        pd_info = get_obs_models_index(obs_path,
                                       forecast_path_list=forecast_paths,
                                       forecast_model_list=forecast_model_names,
                                       forecast_show_model_list=forecast_show_model_names,
                                       start_day=start_day,
                                       end_day=end_day,
                                       cycle_list=cycle_list,
                                       leadtime_list=leadtime_list,
                                       save_file=os.path.join(save_path, 'obs_models_index_info.csv'),
                                       )
    else:
        pd_info = pd.read_csv(index_save_path)

    if loc_range is not None:
        res = 0.05
        loc_range = loc_range.split(',')
        if len(loc_range) != 4:
            raise ValueError('loc_range must be lat_min,lat_max,lon_min,lon_max format!')
        loc_range = [loc.strip(' ') for loc in loc_range]
        loc_range = [np.float32(i) for i in loc_range]
        lat_min, lat_max, lon_min, lon_max = loc_range
        lat_min, lat_max = lat_min - res * 0.2, lat_max + res * 0.2
        lon_min, lon_max = lon_min - res * 0.2, lon_max + res * 0.2
    else:
        loc_range = get_spatial_selection(pd_info, forecast_show_model_names)
        lat_min, lat_max, lon_min, lon_max = loc_range
    logger.info(f'loc_range: {loc_range}')

    selection = {
                'lat': slice(lat_min, lat_max),
                'lon': slice(lon_min, lon_max)
                }

    logger.info('calculate_all_metrics!')
    all_method_params = read_yml_json_file(metric_setting_file)
    calculate_all_metrics(
        pd_info=pd_info,
        spatial_selection=selection,
        all_method_params=all_method_params,
        obs_var=obs_var,
        forecast_variables=forecast_variables,
        forecast_model_names=forecast_show_model_names,
        groupby_coords=groupby_coords,
        save_path=save_path
    )

    return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric_setting_file", type=str,
        default='gridforecast_v2/evaluation/eval_setting_v1.yaml',
        help="The path of metric setting file which including all the metrics need to be evaluated"
    )
    parser.add_argument(
        '--save_path', required=True,
        help="Directory in which to save output.Creates directory if it doesn't exist."
    )
    parser.add_argument(
        '--index_save_path', default='', type=str,
        help="Directory in which to save output.Creates directory if it doesn't exist."
    )
    parser.add_argument(
        '--obs_path', type=str,
        default='/THL8/home/zhq/data/Fusion_data/CMPA/CMPA5_V3_nc/HOR/NRT_0.05_15-60_70-140_UTC_rain',
        help='The directory of the observation'
    )
    parser.add_argument(
        '--obs_var', type=str, default='r3',
        help="The observation variable name of obs. such as 'r3' for past 3-hour rain"
    )
    parser.add_argument(
        '--forecast_paths', type=str,
        default="/THL8/home/zhq/fzl/hydra_experiment/unet_2d_2022-05-29_train/14-27-59/test/10-41-46/inference_nc," +
                "/THL8/home/zhq/data/NWP_data/EC/EC_V3_nc/ecmwf_nc_0.05_31-45_108-124_rain," +
                "/THL8/home/zhq/data/NWP_data/SMS/SMS_V3_nc/SMS_nc_0.05_31-45_108-124_rain," +
                "/THL8/home/zhq/data/NWP_data/GRAPES/GRAPES_V3_nc/GRAPES_nc_0.05_31-45_108-124_rain",
        help="The paths corresponding to the models"
    )
    parser.add_argument(
        '--forecast_variables', type=str,
        default='pre_r3,r3,r3,r3',
        help='The variable name of predicted model.'
    )
    parser.add_argument(
        '--forecast_model_names', type=str,
        default='DL,EC,SMS,GRAPES',
        help='The model name used to match the file format'
    )
    parser.add_argument(
        '--forecast_show_model_names', type=str,
        default='FRNet,ECMWF,CMA-SH9,CMA-MESO',
        help='The final showing name correspanding to the forecast_model_names'
    )
    parser.add_argument(
        '--start_day', type=str,
        default='2021-03-15',
        help="Start date in YYYY-MM-DD format"
        )
    parser.add_argument(
        '--end_day', type=str,
        default='2021-03-25',
        help="End date in YYYY-MM-DD format"
        )
    parser.add_argument(
        '--cycle_list', default='0,12',
        type=str, help="Which cycles need to be processed."
        )
    parser.add_argument(
        '--leadtime_list', default='0,3,6,9,12,15,18,21,24',
        type=str, help="Which leadtime need to be processd"
        )
    parser.add_argument(
        '--loc_range', default=None,
        help="lat_min,lat_max,lon_min,lon_max"
        )
    parser.add_argument(
        '--groupby_coords', default='year,month,leadtime', type=str,
        help="Groupby dimension."
        )

    args = parser.parse_args()
    main(metric_setting_file=args.metric_setting_file,
         save_path=args.save_path,
         index_save_path=args.index_save_path,
         obs_path=args.obs_path,
         obs_var=args.obs_var,
         forecast_paths=args.forecast_paths,
         forecast_variables=args.forecast_variables,
         forecast_model_names=args.forecast_model_names,
         forecast_show_model_names=args.forecast_show_model_names,
         start_day=args.start_day,
         end_day=args.end_day,
         leadtime_list=args.leadtime_list,
         cycle_list=args.cycle_list,
         loc_range=args.loc_range,
         groupby_coords=args.groupby_coords
         )
