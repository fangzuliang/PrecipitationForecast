"""
Func: Calculate some stastic nums given rain thresholds.
Example:
cd /THL8/home/zhq/fzl/forecast
cd ../
PYTHONPATH=`pwd` python gridforecast_v2/data_pipelines/dataloader_v1/calculate_CMPAS_rain_threshold_num.py \
    -i /fs1/home/zhq/users/fzl/data/dataloader_v1/CMPAS_index/NRT_index_train.csv \
    -s /fs1/home/zhq/data/Fusion_data/CMPA/CMPA5_V3_nc/HOR/NRT_0.05_15-60_70-140_UTC_rain \
    -o /fs1/home/zhq/users/fzl/data/dataloader_v1/CMPAS_index/35-45_112-122/NRT_index_train_statistic.csv \
    --loc_range '35,45,112,122' \
    --target_var 'r3' \
    --threshold_list '0.1,1,2,5,10'
"""
import os
import pandas as pd
import argparse
import sys
import numpy as np
import xarray as xr

path = os.getcwd().split('/forecast')[0]
sys.path.append(path)


def calculate(x, selection, threshold_list, var):

    file = x['abs_file']
    ds = xr.open_dataset(file).sel(**selection)
    if var in ds:
        threshold_num_list = []
        for threshold in threshold_list:
            num_threshold = (ds[var] >= threshold).sum().values
            threshold_num_list.append(num_threshold)
        return threshold_num_list
    else:
        return [-1] * len(threshold_list)


def calculate_threshold_num(pd_file,
                            src_path,
                            save_file=None,
                            loc_range='35,45,112,122',
                            target_var='r3',
                            threshold_list='0.1,1,2,5,10'
                            ):

    loc_range = loc_range.split(',')
    loc_range = [loc.strip(' ') for loc in loc_range]
    loc_range = [np.float32(i) for i in loc_range]
    lat_min, lat_max, lon_min, lon_max = loc_range
    assert len(loc_range) == 4

    threshold_list = threshold_list.split(',')
    threshold_list = [threshold.strip(' ') for threshold in threshold_list]
    threshold_list = [np.float32(threshold) for threshold in threshold_list]

    df = pd.read_csv(pd_file)
    df['abs_file'] = df['file'].map(lambda x: os.path.join(src_path, x))

    lat_min, lat_max, lon_min, lon_max = loc_range
    selection = {
        'lon': slice(lon_min, lon_max),
        'lat': slice(lat_min, lat_max),
    }

    threshold_name_list = [f'num_{x}' for x in threshold_list]
    df[threshold_name_list] = df.apply(
                                      calculate,
                                      axis=1,
                                      args=(selection, threshold_list, target_var,),
                                      result_type="expand"
                                      )

    if save_file is not None:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        df.to_csv(save_file, index=None)
        print(f'{save_file} done!')

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--pd_file", required=True,
                        help="Directory of CMPAS index file")
    parser.add_argument('-s', '--src_path', required=True,
                        help="Directory of CMPAS data")
    parser.add_argument('-o', '--save_file', default=None,
                        help="Output file of statistics df")
    parser.add_argument('--loc_range', default='35,45,112,122', type=str,
                        help="lat_min,lat_max,lon_min,lon_max")
    parser.add_argument('--target_var', default='r3', type=str,
                        help="Which variable is target")
    parser.add_argument('--threshold_list', default='0.1,1,2,5,10', type=str,
                        help="rain threshold list")

    args = parser.parse_args()
    calculate_threshold_num(pd_file=args.pd_file,
                            src_path=args.src_path,
                            save_file=args.save_file,
                            loc_range=args.loc_range,
                            target_var=args.target_var,
                            threshold_list=args.threshold_list,
                            )
