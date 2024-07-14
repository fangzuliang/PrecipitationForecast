"""
Func: Filter index by the given rain threshold
Example:
cd /THL8/home/zhq/fzl/forecast
cd ../
PYTHONPATH=`pwd` python gridforecast_v2/data_pipelines/dataloader_v1/resample_CMPAS_index.py \
    -i /fs1/home/zhq/users/fzl/data/dataloader_v1/CMPAS_index/35-45_112-122/NRT_index_train_statistic.csv \
    -o /fs1/home/zhq/users/fzl/data/dataloader_v1/CMPAS_index/35-45_112-122/NRT_index_train_rain1_300.csv \
    --threshold_name 'num_1.0' \
    --min_threshold_num 300

"""
import os
import pandas as pd
import argparse
import sys
import numpy as np
import xarray as xr

path = os.getcwd().split('/forecast')[0]
sys.path.append(path)


def resample_index(pd_file,
                   save_file=None,
                   threshold_name='num_1.0',
                   min_threshold_num=300,
                   ):
    df = pd.read_csv(pd_file)
    assert threshold_name in df
    df = df[df[threshold_name] >= min_threshold_num]
    keep_columns = ['year', 'month', 'day', 'hour', 'file', 'valid_time']
    df = df[keep_columns]

    if save_file is not None:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
    df.to_csv(save_file, index=None)
    print(f'{save_file} done!')

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--pd_file", required=True,
                        help="Directory of CMPAS index file")
    parser.add_argument('-o', '--save_file', default=None,
                        help="Output file of statistics df")
    parser.add_argument('--threshold_name', default='num_1.0', type=str,
                        help="rain column name")
    parser.add_argument('--min_threshold_num', default=300, type=int,
                        help="sum of sample number when rain >= 1.0")

    args = parser.parse_args()
    resample_index(pd_file=args.pd_file,
                   save_file=args.save_file,
                   threshold_name=args.threshold_name,
                   min_threshold_num=args.min_threshold_num,
                    )
