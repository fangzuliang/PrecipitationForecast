"""
Func: Generate CMPAS index for dataloader_v1.py
Example:
PYTHONPATH=`pwd` \
     ~/anaconda3/envs/pyxe/bin/python forecast/gridforecast_v2/data_pipelines/dataloader_v1/generate_CMAPS_index.py \
    -i /THL8/home/zhq/data/Fusion_data/CMPA/CMPA5_V3_nc/HOR/NRT_0.05_15-60_70-140_UTC \
    -o /THL8/home/zhq/fzl/data/gridforecast/dataloader_v1/CMPAS_index/all_NRT_index.csv \
    -s '2017-01-01' \
    -e '2021-12-31' \
    --var 'NRT'
"""
import os
import pandas as pd
import argparse
import sys

path = os.getcwd().split('/forecast')[0]
sys.path.append(path)


# eg: 2019/03/17/CMPA5_0.05_20190317_f020000_HOR_NRT_PRE.nc
FILE_FORMAT = {
    'NRT': '%Y/%m/%d/CMPA5_{res}_%Y%m%d_f%H%M%S_HOR_NRT_PRE.nc',
    'FRT': '%Y/%m/%d/CMPA5_{res}_%Y%m%d_f%H%M%S_HOR_FRT_PRE.nc',
    'FAST': '%Y/%m/%d/CMPA5_{res}_%Y%m%d_f%H%M%S_HOR_FAST_PRE.nc',
}


def generate_index(src_path, var='NRT',
                   start_day='2021-01-01', end_day='2021-10-18',
                   drop_no_exit=True,
                   save_file=None,
                   freq='1h',
                   res=0.05,
                   ):

    assert var in FILE_FORMAT

    filename = "f'" + FILE_FORMAT[var] + "'"
    print(f'filename: {filename}')
    filename = eval(filename)

    all_times = pd.date_range(start_day, end_day, freq=freq)

    all_year = []
    all_month = []
    all_day = []
    all_hour = []
    all_valid_time = []
    all_file = []
    for t in all_times:
        valid_time = t.strftime('%Y%m%dT%H%M%S')
        year, month, day, hour = valid_time[0:4], valid_time[4:6], valid_time[6:8], valid_time[9:11]

        file = t.strftime(filename)
        abs_file = os.path.join(src_path, file)

        if drop_no_exit and not os.path.exists(abs_file):
            print(f'{abs_file} did not exists!')
            continue

        all_year.append(int(year))
        all_month.append(int(month))
        all_day.append(int(day))
        all_hour.append(int(hour))
        all_file.append(file)
        all_valid_time.append(valid_time)

    pd_info = pd.DataFrame()
    pd_info['year'] = all_year
    pd_info['month'] = all_month
    pd_info['day'] = all_day
    pd_info['hour'] = all_hour
    pd_info['file'] = all_file
    pd_info['valid_time'] = all_valid_time

    if save_file is not None:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        pd_info.to_csv(save_file, index=None)
        print(f'{save_file} down!')

    return pd_info


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--src_path", required=True,
                        help="Directory of input files")
    parser.add_argument('-o', '--save_file', required=True,
                        help="The filename to save infos.")
    parser.add_argument('-s', '--start_day', default='2021-02-23', type=str,
                        help="Start date in YYYY-MM-DD. Includes entire month.")
    parser.add_argument('-e', '--end_day', default='2021-11-05', type=str,
                        help="End date in YYYY-MM-DD. Includes entire month.")
    parser.add_argument('--var', default='NRT', type=str,
                        choices=['NRT', 'FRT', 'FAST'],
                        help="NWP name")

    args = parser.parse_args()
    generate_index(src_path=args.src_path,
                   save_file=args.save_file,
                   start_day=args.start_day,
                   end_day=args.end_day,
                   var=args.var,
                   )
