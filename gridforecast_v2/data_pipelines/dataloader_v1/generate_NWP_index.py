"""
Func: Generate NWP index for dataloader_v1.py
Example:
PYTHONPATH=`pwd` \
yhrun -pTH_HPC3N -n1 -N1 \
     python forecast/gridforecast_v2/data_pipelines/data_pipelines/dataloader_v1/generate_NWP_index.py \
    -i /THL8/home/zhq/data/NWP_data/EC/EC_V3_nc/ecmwf_nc_0.05_31-45_108-124_r3 \
    -o /THL8/home/zhq/fzl/data/gridforecast/dataloader_v1/EC_index/all_EC_rain_index.csv \
    -s '2016-01-01' \
    -e '2021-12-31' \
    --NWP 'EC' \
    --mode 'rain'
"""
import os
import pandas as pd
import numpy as np
import argparse
import sys

path = os.getcwd().split('/forecast')[0]
sys.path.append(path)


# eg: '{year}/{month}/{day}/{cycle}/EC_{mode}_0.05_{date}_T{cycle}_f0{leadtime}.nc'
FILE_FORMAT = {
    'EC': '{year}/{month}/{day}/{cycle}/EC_{mode}_{res}_{date}_T{cycle}_f0{leadtime}.nc',
    'SMS': '{year}/{month}/{day}/{cycle}/SMS_{mode}_{res}_{date}_T{cycle}_f{leadtime}.nc',
    'GRAPES': '{year}/{month}/{day}/{cycle}/GRAPES_{mode}_{res}_{date}_T{cycle}_f0{leadtime}.nc'
}


def generate_index(src_path,
                   NWP='EC', mode='rain',
                   start_day='2021-01-01', end_day='2021-10-18',
                   cycle_list=[0, 12],
                   res=0.05,
                   drop_no_exit=True,
                   save_file=None,
                   ):

    assert NWP in ['EC', 'SMS', 'GRAPES']
    assert mode in ['surface', 'pressure', 'rain']

    if mode == 'rain':
        mode = 'surface'

    filename = "f'" + FILE_FORMAT[NWP] + "'"

    leadtime_list = list(np.arange(0, 100))
    leadtime_list = [str(leadtime).rjust(2, '0') for leadtime in leadtime_list]

    cycle_list = [str(cycle).rjust(2, '0') for cycle in cycle_list]
    all_days = pd.date_range(start_day, end_day, freq='1d')

    all_year = []
    all_month = []
    all_day = []
    all_cycle = []
    all_lead_time = []
    all_valid_time = []
    all_file = []
    for t in all_days:
        date = t.strftime('%Y%m%d')
        year, month, day = date[0:4], date[4:6], date[6:8]
        for cycle in cycle_list:
            for leadtime in leadtime_list:
                file = eval(filename)
                abs_file = os.path.join(src_path, file)

                if drop_no_exit and not os.path.exists(abs_file):
                    print(f'{abs_file} did not exists!')
                    continue

                valid_time = t + pd.Timedelta(f'{int(cycle)}h') + pd.Timedelta(f'{int(leadtime)}h')
                valid_time = valid_time.strftime('%Y%m%dT%H%M%S')

                all_year.append(int(year))
                all_month.append(int(month))
                all_day.append(int(day))
                all_cycle.append(int(cycle))
                all_lead_time.append(int(leadtime))
                all_file.append(file)
                all_valid_time.append(valid_time)

    pd_info = pd.DataFrame()
    pd_info['year'] = all_year
    pd_info['month'] = all_month
    pd_info['day'] = all_day
    pd_info['cycle'] = all_cycle
    pd_info['lead_time'] = all_lead_time
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
    parser.add_argument('--NWP', default='EC', type=str,
                        choices=['EC', 'SMS', 'GRAPES'],
                        help="NWP name")
    parser.add_argument('--mode', default='rain', type=str,
                        choices=['surface', 'pressure', 'rain'],
                        help="The mode of NWP.")

    args = parser.parse_args()
    generate_index(src_path=args.src_path,
                   save_file=args.save_file,
                   start_day=args.start_day,
                   end_day=args.end_day,
                   NWP=args.NWP,
                   mode=args.mode
                   )
