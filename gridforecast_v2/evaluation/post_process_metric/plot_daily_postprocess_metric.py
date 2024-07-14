# %%
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.gridspec as gridspec


# %%
def generate_gridspec(nums, rows=2, miss_first=True):

    mode = nums % rows  # 取余
    cols = int(nums / rows) if mode == 0 else np.ceil(nums / rows).astype(int)  # 确定需要多少列

    gs = gridspec.GridSpec(rows, cols * 2)
    gs.update(wspace=0.8)

    ax_list = []
    if mode == 0:
        for row in range(rows):
            for col in range(cols):
                ax = plt.subplot(gs[row, col*2: col*2 + 2])
                ax_list.append(ax)
    else:
        rest = nums - (rows - 1) * cols
        blank = cols - rest
        if miss_first:
            for col in range(rest):
                ax = plt.subplot(gs[0, blank + col * 2: blank + col * 2 + 2])
                ax_list.append(ax)
            for row in range(1, rows):
                for col in range(cols):
                    ax = plt.subplot(gs[row, col*2: col*2 + 2])
                    ax_list.append(ax)
        else:
            for row in range(0, rows - 1):
                for col in range(cols):
                    ax = plt.subplot(gs[row, col*2: col*2 + 2])
                    ax_list.append(ax)
            for col in range(rest):
                ax = plt.subplot(gs[row - 1, blank + col * 2: blank + col * 2 + 2])
                ax_list.append(ax)

    return ax_list


def plot_single_ax(ax,
                   df1,
                   x='leadtime',
                   model_list=['DL', 'EC', 'SMS', 'GRAPES'],
                   metric='TS',
                   threshold=0.1,
                   index=0
                   ):

    x_value = df1[x].values
    y_list = []
    for model in model_list:
        column = f'{model}-{threshold}-{metric}'
        ax.plot(x_value, df1[column], '*-', label=model)

    # ax.set_xlabel(x)
    ax.set_xticks(x_value)
    ax.set_xticklabels(x_value)
    ax.set_ylabel('score')
    ax.legend()
    ax.set_title(f'{metric} Threshold={threshold}')

    return ax


def plot_single_issue_time_multi_ax(df, model_list=['DL', 'EC', 'SMS', 'GRAPES'],
                                    threshold_list=['1', '10', '20', '40', '50', '60'],
                                    metric='TS',
                                    rows=3,
                                    save_file=None):

    fig = plt.figure(figsize=(24, 16))
    nums = len(threshold_list)
    ax_list = generate_gridspec(nums, rows=rows, miss_first=True)
    for i in range(nums):
        ax, threshold = ax_list[i], threshold_list[i]
        plot_single_ax(ax,
                       df,
                       x='leadtime',
                       model_list=model_list,
                       metric=metric,
                       threshold=threshold,
                       index=i,
                       )

    if save_file is not None:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        plt.savefig(save_file, bbox_inches='tight', dpi=200)

    # plt.show()
    plt.clf()
    plt.close()
    return None


def generate_issue_time(x):
    y = str(x['year']) + str(x['month']).rjust(2, '0') + str(x['day']).rjust(2, '0') + 'T' + str(x['cycle']).rjust(2, '0')
    return y


def main(pd_file,
         save_dir,
         start_day='2021-06-01',
         end_day='2021-07-01',
         model_list=['DL', 'EC', 'SMS', 'GRAPES'],
         threshold_list=['1', '10', '20', '40', '50', '60'],
         metric_list=['TS', 'BIAS', 'FAR', 'POD'],
         rows=3,
         overwrite=False,
         ):

    # pd_file = '/fs1/home/zhq/users/fzl/experiments/Unet/baseline/ensemble/metric/metric_20210701-20210901/binary_metric.csv'
    df = pd.read_csv(pd_file)
    df['issue_time'] = pd.to_datetime(df.apply(generate_issue_time, axis=1))

    start_day = pd.to_datetime(start_day)
    end_day = pd.to_datetime(end_day)
    df = df[df['issue_time'] > start_day]
    df = df[df['issue_time'] <= end_day]
    df['issue_time'] = df['issue_time'].map(lambda x: pd.to_datetime(x).strftime('%Y%m%dT%H'))

    column_list = ['issue_time', 'year', 'month', 'day', 'cycle', 'leadtime', 'valid_time', 'date']
    for threshold in threshold_list:
        for metric in metric_list:
            for model in model_list:
                column = f'{model}-{threshold}-{metric}'
                column_list.append(column)
    df1 = df[column_list]

    issue_time_list = list(set(list(df1['issue_time'].values)))
    issue_time_list.sort()

    for issue_time in issue_time_list:
        df_issue = df1[df1['issue_time'] == issue_time]
        year, month = issue_time[0: 4], issue_time[4: 6]
        for metric in metric_list:
            save_file = os.path.join(save_dir, metric, year, month, f'{issue_time}_{metric}.png')
            if os.path.exists(save_file) and not overwrite:
                print(f'{save_file} exists. Do no overwrite!')
                continue
            try:
                plot_single_issue_time_multi_ax(df_issue,
                                                model_list=model_list,
                                                threshold_list=threshold_list,
                                                metric=metric,
                                                rows=rows,
                                                save_file=save_file
                                                )
                print(f'{save_file} down!')
            except Exception as e:
                print(f'{save_file} failed due to {e}')
                continue
    return None


if __name__ == "__main__":

    # main(
    #     pd_file='/fs1/home/zhq/users/fzl/experiments/Unet/baseline/ensemble/metric/metric_test_2021/binary_metric.csv',
    #     save_dir='/fs1/home/zhq/users/fzl/experiments/Unet/baseline/ensemble/metric/post_metric_issuetime',
    #     start_day='2021-01-01',
    #     end_day='2021-11-01',
    #     model_list=['DL', 'EC', 'SMS', 'GRAPES'],
    #     threshold_list=['0.1', '1', '5', '10', '20', '30', '40', '50', '60'],
    #     metric_list=['TS', 'BIAS', 'POD', 'FAR'],
    #     rows=3,
    #     overwrite=False
    # )
    
    # main(
    #     pd_file='/fs1/home/zhq/users/fzl/experiments/Unet/baseline/ensemble/metric/new_name/9_grades/metric_test_20220615-20220831/binary_metric.csv',
    #     save_dir='/fs1/home/zhq/users/fzl/experiments/Unet/baseline/ensemble/metric/post_metric_issuetime',
    #     start_day='2022-06-15',
    #     end_day='2022-08-31',
    #     # model_list=['DL', 'EC', 'SMS', 'GRAPES'],
    #     model_list=['ECMWF', 'CMA-SH9', 'CMA-3KM', 'FRNet', 'GFRNet'],
    #     threshold_list=['0.1', '1', '2', '5', '10', '20', '30', '40', '50'],
    #     metric_list=['TS', 'BIAS', 'POD', 'FAR'],
    #     rows=3,
    #     overwrite=False
    # )


    main(
        pd_file='/fs1/home/zhq/users/fzl/experiments/GAN/v4_dataset/BCELoss/metric/Baseline_Exp3_NWPs/9_grades/metric_test_20220615-20220831/binary_metric.csv',
        save_dir='/fs1/home/zhq/users/fzl/experiments/GAN/v4_dataset/BCELoss/metric/Baseline_Exp3_NWPs/9_grades/metric_test_20220615-20220831/post_metric_issuetime',
        start_day='2022-06-15',
        end_day='2022-08-31',
        # model_list=['DL', 'EC', 'SMS', 'GRAPES'],
        model_list=['ECMWF', 'CMA-SH9', 'CMA-3KM', 'FRNet', 'GFRNet'],
        threshold_list=['0.1', '1', '2', '5', '10', '20', '30', '40', '50'],
        metric_list=['TS', 'BIAS', 'POD', 'FAR'],
        rows=3,
        overwrite=False
    )