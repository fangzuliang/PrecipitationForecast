# %%
import os
import numpy as np
import sys
import xarray as xr
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

path = os.getcwd().split('/forecast')[0]
path = '/THL8/home/zhq/fzl/branch/forecast'
sys.path.append(path)

from gridforecast_v2.data_pipelines.dataloader_v1 import dataloader_v1
from gridforecast_v2.data_pipelines.utils import read_yml_json_file
# %%
def plot_grid(value, title=None):

    plt.imshow(value[::-1, :])
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.show()

    return None

# %%
data_yml_file = '/THL8/home/zhq/fzl/branch/forecast/gridforecast_v2/data_pipelines/dataloader_v1/test_dataloader_v1/feature.yaml'

# params = read_yml_json_file(data_yml_file)
# target_params = params['Target']

# print(params['Target'])
# target_index = pd.read_csv(target_params['index_file'])
# %%

data_train = dataloader_v1.GridIterableDataset(
    data_feature=data_yml_file,
    lead_time_list=list(np.arange(1, 24, 1)),
    cycle_list=[0, 12],
    sample_num=-1,
    if_channel=True,
    shuffle=False
)


# %%
index_info = data_train.index_info
in_dim_list = data_train.in_dim_list
decoder_input_feature = data_train.decoder_input_feature

groups_dict = data_train.groups_dict
params = data_train.params


print(f'index_info: {index_info.shape}')

# print(f'groups_dict: {groups_dict}')
# %%
dataload_train = DataLoader(data_train,
                            batch_size=2,
                            num_workers=2,
                            worker_init_fn=dataloader_v1.worker_init_fn
                            )
# %%
for i, (index, all_group_data, target, decoder_input) in enumerate(dataload_train):

    # print(f'i: {i}.  index: {index}. {target.shape}')
    if i > 0:
        break

# %%
# cmpas = target.cpu().numpy()
# ec_rain = all_group_data[0].cpu().numpy()
# sms_rain = all_group_data[2].cpu().numpy()

# lon, lat, z = all_group_data[3][0, 0].cpu().numpy(), all_group_data[3][0, 1].cpu().numpy(), all_group_data[3][0, 2].cpu().numpy()

# # %%
# plot_grid(lon, title='lon')
# plot_grid(lat, title='lat')
# plot_grid(z, title='z')


# # %%
# plot_grid(cmpas[0, 0] * 90, title='cmpas_rain')
# plot_grid(ec_rain[0, 1] * 90, title='ec_rain')
# plot_grid(sms_rain[0, 1] * 90, title='sms_rain')

# %%
[print(data.shape) for data in all_group_data]
# %%
for i in range(10):
    if i in groups_dict:
        print(i, len(groups_dict[i]))

# # # [print(len) for data in all_group_data]
# # # %%


# # %%

# # %%
# p1 = index_info.iloc[index.cpu().numpy()[0], :]

# cmaps_path = params['Target']['path']
# ec_rain_path = params['EC']['rain']['path']
# sms_rain_path = params['SMS']['rain']['path']


# cmpas_file = os.path.join(cmaps_path, p1['target_file'])
# ec_file = os.path.join(ec_rain_path, p1['EC_rain_file'])
# sms_file = os.path.join(sms_rain_path, p1['SMS_rain_file'])

# # %%
# selection = {
#     'lon': slice(110, 119.6),
#     'lat': slice(32, 41.6)
# }

# ds_cmaps = xr.open_dataset(cmpas_file).sel(**selection)
# ds_ec = xr.open_dataset(ec_file).sel(**selection)
# ds_sms = xr.open_dataset(sms_file).sel(**selection)
# # %%
# plot_grid(ds_cmaps['rain'].values, title='cmpas rain')
# plot_grid(ds_ec['rain'].values, title='ec rain')
# plot_grid(ds_sms['rain'].values, title='sms rain')
# %%
