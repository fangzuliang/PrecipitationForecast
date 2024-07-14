import xarray as xr
from collections import defaultdict
import numpy as np
import os
import pandas as pd
import math
import random
import logging
from torch.utils.data import get_worker_info, IterableDataset
from gridforecast_v2.data_pipelines.utils import (read_yml_json_file, configure_logging,
                                                  generate_issue_time, center_crop)


configure_logging()
logger = logging.getLogger(__name__)


class GridIterableDataset(IterableDataset):
    """
    func: Iterable-style dataset for use with torch DataLoader class.
    Parameter
    ---------
    data_feature: str or instance of data_feature dict.
        eg: '/THL8/home/zhq/fzl/forecast/gridforecast/param/param_file/train_data_param.yml'
        the path of .yml file which contains all the informations of features from different sources.
        including: ['Target', 'EC', 'SMS', 'GRAPES', 'META', 'Time'].
    lead_time_list: list
        default: list(np.arange(1, 24, 1)).
        From which leadtime that we get the samples.
    cycle_list:
        default: [0, 12].
        From which cycle that we get the samples.
    sample_nums: int
        default -1. How many samples used. when -1, all the valid samples will be used.
        eg: 20. only 20 samples will be used.
    if_channel: bool
        default False. Whether combine all the features according the channel dimension.
    shuffle: bool
        whether shuffle the dataset.
    """
    def __init__(self,
                 data_feature,
                 lead_time_list=list(np.arange(1, 24, 1)),
                 cycle_list=[0, 12],
                 sample_num=-1,
                 if_channel=False,
                 shuffle=True,
                 ):

        # data_feature maybe .yaml file. or maybe params dict.
        if isinstance(data_feature, str):
            suffix = data_feature.split('.')[-1]
            if suffix in ['yml', 'yaml']:
                self.params = read_yml_json_file(data_feature)
        else:
            self.params = data_feature

        self.target_params = self.params['Target']
        self.obs_params = self.params.get('OBS', None)
        self.META_param = self.params['META']
        self.Time_param = self.params['Time']

        self.target_path = self.target_params['path']

        self.lead_time_list = lead_time_list
        self.cycle_list = cycle_list
        self.sample_num = sample_num
        self.if_channel = if_channel
        self.shuffle = shuffle

        # get the feature-group informations.
        self.groups_dict = defaultdict(list)

        # get the feature_name of the decoder_input.
        self.decoder_input_feature = []

        # merge all NWP index_file and combine feature file with target file.
        self.NWP_index_info, self.valid_NWP, self.valid_mode = self.get_NWP_index_info()
        self.NWP_index_info['valid_time'] = pd.to_datetime(self.NWP_index_info['valid_time'])
        time_bias = self.target_params['target_valid_time_bias']
        self.NWP_index_info['valid_time'] = self.NWP_index_info['valid_time'] + pd.Timedelta(time_bias)

        self.target_index = pd.read_csv(self.target_params['index_file'])
        self.target_index = self.target_index.rename(columns={'file': 'target_file',
                                                              'year': 'target_year',
                                                              'month': 'target_month',
                                                              'day': 'target_day',
                                                              }
                                                     )
        self.target_index = self.drop_unnamed(self.target_index)

        self.target_index['valid_time'] = pd.to_datetime(self.target_index['valid_time'])
        self.index_info = self.NWP_index_info.merge(self.target_index, on=['valid_time'])

        # Add issue_time colume.
        self.index_info['issue_time'] = self.index_info.apply(generate_issue_time, axis=1)
        self.valid_obs_source = self.get_OBS_index_info()
        logger.info(f'valid_obs_source: {self.valid_obs_source}')

        logger.info(f'NWP_index_info: {self.NWP_index_info.shape}. {self.NWP_index_info.columns}')
        logger.info(f'target_index: {self.target_index.shape}. {self.target_index.columns}')
        logger.info(f'index_info: {self.index_info.shape} {self.index_info.columns}')

        self.start = 0
        self.end = len(self.index_info)
        index = list(range(self.start, self.end))
        logger.info(f'Shuffle: {self.shuffle}')
        if self.shuffle:
            random.shuffle(index)
        self.index_info.index = index
        self.index_info = self.index_info.sort_index()

        if sample_num != -1:
            self.index_info = self.index_info.iloc[0: self.sample_num, :]
            logger.info(f'sample nums: {sample_num}')
            logger.info(f'index_info: {self.index_info.shape} {self.index_info.columns}')
            self.end = sample_num

        # use slide to crop samples.
        self.slide = self.META_param.get('slide', False)
        self.input_size = self.META_param.get('input_size', None)
        self.target_size = self.META_param.get('target_size', self.input_size)
        assert self.target_size[0] <= self.input_size[0] and self.target_size[1] <= self.input_size[1]
        logger.info(f'input_size: {self.input_size}. target_size: {self.target_size}')
        self.lat_range, self.lon_range, self.lon_grid, self.lat_grid = self.get_META_info()

        # if do not use spatial slide. then use specific lat_range and lon_range.
        logger.info(f'slide: {self.slide}')
        if not self.slide:
            if self.input_size is None:
                self.input_size = (len(self.lat_range), len(self.lon_range))
            else:
                self.lat_range = self.lat_range[0: self.input_size[0]]
                self.lon_range = self.lon_range[0: self.input_size[1]]
                self.lat_grid = self.lat_grid[0: self.input_size[0], 0: self.input_size[1]]
                self.lon_grid = self.lon_grid[0: self.input_size[0], 0: self.input_size[1]]
            self.meta_data_dict = {}
            self.get_META_data(self.lat_range, self.lon_range, self.lat_grid, self.lon_grid)

            logger.info(f"self.lat_range: {self.lat_range[0]} -- {self.lat_range[-1]} len: {len(self.lat_range)}")
            logger.info(f"self.lon_range: {self.lon_range[0]} -- {self.lon_range[-1]} len: {len(self.lon_range)}")

        # get the time_embedding info. get the in_dim_list
        self.get_time_embedding_group_info()
        self.in_dim_list = self.get_in_dim_list()
        logger.info(f'in_dim_list: {self.in_dim_list}')
        logger.info(f'groups_dict: {self.groups_dict}')
        logger.info(f'decoder_input_feature: {self.decoder_input_feature}')

    def __iter__(self):

        index_list = list(np.arange(self.start, self.end))
        if self.shuffle:
            random.shuffle(index_list)
        for i in index_list:
            info = self.index_info.iloc[i, :]
            year = str(info['year']).rjust(4, '0')
            month = str(info['month']).rjust(2, '0')
            day = str(info['day']).rjust(2, '0')
            cycle = str(info['cycle']).rjust(2, '0')
            lead_time = str(info['lead_time']).rjust(2, '0')
            index_time = f"{year}{month}{day}_T{cycle}_f{lead_time}"
            valid_time = info['valid_time']

            # logger.info(f'Index: {i}. NWP_time: {index_time} valid_time: {valid_time}')
            # step1: get the meta data.
            if self.slide:
                lat_range, lon_range, lat_grid, lon_grid = self.generate_spatial_slide()
                self.meta_data_dict = {}
                self.get_META_data(lat_range, lon_range, lat_grid, lon_grid)
            else:
                lat_range, lon_range = self.lat_range, self.lon_range
                lat_grid, lon_grid = self.lat_grid, self.lon_grid

            # step2: get the NWP data
            try:
                NWP_data_dict = self.get_NWP_data(info, lat_range=lat_range, lon_range=lon_range)
            except Exception as e:
                logger.debug(f'Index: {i}. {index_time}. Failed to get the NWP_data due to {e}')
                continue

            # step3: get the OBS feature before the issue_time
            if len(self.valid_obs_source) > 0:
                try:
                    OBS_data_dict = self.get_OBS_data(info, lat_range=lat_range, lon_range=lon_range)
                except Exception as e:
                    logger.debug(f'Index: {i}. {index_time}. Failed to get the OBS_data due to {e}')
                    # logger.info(f'Index: {i}. {index_time}. Failed to get the OBS_data due to {e}')
                    continue

            # step4: get the TimeEmbedding feature
            Time_data_dict = self.get_time_embedding_data(info)

            # step5: get the target data
            target_file = os.path.join(self.target_path, info['target_file'])
            try:
                target, mask = self.get_target_data(target_file, lat_range=lat_range, lon_range=lon_range)
                target = np.expand_dims(target, axis=0) if target.shape[0] != 1 else target
                mask = np.expand_dims(mask, axis=0) if mask.shape[0] != 1 else mask
            except Exception as e:
                logger.debug(f"Index: {i}. {valid_time} Failed to get the target data due to {e}")
                continue

            data_dict = self.meta_data_dict.copy()
            if len(NWP_data_dict) > 0:
                data_dict.update(NWP_data_dict)
            if len(self.valid_obs_source) > 0:
                data_dict.update(OBS_data_dict)
            data_dict.update(Time_data_dict)
            all_group_data, decoder_input = self.get_group_data(data_dict,
                                                                groups_dict=self.groups_dict,
                                                                decoder_input_feature=self.decoder_input_feature
                                                                )

            # concatenate all group data into channel dimension.
            if self.if_channel:
                all_group_data = np.concatenate(all_group_data, axis=0)

            if self.input_size != self.target_size:
                target = center_crop(target, target_size=self.target_size)
                mask = center_crop(mask, target_size=self.target_size)

            if len(decoder_input) == 0:
                yield i, all_group_data, [target, mask]
            else:
                decoder_input = np.asarray(decoder_input).mean(axis=0, keepdims=True)
                yield i, all_group_data, [target, mask], decoder_input

    def get_group_data(self, data_dict, groups_dict=None, decoder_input_feature=None):
        '''
        divide all the variables-data into different groups.
        '''
        if groups_dict is None:
            groups_dict = self.groups_dict
        if decoder_input_feature is None:
            decoder_input_feature = self.decoder_input_feature

        groups_list = list(groups_dict.keys())
        groups_list.sort()
        all_group_data = []
        decoder_input = []
        for group in groups_list:
            var_list = groups_dict[group]
            group_data = []
            for var in var_list:
                data = data_dict[var]
                group_data.append(data)
                if var in decoder_input_feature:
                    decoder_input.append(data)
            # shape: [in_dim, H, W]
            group_data = np.expand_dims(group_data[0], axis=0) if len(group_data) == 1 else np.asarray(group_data)
            # logger.info(f"group: {group} var_list: {var_list}. group_data.shape: {group_data.shape}")
            all_group_data.append(group_data)

        return all_group_data, decoder_input

    def get_target_data(self, target_file, lat_range=None, lon_range=None):
        '''
        func: get the specific target data and mask.
        '''
        if lat_range is None or lon_range is None:
            lat_range, lon_range = self.lat_range, self.lon_range
        ds = xr.open_dataset(target_file)
        ds = ds.sel(lon=lon_range, lat=lat_range, method='nearest')
        target = ds[self.target_params['variables']['value']['raw_name']].values
        mask_name = self.target_params['variables']['mask']['raw_name']
        mask = np.ones_like(target) if mask_name not in ds else ds[mask_name].values

        var_info = self.target_params['variables']['value']
        # replace np.nan with 0.
        mask[np.isnan(target)] = 0
        target[np.isnan(target)] = 0
        target = self.normalize_data(target, var_info)

        return target, mask

    def get_NWP_data(self, info, lat_range=None, lon_range=None):
        '''
        func: get all the NWP data according to the data yml file.
        '''
        if lat_range is None or lon_range is None:
            lat_range, lon_range = self.lat_range, self.lon_range
        NWP_data_dict = {}
        for NWP, mode in zip(self.valid_NWP, self.valid_mode):
            lead_time = info['lead_time']
            multi_file = info[f'{NWP}_{mode}_multi_file']
            lead_time_list = np.array(info[f'{NWP}_{mode}_multi_lead_time']) - lead_time
            ds_list = []
            for i in range(len(multi_file)):
                file = multi_file[i]
                ds = xr.open_dataset(os.path.join(self.params[NWP][mode]['path'], file))
                ds = ds.sel(lat=lat_range, lon=lon_range, method='nearest')
                ds_list.append(ds)

            variable_info = self.params[NWP][mode]['variables']
            for var in variable_info:
                var_info = variable_info[var]
                if var_info['use'] == 1:
                    for j, lead_time in enumerate(lead_time_list):
                        if mode == 'pressure':
                            level_list = var_info.get('level', None)
                            if level_list is not None:
                                for level in level_list:
                                    data = ds_list[j][var_info['raw_name']].sel(level=level).values
                                    data = self.normalize_data(data, var_info)
                                    NWP_mode_leadtime_var = f'{NWP}_{mode}_level{level}_leadtime{lead_time}_{var}'
                                    NWP_data_dict[NWP_mode_leadtime_var] = data
                        else:
                            data = ds_list[j][var_info['raw_name']].values
                            data = self.normalize_data(data, var_info)
                            NWP_mode_leadtime_var = f'{NWP}_{mode}_leadtime{lead_time}_{var}'
                            NWP_data_dict[NWP_mode_leadtime_var] = data

        return NWP_data_dict

    def get_OBS_data(self, info, lat_range=None, lon_range=None):
        '''
        func: get all the OBS data according to the data yml file.
        '''
        if lat_range is None or lon_range is None:
            lat_range, lon_range = self.lat_range, self.lon_range
        OBS_data_dict = {}
        for obs_source in self.valid_obs_source:
            past_input_time = self.obs_params[obs_source]['time']['past_input_time']  # eg:['0h', '-1h']
            past_input_file = info[f'{obs_source}_multi_file']

            ds_list = []
            for i in range(len(past_input_file)):
                file = os.path.join(self.obs_params[obs_source]['path'], past_input_file[i])
                ds = xr.open_dataset(file).sel(lat=lat_range, lon=lon_range, method='nearest')
                ds_list.append(ds)

            variable_info = self.obs_params[obs_source]['variables']
            for var in variable_info:
                var_info = variable_info[var]
                if var_info['use'] == 1:
                    for j, past_time in enumerate(past_input_time):
                        data = ds_list[j][var_info['raw_name']].values
                        data = np.squeeze(data)
                        data[np.isnan(data)] = 0
                        data = self.normalize_data(data, var_info)
                        obs_time_var = f'{obs_source}_past{past_time}_{var}'
                        OBS_data_dict[obs_time_var] = data

        return OBS_data_dict

    def get_time_embedding_data(self, info):
        '''
        func: embedding time info. including [lead_time, cycle, day, month]
        '''
        lead_time = int(info['lead_time'])
        cycle = int(info['cycle'])
        day = int(info['target_day'])
        month = int(info['target_month'])

        Time_data_dict = {}
        variables_info = self.Time_param['variables']
        for var in variables_info:
            var_info = variables_info[var]
            use = var_info.get('use', 0)
            if use == 1:
                # group = var_info['group']
                data = np.ones(shape=(self.input_size[0], self.input_size[1]))
                if var == 'leadtime_sin':
                    data = data * np.sin(lead_time / 24.0 * (np.pi / 2))
                elif var == 'leadtime_cos':
                    data = data * np.cos(lead_time / 24.0 * (np.pi / 2))
                elif var == 'cycle_sin':
                    data = data * np.sin(cycle / 12.0 * (np.pi / 2))
                elif var == 'cycle_cos':
                    data = data * np.cos(cycle / 12.0 * (np.pi / 2))
                elif var == 'day_sin':
                    data = data * np.sin(day / 31.0 * (np.pi / 2))
                elif var == 'day_cos':
                    data = data * np.cos(day / 31.0 * (np.pi / 2))
                elif var == 'month_sin':
                    data = data * np.sin(month / 12 * (np.pi / 2))
                elif var == 'month_cos':
                    data = data * np.cos(month / 12 * (np.pi / 2))
                else:
                    raise Exception(f" Not such variable: {var}")
                Time_data_dict[var] = data

        return Time_data_dict

    def get_META_data(self, lat_range=None, lon_range=None, lat_grid=None, lon_grid=None):
        '''
        func: get the META data. including lon/lat/height or other topography features.
        '''
        file = self.META_param['path']
        variables_info = self.META_param['variables']
        ds = xr.open_dataset(file)
        ds = ds.sel(lon=lon_range, lat=lat_range, method='nearest')
        for var in variables_info.keys():
            var_info = variables_info[var]
            if var_info['use'] == 1:
                if var == 'lon':
                    data = lon_grid
                elif var == 'lat':
                    data = lat_grid
                else:
                    data = ds[var].values
                data = self.normalize_data(data, var_info)
                self.meta_data_dict[var] = data

    def get_META_info(self, loc_range=None, res=None):
        '''
        func: get the group info of META-features.
        '''
        if loc_range is None:
            loc_range = self.META_param['loc_range']
        if res is None:
            res = self.META_param['res']
        lat_min, lat_max, lon_min, lon_max = loc_range
        lat_range = np.arange(lat_min, lat_max + res, res).astype(np.float32)
        lon_range = np.arange(lon_min, lon_max + res, res).astype(np.float32)
        lat_range = lat_range[lat_range <= lat_max]
        lon_range = lon_range[lon_range <= lon_max]
        lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)

        variables_info = self.META_param['variables']
        for var in variables_info.keys():
            var_info = variables_info[var]
            decoder_input = var_info.get('decoder_input', False)
            if var_info['use'] == 1:
                self.groups_dict[var_info['group']].append(var)
                if decoder_input:
                    self.decoder_input_feature.append(var)
        return lat_range, lon_range, lon_grid, lat_grid

    def get_time_embedding_group_info(self):
        '''
        func: get the group info of time-features
        '''
        variables_info = self.Time_param['variables']
        for var in variables_info:
            var_info = variables_info[var]
            decoder_input = var_info.get('decoder_input', False)
            use = var_info.get('use', 0)
            if use == 1:
                group = var_info['group']
                if var not in self.groups_dict[group]:
                    self.groups_dict[group].append(var)
                    if decoder_input:
                        self.decoder_input_feature.append(var)

    def get_OBS_index_info(self):

        valid_obs_source = []
        if self.obs_params is None:
            return valid_obs_source

        obs_list = list(self.obs_params.keys())
        if len(obs_list) > 0:
            for obs_source in obs_list:
                obs_source_params = self.obs_params[obs_source]
                flag = self.get_OBS_group_and_index_info(obs_source_params, obs_source)
                if flag:
                    valid_obs_source.append(obs_source)

        return valid_obs_source

    def get_OBS_group_and_index_info(self, model_param, obs_source='CMPAS'):
        """
        Args:
            obs_source_params (_type_): _description_
            obs_source (str, optional): _description_. Defaults to 'CMPAS'.
        """
        def get_past_input_time_list(x, past_input_time):
            issue_time = pd.to_datetime(x['issue_time'])
            past_input_valid_time = [issue_time + pd.Timedelta(pt) for pt in past_input_time]
            return past_input_valid_time

        def get_past_input_file_list(x, obs_pd_info, directory):
            past_input_time = x['obs_multi_time']
            past_input_time = [t.strftime('%Y%m%dT%H%M%S') for t in past_input_time]
            past_input_file = []
            for valid_time in past_input_time:
                try:
                    info = obs_pd_info.loc[valid_time]
                    file = info['file']
                    # logger.info(f'-------valid_time: {valid_time}. file: {file}')
                except KeyError:
                    file = None
                if file is None or not os.path.exists(os.path.join(directory, file)):
                    past_input_file = np.nan
                    break
                else:
                    past_input_file.append(file)
            return past_input_file

        index_file = model_param.get('index_file', None)
        path = model_param.get('path', None)

        if index_file is None or path is None:
            logger.debug(f'{obs_source} no index_file or path')
            return False

        logger.info(f"obs_source: {obs_source}")
        pd_info = pd.read_csv(index_file)
        pd_info = self.drop_unnamed(pd_info)
        pd_info = pd_info.set_index('valid_time')

        logger.info(f'{obs_source}.shape: {pd_info.shape} {obs_source}.columns: {pd_info.columns}')
        logger.info(f"Aline {obs_source}'s valid files into self.index_info according to the issue_time")

        # When forecasting as specific issue_time, we can only get the OBS data before the cycle.
        past_input_time = model_param['time']['past_input_time']  # eg: ['0h', '-3h']
        self.index_info['obs_multi_time'] = self.index_info.apply(get_past_input_time_list,
                                                                  past_input_time=past_input_time,
                                                                  axis=1
                                                                  )
        self.index_info['obs_multi_file'] = self.index_info.apply(get_past_input_file_list,
                                                                  obs_pd_info=pd_info,
                                                                  directory=path,
                                                                  axis=1
                                                                  )
        self.index_info = self.index_info.dropna()
        self.index_info = self.index_info.rename(columns={'obs_multi_time': f'{obs_source}_multi_time',
                                                          'obs_multi_file': f'{obs_source}_multi_file'})
        logger.info(f'After aline. {obs_source}: shape: {self.index_info.shape} columns: {self.index_info.columns}')
        variable_info = model_param['variables']
        logger.info(f'variable_info: {variable_info}')
        if variable_info is not None:
            for var in variable_info:
                var_info = variable_info[var]
                use = var_info.get('use', 0)
                group = var_info['group']
                if use == 1:
                    for past_time in past_input_time:
                        obs_time_var = f'{obs_source}_past{past_time}_{var}'
                        self.groups_dict[group].append(obs_time_var)
        return True

    def get_NWP_index_info(self):
        '''
        func: merge add valid NWP-mode index_file.
        '''
        # step1: get all the index_file of all the NWP and mode.
        valid_NWP = []
        valid_mode = []
        valid_file = []
        NWP_list = ['EC', 'SMS', 'GRAPES']
        mode_list = ['rain', 'surface', 'pressure']
        for NWP in NWP_list:
            NWP_param = self.params.get(NWP, None)
            if NWP_param is not None:
                for mode in mode_list:
                    mode_param = NWP_param.get(mode, None)
                    if mode_param is not None:
                        mode_file = self.get_NWP_mode_group_and_time_info(mode_param, NWP=NWP, mode=mode)
                        if mode_file is not None:
                            valid_NWP.append(NWP)
                            valid_mode.append(mode)
                            valid_file.append(mode_file)

        # step2: merge all the index_file by date.
        if len(valid_file) == 0:
            return None

        if len(valid_file) == 1:
            nwp_file = valid_file[0]
        else:
            nums = len(valid_file)
            nwp_file = valid_file[0]
            for i in range(1, nums):
                p2 = valid_file[i]
                nwp_file = pd.merge(nwp_file, p2, on=['year', 'month', 'day', 'cycle', 'lead_time', 'valid_time'])

        return nwp_file, valid_NWP, valid_mode

    def judge_NWP_file_exists(self, lead_time_files, path):

        flag = True
        for file in lead_time_files:
            if not os.path.exists(os.path.join(path, file)):
                flag = False
                break
        return flag

    def get_NWP_mode_group_and_time_info(self, mode_param, NWP='EC', mode='surface'):
        """
        func: get the files belong to the past_lead_time & future_lead_time.
              1. add {NWP}_{mode}_multi_lead_time & {NWP}_{mode}_multi_file into model_param[index_file]
              2. add leadtime info to variable and divide them into group.
        Parameter
        ---------
        NWP: str
            choices: ['EC', 'SMS', 'GRAPES']
        mode: str
            choices: ['rain', 'surface', 'pressure']
        """
        index_file = mode_param.get('index_file', None)
        path = mode_param.get('path', None)
        if index_file is None or path is None:
            logger.debug(f'{NWP}-{mode} no index_file or path')
            return None

        logger.info(f"NWP: {NWP} mode: {mode}")
        pd_info = pd.read_csv(index_file)
        pd_info = self.drop_unnamed(pd_info)

        # check the time resolution and lead_time gap.
        path = mode_param['path']
        time_res = mode_param['time']['time_res']
        past_lead_time = mode_param['time']['past_lead_time']
        future_lead_time = mode_param['time']['future_lead_time']

        # eg: past_gap = [1] or [0]  future_gap = [1] or [0]
        if past_lead_time >= time_res:
            past_gap = np.arange(1, past_lead_time // time_res + 1) * time_res
        else:
            past_gap = np.arange(0, 1)

        if future_lead_time >= time_res:
            future_gap = np.arange(1, future_lead_time // time_res + 1) * time_res
        else:
            future_gap = np.arange(0, 1)

        min_lead_time = np.min(self.lead_time_list) + np.max(past_gap)
        max_lead_time = np.max(self.lead_time_list) - np.max(future_gap)

        # ensure the lead_time in past_gap ~ future_gap are all in self.lead_time_list
        pd_info = pd_info[pd_info['lead_time'] >= min_lead_time]
        pd_info = pd_info[pd_info['lead_time'] <= max_lead_time]

        # drop invalid cycle samples
        pd_info = pd_info[pd_info['lead_time'].isin(self.lead_time_list)]
        pd_info = pd_info[pd_info['cycle'].isin(self.cycle_list)]

        # get the lead_time_list. eg: [-3, -2, -1, 0, 1, 2, 3]. or [-1, 0, 1] or [0]
        lead_time = 0
        lead_time_list = list(set(list(lead_time - past_gap) + [lead_time] + list(lead_time + future_gap)))
        lead_time_list.sort()
        lead_time_list = np.array(lead_time_list)

        # get the lead_time_list file per lead_time.
        nums = len(pd_info)
        all_multil_lead_time = []
        all_multil_file = []
        all_valid_index = []
        for i in range(nums):
            info = pd_info.iloc[i, :]
            # eg: 'SMS_surface_0.05_20210704_T12_f44.nc'
            src_file = info['file']
            lead_time = info['lead_time']

            new_lead_time_list = list(lead_time + lead_time_list)
            new_lead_time_list.sort()

            lead_time_files = []
            src_f_lead_time = str(int(lead_time)).rjust(2, '0') + '.nc'
            for step in new_lead_time_list:
                dst_f_lead_time = str(int(step)).rjust(2, '0') + '.nc'
                dst_file = src_file.replace(src_f_lead_time, dst_f_lead_time)
                lead_time_files.append(dst_file)

            exist_flag = self.judge_NWP_file_exists(lead_time_files, path)
            if exist_flag:
                all_multil_lead_time.append(new_lead_time_list)
                all_multil_file.append(lead_time_files)
                all_valid_index.append(i)

        # add {NWP}_{mode}_multi_lead_time" & '{NWP}_{mode}_multi_file' into index_file
        pd_info = pd_info.iloc[all_valid_index, :]
        pd_info.index = range(len(pd_info))
        pd_info[f'{NWP}_{mode}_multi_lead_time'] = all_multil_lead_time
        pd_info[f'{NWP}_{mode}_multi_file'] = all_multil_file
        pd_info = pd_info.rename(columns={'file': f'{NWP}_{mode}_file'})

        logger.info(f'pd_info.shape: {pd_info.shape}  pd_info.columns: {pd_info.columns}')
        logger.info(f'mode_param: {mode_param}')

        # add leadtime to the variable and divide them by group.
        variable_info = mode_param['variables']
        logger.info(f'variable_info: {variable_info}')
        if variable_info is not None:
            for var in variable_info:
                var_info = variable_info[var]
                decoder_input = var_info.get('decoder_input', False)
                use = var_info.get('use', 0)
                group = var_info['group']
                if use == 1:
                    for lead_time in lead_time_list:
                        if mode == 'pressure':
                            level_list = var_info.get('level', None)
                            if level_list is not None:
                                for level in level_list:
                                    NWP_mode_leadtime_var = f'{NWP}_{mode}_level{level}_leadtime{lead_time}_{var}'
                                    self.groups_dict[group].append(NWP_mode_leadtime_var)
                                    if decoder_input and lead_time == 0:
                                        self.decoder_input_feature.append(NWP_mode_leadtime_var)
                        else:
                            NWP_mode_leadtime_var = f'{NWP}_{mode}_leadtime{lead_time}_{var}'
                            self.groups_dict[group].append(NWP_mode_leadtime_var)
                            if decoder_input and lead_time == 0:
                                self.decoder_input_feature.append(NWP_mode_leadtime_var)

        return pd_info

    def merge_by_date(self, p1, p2,
                      p1_NWP='EC', p2_NWP='SMS',
                      p1_mode='rain', p2_mode='surface',
                      merge_columns=['year', 'month', 'day', 'cycle', 'lead_time']):
        '''
        func: merge p1 & p2 by merge_columns.
        '''
        left_on = [f'{p1_NWP}_{p1_mode}_{col}' for col in merge_columns]
        right_on = [f'{p2_NWP}_{p2_mode}_{col}' for col in merge_columns]
        p1 = p1.drop(columns=['valid_time']) if 'valid_time' in p1 else p1
        p3 = pd.merge(p1, p2, left_on=left_on, right_on=right_on)
        return p3

    def generate_spatial_slide(self):

        input_size = self.input_size
        lat_rest = len(self.lat_range) - input_size[0]
        lon_rest = len(self.lon_range) - input_size[1]

        rand_lat, rand_lon = np.random.randint(lat_rest), np.random.randint(lon_rest)
        lat_index = np.arange(0 + rand_lat, input_size[0] + rand_lat)
        lon_index = np.arange(0 + rand_lon, input_size[1] + rand_lon)

        lat_range = self.lat_range[lat_index]
        lon_range = self.lon_range[lon_index]
        lat_grid = self.lat_grid[lat_index, :][:, lon_index]
        lon_grid = self.lon_grid[lat_index, :][:, lon_index]

        return lat_range, lon_range, lat_grid, lon_grid

    def get_in_dim_list(self):
        '''
        func: get the in_dim_list of all features.
        '''
        groups_dict = self.groups_dict
        groups_list = list(groups_dict.keys())
        groups_list.sort()
        in_dim_list = [len(groups_dict[group]) for group in groups_list]
        if self.if_channel:
            in_dim_list = np.sum(in_dim_list)
        return in_dim_list

    def normalize_data(self, data, var_info):
        '''
        func: change the distribution of data according to the var-info.
        '''
        operation = var_info.get('operation', None)
        mu = var_info.get('mu', None)
        scale = var_info.get('scale', None)
        mean = var_info.get('mean', 0.5)
        std = var_info.get('std', 0.25)

        if operation is not None:
            data = eval('data' + operation)
        if mu is not None and scale is not None:
            data = self.standard_scaler(data, mu=mu, scale=scale)
        else:
            data = self.norm(data, mean=mean, std=std)
        return data

    def norm(self, data, mean=0.5, std=0.25):
        '''
        func: normalize the data. norm one given feature.
        '''
        ori_mean = np.mean(data)
        ori_std = np.std(data)
        y = data - ori_mean
        y = y / ori_std if ori_std != 0 else y
        y = y * std + mean
        return y

    def standard_scaler(self, value, mu, scale):
        return (value - mu) / scale

    def drop_unnamed(self, index_info):
        for i in range(3):
            name = f'Unnamed: {i}'
            if name in index_info:
                index_info = index_info.drop(columns=[name])
        return index_info


def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    # the dataset copy in this worker process
    dataset = worker_info.dataset
    worker_id = worker_info.id
    worker_cnt = worker_info.num_workers
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_cnt)))
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)
