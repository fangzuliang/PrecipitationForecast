import yaml
import shutil
import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap


def stack_col(data):
    '''
    Func:(8, 128, 128) --> (128, 8*128)
    '''
    b, h, w = data.shape
    new_data = np.zeros((h, b*w))
    for i in range(b):
        new_data[:, i*w: (i+1)*w] = data[i, :, :]
    return new_data


def aggregate_obs_pre(obs, pre, mu=0, scale=90):
    '''
    func:
        obs & pre: (8, 128, 128) --> (256, 8 * 128)
    '''
    obs = obs.squeeze()
    pre = pre.squeeze()

    data = np.concatenate([obs, pre], axis=1)
    new_data = stack_col(data)

    new_data = new_data * scale + mu

    return new_data


def plot_rain(data,
              levels=[0, 0.1, 2, 5, 10, 20, 30, 50, 70, 90],
              title=None,
              save_file=None
              ):

    RADAR_RAIN_COLOR_ARRAY = [
                        [1,	160, 246, 0],
                        [1, 160, 246, 255],
                        [0, 236, 236, 255],
                        [0, 216, 0,	255],
                        [1,	144, 0,	255],
                        [255, 255, 0, 255],
                        [231, 192, 0, 255],
                        [255,	144,	0,	255],
                        [255,	0,	0,	255],
                        [214,	0,	0,	255],
                        [192,	0,	0,	255],
                        [255,	0,	240, 255],
                        [150,	0,	180, 255],
                        [173,	144, 240, 255]
                        ]

    h, w = data.shape
    plt.figure(figsize=(int(w / h) * 5 + 5, 5))

    if levels is None:
        levels = [0, 0.1, 1, 2, 5, 10, 15, 20, 25, 30, 50, 90]

    color_array = [[a[i]/255.0 for i in range(4)] for a in RADAR_RAIN_COLOR_ARRAY]
    n = len(levels) - 1
    # map color with array
    cmap = ListedColormap(color_array[0:n])
    # 设置norm，进行levels和颜色的对应
    norm = matplotlib.colors.BoundaryNorm(levels, cmap.N)
    h = plt.contourf(data, levels=levels, extend='max', cmap=cmap, norm=norm)

    cb = plt.colorbar(h)
    cb.set_label('mm', fontsize=10)
    cb.ax.tick_params(labelsize=10)
    # 设置colorbar的刻度及其刻度上对应的label
    cb.set_ticks(levels)
    cb.set_ticklabels(levels)

    if title is not None:
        plt.title(title, fontsize=10)

    if save_file is not None:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        plt.savefig(save_file, bbox_inches='tight', dpi=200)

    plt.clf()
    plt.close()
    return None


def make_log(log_file):
    logging.basicConfig(
        level=logging.INFO,
        filename=log_file,
        filemode='w',
        format='%(asctime)s - %(message)s'
        )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO())
    return logger


def configure_logging(verbose=1):
    verbose_levels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
        3: logging.NOTSET
    }
    if verbose not in verbose_levels.keys():
        verbose = 1
    logger = logging.getLogger()
    logger.setLevel(verbose_levels[verbose])
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [PID=%(process)d] "
        "[%(levelname)s %(filename)s:%(lineno)d] %(message)s"))
    handler.setLevel(verbose_levels[verbose])
    logger.addHandler(handler)


def read_yml_json_file(file):
    if file.endswith('.json'):
        param_info = json.load(file)
    elif file.endswith('.yml') or file.endswith('.yaml'):
        with open(file, 'r') as f:
            param_info = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise Exception(f'the format of the {file} should be .json or .yml')
    return param_info


def write_yml_file(save_file, data):
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    fr = open(save_file, 'w')
    yaml.dump(data, fr)
    fr.close()


def copy_yaml_json_file(file, save_dir, suffix=None):

    os.makedirs(save_dir, exist_ok=True)
    if suffix is None:
        shutil.copy(file, save_dir)
    else:
        format = file.split('.')[-1]
        save_file = os.path.join(save_dir, f'{suffix}.{format}')
        shutil.copyfile(file, save_file)
    return None


def str_to_list(list_or_str):
    if type(list_or_str) is str:
        strs = list_or_str.split(',')
        lists = []
        for s in strs:
            if '-' in s:
                start, end = s.split('-')
                lists += list(range(int(start), int(end) + 1))
            else:
                lists.append(int(s))
        return lists
    elif type(list_or_str) is list:
        return list_or_str
    else:
        raise Exception("Can not get lead_time_list")


def save_params(params, save_path, param_file_name='train_param.yml'):
    os.makedirs(save_path, exist_ok=True)
    shutil.copyfile(params['Data']['data_yml_file'], os.path.join(save_path, 'data_param.yml'))
    with open(os.path.join(save_path, param_file_name), 'w') as f:
        yaml.safe_dump(params, f)


def standard_scaler(value, mean, std):
    return (value - mean) / std


def recover_standard_scaler(value, mean, std):
    return value * std + mean
