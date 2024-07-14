import yaml
import shutil
import os
import json
import logging


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


def generate_issue_time(info):

    year = str(info['year']).rjust(4, '0')
    month = str(info['month']).rjust(2, '0')
    day = str(info['day']).rjust(2, '0')
    cycle = str(info['cycle']).rjust(2, '0')
    issue_time = f"{year}-{month}-{day}T{cycle}"

    return issue_time


def center_crop(data, target_size=[64, 64]):

    # [192, 192] --> [64, 64]
    h, w = target_size
    ori_size = data.shape[-2], data.shape[-1]
    ph = int((ori_size[0] - target_size[0]) / 2)
    pw = int((ori_size[1] - target_size[1]) / 2)

    return data[..., ph: ph + h, pw: pw+w]
