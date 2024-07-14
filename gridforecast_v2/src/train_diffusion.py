import numpy as np
import hydra
import os
import torch
import logging
from hydra.utils import instantiate
from omegaconf import OmegaConf
from gridforecast_v2.src import factory
from gridforecast_v2.src.utils import configure_logging, write_yml_file
from gridforecast_v2.utils.torch_utils import try_gpu, try_all_gpus
from gridforecast_v2.evaluation import metric_obs_pre
from gridforecast_v2.utils.data_preprocess import de_standard_scaler
import copy


configure_logging()
logger = logging.getLogger(__name__)
device = try_gpu(0)


metric_setting_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                                   'evaluation', 'eval_setting_v1.yaml')
assert os.path.exists(metric_setting_file)


def save_index_info(data, save_path, flag='train'):

    if 'index_info' in data.__dir__():
        index_info = data.index_info
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f'{flag}_index_info.csv')
        index_info.to_csv(save_file, index=None)

    return None


def rescale_dataloader_v1(data, dataloader_param):

    assert dataloader_param['name'] == 'dataloader_v1'
    # get mu and scale of target.
    target_feature = dataloader_param['train']['data_feature']['Target']['variables']['value']
    mu = target_feature.get('mu', None)
    scale = target_feature.get('scale', None)
    # convert obs and pre into the proper scale.
    logger.info(f'Reconvert data to proper scale. mu: {mu}. scale: {scale}')
    if mu is not None and scale is not None:
        data = de_standard_scaler(data, mu=mu, scale=scale)
    return data


def replace_path_index(src_feature, dst_feature,
                       key_list=['path', 'index_file'],
                       nwp_list=['EC', 'SMS', 'GRAPES'],
                       nwp_mode_list=['rain', 'surface', 'pressure'],
                       obs_source_list=None
                       ):
    # deepcopy
    new_feature = copy.deepcopy(src_feature)
    for key in key_list:
        new_feature['Target'][key] = dst_feature['Target'][key]

    for nwp in nwp_list:
        nwp_exist = dst_feature.get(nwp, None)
        if nwp_exist is not None:
            for mode in nwp_mode_list:
                mode_exist = dst_feature[nwp].get(mode, None)
                if mode_exist is not None:
                    for key in key_list:
                        key_exist = dst_feature[nwp][mode].get(key, None)
                        if key_exist is not None:
                            new_feature[nwp][mode][key] = dst_feature[nwp][mode][key]

    obs_exist = dst_feature.get('OBS', None)
    if obs_exist is not None:
        if obs_source_list is not None:
            for obs_source in obs_source_list:
                obs_source_exist = dst_feature['OBS'].get(obs_source, None)
                if obs_source_exist is not None:
                    for key in key_list:
                        key_exist = dst_feature['OBS'][obs_source].get(key, None)
                        if key_exist is not None:
                            new_feature['OBS'][obs_source][key] = dst_feature['OBS'][obs_source][key]

    return new_feature


def consistency_train_valid_test_feature_dataloader_v1(dataloader_param):
    """
    Function: 保证train | valid | test 数据集中除了index不一致外，特征保持一致.
    """
    key_list = ['path', 'index_file']
    nwp_list = ['EC', 'SMS', 'GRAPES']
    nwp_mode_list = ['rain', 'surface', 'pressure']

    train_feature = dataloader_param['train']['data_feature']
    valid_feature = dataloader_param['valid']['data_feature']
    test_feature = dataloader_param['test']['data_feature']

    obs_source_list = None
    obs_params = train_feature.get('OBS', None)
    if obs_params is not None:
        obs_source_list = list(obs_params.keys())  # eg: ['CMPAS']

    test_train_feature = replace_path_index(train_feature, test_feature,
                                            key_list=key_list,
                                            nwp_list=nwp_list,
                                            nwp_mode_list=nwp_mode_list,
                                            obs_source_list=obs_source_list
                                            )

    valid_train_feature = replace_path_index(train_feature, valid_feature,
                                             key_list=key_list,
                                             nwp_list=nwp_list,
                                             nwp_mode_list=nwp_mode_list,
                                             obs_source_list=obs_source_list
                                             )

    dataloader_param['test']['data_feature'] = test_train_feature
    dataloader_param['valid']['data_feature'] = valid_train_feature

    return dataloader_param


@hydra.main(config_path="../config", config_name='train_diffusion')
def train(cfg):

    save_path = cfg['output_dir']
    os.makedirs(save_path, exist_ok=True)

    logger.info(f"device: {device}")
    logger.info(f"save_path: {save_path}")

    # Convert everything to primitive containers
    # link: https://hydra.cc/docs/next/advanced/instantiate_objects/overview/#parameter-conversion-strategies
    # Convert parameter from DictConfig. ListConfig to dict and list.
    cfg = instantiate(cfg, _convert_="all")

    train_test_wrapper_param = cfg['train_test_wrapper']
    dataloader_param = cfg['dataloader']
    diffusion_param = cfg['diffusion']
    model_param = cfg['diffusion']['model']
    loss_param = cfg['loss']
    optimizer_param = cfg['optimizer']
    lr_scheduler_param = cfg['lr_scheduler']

    logger.info('----------------dataload parameter--------------------')
    logger.info(f'{dataloader_param}')

    dataloader_name = dataloader_param.get('name', None)
    if dataloader_name == 'dataloader_v1':
        dataloader_param = consistency_train_valid_test_feature_dataloader_v1(dataloader_param)
        cfg['dataloader'] = dataloader_param
        logger.info(f'dataloader_name: {dataloader_name}. add consistency_train_valid_test_feature operation:')
        logger.info(f'{dataloader_param}')

    logger.info('create dataloader.')
    train_data, train_dataloader = factory.create_dataloader(dataloader_param, flag='train')
    valid_data, valid_dataloader = factory.create_dataloader(dataloader_param, flag='valid')
    test_data, test_dataloader = factory.create_dataloader(dataloader_param, flag='test')

    save_index_info(train_data, save_path=save_path, flag='train')
    save_index_info(valid_data, save_path=save_path, flag='valid')
    save_index_info(test_data, save_path=save_path, flag='test')

    # if 'in_dim_list' in train_data.__dir__():
    #     assert valid_data.in_dim_list == train_data.in_dim_list
    #     in_dim_list = train_data.in_dim_list
    #     in_dim_list = in_dim_list if isinstance(in_dim_list, list) else [int(in_dim_list)]
    #     if 'in_dim_list' in model_param['config']:
    #         model_param['config']['in_dim_list'] = in_dim_list
    #         assert isinstance(in_dim_list, list)
    #         cfg['model']['config']['in_dim_list'] = in_dim_list


    # create model | optimizer | loss | scheduler
    logger.info('create model | optimizer | scheduler | loss | train_test_wrapper')

    logger.info('----------------model parameter--------------------')
    logger.info(f'{model_param}')
    logger.info('----------------diffusion parameter--------------------')
    logger.info(f'{diffusion_param}')
    logger.info('----------------optimizer parameter--------------------')
    logger.info(f'{optimizer_param}')
    logger.info('----------------lr scheduler parameter--------------------')
    logger.info(f'{lr_scheduler_param}')
    logger.info('----------------loss parameter--------------------')
    logger.info(f'{loss_param}')
    logger.info('----------------train_test_wrapper parameter--------------------')
    logger.info(f'{train_test_wrapper_param}')

    model = factory.create_model(model_param)
    diffusion_model = factory.create_diffusion(diffusion_param)

    loss = factory.create_loss(loss_param)
    optimizer = factory.create_optimizer(optimizer_param, diffusion_model)
    lr_scheduler = factory.create_lr_scheduler(lr_scheduler_param, optimizer)

    instances_config = {
        'device': device,
        'save_path': save_path,
        'model': diffusion_model,
        'loss_fn': loss,
        'optimizer': optimizer,
        'scheduler': lr_scheduler,
    }

    train_test_wrapper = factory.create_train_test_wrapper(train_test_wrapper_param,
                                                           instances_config
                                                           )

    logger.info('----------------The final parameter-------------------')
    write_yml_file(os.path.join(save_path, 'all_parameter.yaml'), cfg)
    cfg_yaml = OmegaConf.to_yaml(cfg)
    logger.info(f'{cfg_yaml}')

    model_parameter = sum(torch.numel(parameter) for parameter in model.parameters())
    logger.info(f'Parameter of model: {model_parameter}')
    logger.info('Start training!')
    train_test_wrapper.train(train_dataloader, valid_dataloader)
    logger.info('Training finished!')

    # start to metric on test_dataloader.
    # index_id.shape (N,) preds.shape = obss.shape (N, c, h, w)
    logger.info('Start inference!')
    index_id, obss, preds = train_test_wrapper.inference(test_dataloader)
    logger.info('Inference finished!')
    if dataloader_name == 'dataloader_v1':
        logger.info('Reconvert obs pre to proper scale.')
        obss = rescale_dataloader_v1(obss, dataloader_param)
        preds = rescale_dataloader_v1(preds, dataloader_param)
    logger.info(f'obs.shape: {obss.shape}. pre.shape: {preds.shape}')
    obs_pre_save_dir = os.path.join(save_path, 'npy')
    os.makedirs(obs_pre_save_dir, exist_ok=True)
    np.save(os.path.join(obs_pre_save_dir, 'obs.npy'), obss)
    np.save(os.path.join(obs_pre_save_dir, 'pre.npy'), preds)
    logger.info(f'Save obs pre to {obs_pre_save_dir}')
    logger.info('Start metric!')
    metric_obs_pre.calculate_metrics(
        obs=obss,
        pre=preds,
        metric_setting_file=metric_setting_file,
        save_path=os.path.join(save_path, 'metric_obs_pre')
    )
    logger.info('Metric finished!')


if __name__ == "__main__":
    train()
