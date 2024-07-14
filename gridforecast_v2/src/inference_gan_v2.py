import hydra
import os
import torch
import logging
import xarray as xr
from gridforecast_v2.src import factory
from gridforecast_v2.src.utils import configure_logging
from gridforecast_v2.utils.torch_utils import try_gpu, try_all_gpus
from gridforecast_v2.utils.data_preprocess import de_standard_scaler, ds_to_float32, zip_ds
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


configure_logging()
logger = logging.getLogger(__name__)
device = try_gpu(0)


def save_index_info(data, save_path, flag='test'):

    if 'index_info' in data.__dir__():
        index_info = data.index_info
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f'{flag}_index_info.csv')
        index_info.to_csv(save_file, index=None)

    return index_info


def save_obs_pre_dataloaer_v1(indexs, obss, predicts, test_index, test_data, dataloader_param, save_path):

    assert dataloader_param['name'] == 'dataloader_v1'

    # get mu and scale of target.
    target_feature = dataloader_param['test']['data_feature']['Target']['variables']['value']
    mu = target_feature.get('mu', None)
    scale = target_feature.get('scale', None)
    raw_name = target_feature['raw_name']  # eg: r3

    # convert obs and pre into the proper scale.
    logger.info(f'Reconvert data to proper scale. mu: {mu}. scale: {scale}')
    if mu is not None and scale is not None:
        obss = de_standard_scaler(obss, mu=mu, scale=scale)
        predicts = de_standard_scaler(predicts, mu=mu, scale=scale)
    logger.info(f'obs.shape: {obss.shape}')
    logger.info(f'pred.shape: {predicts.shape}')

    # get the lat and lon
    lat_range, lon_range = test_data.lat_range, test_data.lon_range
    lat_grid, lon_grid = test_data.lat_grid, test_data.lon_grid

    assert obss.shape == predicts.shape
    batch, channel, width, height = predicts.shape
    assert channel == 1

    for i, index in enumerate(indexs):
        info = test_index.iloc[index, :]
        year = str(info['year']).rjust(4, '0')
        month = str(info['month']).rjust(2, '0')
        day = str(info['day']).rjust(2, '0')
        cycle = str(info['cycle']).rjust(2, '0')
        lead_time = str(info['lead_time']).rjust(2, '0')
        valid_time = info['valid_time']

        pre = predicts[i, 0, :, :]
        obs = obss[i, 0, :, :]

        ds = xr.Dataset(coords={
            'lon': (('lon'), lon_range),
            'lat': (('lat'), lat_range),
            'lon_grid': (('lat', 'lon'), lon_grid),
            'lat_grid': (('lat', 'lon'), lat_grid),
        })

        ds.coords['date'] = f'{year}{month}{day}'
        ds.coords['cycle'] = int(cycle)
        ds.coords['lead_time'] = int(lead_time)
        ds.coords['valid_time'] = valid_time

        ds[f'pre_{raw_name}'] = (('lat', 'lon'), pre)
        ds[f'obs_{raw_name}'] = (('lat', 'lon'), obs)

        filename = f"{year}/{month}/{day}/{cycle}/{year}{month}{day}_T{cycle}_f{lead_time}.nc"
        save_file = os.path.join(save_path, filename)
        os.makedirs(os.path.dirname(save_file), exist_ok=True)

        ds = ds_to_float32(ds)
        ds = zip_ds(ds)
        ds.to_netcdf(save_file)
        logger.info(f'{i} {save_file} down!')

    return None


@hydra.main(config_path="../config", config_name='inference_gan_v2')
def run(cfg):

    # Convert everything to primitive containers
    # link: https://hydra.cc/docs/next/advanced/instantiate_objects/overview/#parameter-conversion-strategies
    # Convert parameter from DictConfig. ListConfig to dict and list.
    cfg = instantiate(cfg, _convert_="all")

    logger.info('Start inference!')
    input_dir = cfg['input_dir']
    output_dir = cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"device: {device}")
    logger.info(f"input_dir: {input_dir}")
    logger.info(f"output_dir: {output_dir}")

    inference_wrapper_param = cfg['inference_wrapper']
    dataloader_param = cfg['dataloader']
    model_param = cfg['model']
    dis_model_param = cfg['discriminator_model']
    loss_param = cfg['loss']
    dis_loss_param = cfg['discriminator_loss']

    logger.info('create dataloader.')
    logger.info('----------------dataload parameter--------------------')
    logger.info(f'{dataloader_param}')

    test_data, test_dataloader = factory.create_dataloader(dataloader_param, flag='test')
    test_index = save_index_info(test_data, save_path=output_dir, flag='test')

    # slide = dataloader_param['test']['data_feature']['META']['slide']
    # print(f'slide: {slide}')
    # if slide:
    #     print('-----------------------------')

    if 'in_dim_list' in test_data.__dir__():
        in_dim_list = test_data.in_dim_list
        in_dim_list = in_dim_list if isinstance(in_dim_list, list) else [int(in_dim_list)]
        if 'in_dim_list' in model_param['config']:
            model_param['config']['in_dim_list'] = in_dim_list
            assert isinstance(in_dim_list, list)
            cfg['model']['config']['in_dim_list'] = in_dim_list

    # create model | optimizer | loss | scheduler
    logger.info('create model | loss | inference_wrapper')

    logger.info('----------------model parameter--------------------')
    logger.info(f'{model_param}')
    logger.info('----------------inference_wrapper parameter--------------------')
    logger.info(f'{inference_wrapper_param}')

    gen_model = factory.create_model(model_param)
    dis_model = factory.create_model(dis_model_param)
    gen_loss = factory.create_loss(loss_param)
    dis_loss = factory.create_loss(dis_loss_param)

    instances_config = {
        'device': device,
        'save_path': output_dir,
        'gen_model': gen_model,
        'dis_model': dis_model,
        'gen_loss': gen_loss,
        'dis_loss': dis_loss,
    }

    inference_wrapper = factory.create_inference_wrapper(inference_wrapper_param,
                                                         instances_config
                                                         )

    gen_model_parameters = sum(torch.numel(parameter) for parameter in gen_model.parameters())
    dis_model_parameters = sum(torch.numel(parameter) for parameter in dis_model.parameters())
    logger.info(f'Parameter of generator model: {gen_model_parameters}')
    logger.info(f'Parameter of discriminator model: {dis_model_parameters}')

    # index_id.shape (N,) preds.shape = obss.shape (N, c, h, w)
    index_id, obss, preds = inference_wrapper.inference(test_dataloader)
    logger.info(f"Name of dataloader: {dataloader_param['name']}")
    if dataloader_param['name'] == 'dataloader_v1':
        logger.info('Start saving pre and obs.')
        save_obs_pre_dataloaer_v1(indexs=index_id,
                                  obss=obss,
                                  predicts=preds,
                                  test_index=test_index,
                                  test_data=test_data,
                                  dataloader_param=dataloader_param,
                                  save_path=os.path.join(output_dir, 'inference_nc'))

    logger.info('Inference finished!')
    logger.info('----------------The final parameter-------------------')
    cfg = OmegaConf.to_yaml(cfg)
    # logger.info(f'{cfg}')


if __name__ == "__main__":
    run()
