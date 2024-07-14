import torch
from torch.utils.data import DataLoader
from gridforecast_v2.data_pipelines.dataloader_v1 import dataloader_v1
from gridforecast_v2.src import (train_test_wrapper_v1, inference_wrapper_v1,
                                 train_test_wrapper_gan_v2, inference_wrapper_gan_v2,
                                 train_test_wrapper_diffusion, inference_wrapper_diffusion)
from gridforecast_v2.model.unet_2d import unet_2d
from gridforecast_v2.model.swin_unet import swin_unet
from gridforecast_v2.model.GAN import dcgan_discriminator, wgan_discriminator
from gridforecast_v2.model.Diffusion.SR3 import unet2d, diffusion
from gridforecast_v2.loss import loss_ensemble
from gridforecast_v2.loss.loss_ensemble import BMSAELoss, ExpWeightedMSAELoss, BEXPMSAELoss


def create_diffusion(diffusion_param, model):
    name = diffusion_param.get('name', None)
    config = diffusion_param['config']
    config['denoise_fn'] = model
    if name == 'GaussianDiffusion':
        return diffusion.GaussianDiffusion(**config)
    else:
        raise Exception(f'No such model: {name}')


def create_dataloader(dataload_param, flag='train'):
    assert flag in ['train', 'valid', 'test']
    name = dataload_param.get('name', None)
    batch_size = dataload_param.get('batch_size', 8)
    num_workers = dataload_param.get('num_workers', 8)
    config = dataload_param[flag]
    if name == 'dataloader_v1':
        data = dataloader_v1.GridIterableDataset(**config)
        dataloader = DataLoader(data, batch_size=batch_size,
                                num_workers=num_workers,
                                worker_init_fn=dataloader_v1.worker_init_fn,
                                )
        return data, dataloader
    else:
        raise Exception(f'No such dataload: {name}')


def create_model(model_param):
    name = model_param.get('name', None)
    config = model_param['config']
    if name == 'unet_2d':
        return unet_2d.UnetDecode(**config)
    if name == 'swin_unet':
        return swin_unet.SwinTransformerSys(**config)
    if name == 'dcgan_discriminator':
        return dcgan_discriminator.Discriminator(**config)
    if name == 'wgan_discriminator':
        return wgan_discriminator.Discriminator(**config)
    if name == 'unet2d_diffusion':
        return unet2d.UNet(**config)
    else:
        raise Exception(f'No such model: {name}')


def create_loss(loss_param):
    name = loss_param.get('name', None)
    config = loss_param['config']
    if name == 'SmoothL1Loss':
        return torch.nn.SmoothL1Loss(**config)
    if name == 'HuberLoss':
        return torch.nn.HuberLoss(**config)
    if name == 'L1Loss':
        return torch.nn.L1Loss(**config)
    if name == 'MSELoss':
        return torch.nn.MSELoss(**config)
    if name == 'BMSAELoss':
        return BMSAELoss(**config)
    if name == 'ExpWeightedMSAELoss':
        return ExpWeightedMSAELoss(**config)
    if name == 'BEXPMSAELoss':
        return BEXPMSAELoss(**config)
    if name == 'BCELoss':
        return torch.nn.BCELoss(**config)
    if name == 'HingeLoss':
        return loss_ensemble.HingeLoss
    if name == 'DGMRLoss':
        return loss_ensemble.DGMRLoss
    if name == 'WGANLoss':
        return loss_ensemble.WGANLoss
    else:
        raise Exception(f'No such loss: {name}')


def create_optimizer(optimizer_param, model):
    name = optimizer_param.get('name', None)
    config = optimizer_param['config']
    if name == 'Adam':
        return torch.optim.Adam(model.parameters(), **config)
    if name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), **config)
    if name == 'SGD':
        return torch.optim.SGD(model.parameters(), **config)
    else:
        raise Exception(f'No such optimizer: {name}')


def create_lr_scheduler(lr_scheduler_param, optimizer):
    name = lr_scheduler_param.get('name', None)
    config = lr_scheduler_param['config']
    if name == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **config)
    if name == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config)
    else:
        raise Exception(f'No such scheduler :{name}')


def create_train_test_wrapper(train_test_wrapper_param,
                              instances_config
                              ):
    name = train_test_wrapper_param.get('name', None)
    config = train_test_wrapper_param['config']
    # all_config = instances_config.update(config)
    all_config = {**config, **instances_config}
    print(f'all_config.keys(): {all_config.keys()}')
    print(f'instances_config.keys: {instances_config.keys()}')
    if name == 'train_test_wrapper_v1':
        train_test_wrapper = train_test_wrapper_v1.TrainTestModel(**all_config)
        return train_test_wrapper
    elif name == 'train_test_wrapper_v2':
        train_test_wrapper = train_test_wrapper_gan_v2.TrainTestModel(**all_config)
        return train_test_wrapper
    elif name == 'train_test_wrapper_diffusion':
        train_test_wrapper = train_test_wrapper_diffusion.TrainTestModel(**all_config)
    else:
        raise Exception(f"No such train_test_wrapper: {name}")


def create_inference_wrapper(inference_wrapper_param,
                             instances_config
                             ):
    name = inference_wrapper_param.get('name', None)
    config = inference_wrapper_param['config']
    all_config = {**config, **instances_config}
    # print(f'all_config.keys: {all_config.keys()}')
    if name == 'inference_wrapper_v1':
        inference_wrapper = inference_wrapper_v1.InferenceModel(**all_config)
        return inference_wrapper
    elif name == 'inference_wrapper_gan_v2':
        inference_wrapper = inference_wrapper_gan_v2.InferenceModel(**all_config)
        return inference_wrapper
    elif name == 'inference_wrapper_diffusion':
        inference_wrapper = inference_wrapper_diffusion.InferenceModel(**all_config)
        return inference_wrapper
    else:
        raise Exception(f"No such inference_wrapper: {name}")
