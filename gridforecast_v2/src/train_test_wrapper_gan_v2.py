import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import logging
from gridforecast_v2.optim.swa import SWA
import numpy as np
import pandas as pd
from gridforecast_v2.src.utils import make_log, plot_rain, aggregate_obs_pre
from gridforecast_v2.data_pipelines.utils import configure_logging


configure_logging()
logger = logging.getLogger(__name__)


def flip_lables(labels, p):
    """
    labels: tensor，待翻转的标签
    p: float，翻转的概率
    """
    if p == 0:
        return labels
    return torch.where(torch.rand_like(labels) < p, 1 - labels, labels)


def clip_gradients(model, clip_value):
    for p in model.parameters():
        p.grad.data.clamp_(-clip_value, clip_value)


def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).cuda()
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(d_interpolates.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class TrainTestModel(object):
    def __init__(self,
                 device,
                 save_path,
                 gen_model,
                 dis_model,
                 gen_optimizer,
                 dis_optimizer,
                 scheduler=None,
                 gen_step=2,
                 dis_step=2,
                 gen_loss=nn.MSELoss(),
                 dis_loss=nn.BCELoss(),
                 gen_loss_weight=1,
                 dis_loss_weight=0.1,
                 logger=logger,
                 gen_checkpoint_path=None,
                 dis_checkpoint_path=None,
                 num_epochs=200,
                 patience=50,
                 display_interval=25,
                 gradient_clipping=False,
                 clipping_threshold=1,
                 if_initial=True,
                 if_swa=True,
                 start_swa=5,
                 if_mask=False,
                 mask_weight_dict=None,
                 ):
        """
        func: train + test process.
        Parameter
        ------------
        device: torch.device
            CPU or CUDA.
        save_path: str
            output dir of logger info. including the loss info.
            eg: /home/zuliang/model_train/test1
        gen_model: instance of the generator model.
        dis_model: instance of the discriminator model.
        gen_optimizer: optimizer instance for generator model.
            instance of optimizer. such as torch.optim.Adam(gen_model.parameters(), lr=0.001)
        dis_optimizer: optimizer instance for discriminator model.
        scheduler: lr_scheduler instance.
            instance of learning rate scheduler.
            eg: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-8).
        gen_loss: instance of loss function used for generator model.
            default: nn.MSELoss().
        dis_loss: instance of loss function used for discriminator model.
            default: nn.BCELoss().
        dis_loss_weight: default 0.1
            sum_gen_loss = gen_loss + dis_loss_weight * dis_loss(preds, 1)
        gen_loss_weight: default 1
            sum_gen_loss = gen_loss_weight * gen_loss + dis_loss_weight * dis_loss
        logger: logging.
            Record the all the log info. if not. create a new log which saved into save_path.
        gen_checkpoint_path: The checkpoint path of the trained generator model.
        dis_checkpoint_path: The checkpoint path of the trained discriminator model.
        num_epochs: int
            default 200, the max epochs the model trainging.
        patience: int
            default 5. earlystop epoch patience on valid dataset.
        display_interval: int
            default 25. the iterations interval to print the the loss infos.
        gradient_clipping: bool
            default False. using gradient_clipping or not.
        if_initial: bool
            default True. whether initial all the conv-kernel parameters.
        if_swa: bool
            default True. whether using SWA strategy.
        start_swa: int
            start to use swa at specific epoch. default 5.
        if_mask: bool
            when True. only caculate the loss on the mask point whose mask value is 1.
        mask_weight: dict.
            when if_mask=True, it works. mask_weight set different weight to different mask.
            for example. the value of mask can be [1, 2, 3]. then we can set
            mask_weight= {1: 10, 2: 20, 3: 5}.
        """
        self.device = device
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        if not logger:
            log_file = os.path.join(save_path, 'train_log.txt')
            logger = make_log(log_file)

        self.logger = logger
        self.gen_model = gen_model.to(self.device)
        self.dis_model = dis_model.to(self.device)
        self.if_initial = if_initial

        if gen_checkpoint_path is not None and os.path.exists(gen_checkpoint_path):
            self.logger.info(f'load trained generator model from: {gen_checkpoint_path}')
            self.gen_model, _ = self.load_model(self.gen_model, gen_checkpoint_path)
            self.if_initial = False
        if dis_checkpoint_path is not None and os.path.exists(dis_checkpoint_path):
            self.logger.info(f'load trained discriminator model from: {dis_checkpoint_path}')
            self.dis_model, _ = self.load_model(self.dis_model, dis_checkpoint_path)
            self.if_initial = False

        self.logger.info(f'initial model: {self.if_initial}')
        self.optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.lr_scheduler = scheduler
        self.gen_step = gen_step
        self.dis_step = dis_step
        self.loss_fn = gen_loss
        self.dis_fn = dis_loss
        self.dis_loss_weight = dis_loss_weight
        self.gen_loss_weight = gen_loss_weight
        self.if_swa = if_swa

        # Get the the name of discriminator loss. if dis_loss is a function, then we will get the function name
        self.dis_loss_name = dis_loss.__name__ if '__name__' in dis_loss.__dir__() else None
        if self.dis_loss_name is not None:
            self.logger.info(f'dis_loss_name: {self.dis_loss_name}')

        # Get the name of discriminator model. Eg: WGAN
        self.dis_model_name = self.dis_model.__name__() if '__name__' in self.dis_model.__dir__() else None
        if self.dis_model_name is not None:
            self.logger.info(f'dis_model_name: {self.dis_model_name}')

        self.num_epochs = num_epochs
        self.patience = patience
        self.display_interval = display_interval
        self.gradient_clipping = gradient_clipping
        self.clipping_threshold = clipping_threshold

        if self.if_swa:
            self.optimizer = SWA(self.optimizer)
        self.start_swa = start_swa
        self.if_mask = if_mask
        self.mask_weight_dict = mask_weight_dict
        if self.mask_weight_dict is not None:
            assert isinstance(self.mask_weight, dict)

        # judge whether 'get_lr' method is in self.lr_scheduler
        self.lr_flag = 'get_lr' in self.lr_scheduler.__dir__()

        self.logger.info('--------------------------optimizer----------------------------')
        if '__str__' in self.optimizer.__dir__():
            self.logger.info(self.optimizer.__str__())
        self.logger.info('--------------------------lr_scheduler-------------------------')
        if '__str__' in self.lr_scheduler.__dir__():
            self.logger.info(self.lr_scheduler.__str__())

    def log_shape(self, input_x, input_y):
        logger.info(f'input_x: {type(input_x)}')
        if isinstance(input_x, list):
            for i, x in enumerate(input_x):
                logger.info(f'{i} x.shape: {x.shape}')
        else:
            logger.info(f'input_x.shape: {input_x.shape}')

        logger.info(f'input_y: {type(input_y)}. shape:')
        if isinstance(input_y, list):
            for i, y in enumerate(input_y):
                logger.info(f'{i} y.shape: {y.shape}')
        else:
            logger.info(f'input_y.shape: {input_y.shape}')

    def eval_discriminator_once(self, target_y, pre_y):

        self.dis_model.eval()
        with torch.no_grad():
            real_preds = self.dis_model(target_y)
            fake_preds = self.dis_model(pre_y)

            if self.dis_loss_name == 'WGANLoss':
                # gp = compute_gradient_penalty(self.dis_model, target_y.data, pre_y.data)
                loss = self.dis_loss_weight * (self.dis_fn(fake_preds) - self.dis_fn(real_preds))
            else:
                real_preds = torch.clamp(real_preds, min=1e-2, max=1 - 1e-2)
                fake_preds = torch.clamp(fake_preds, min=1e-2, max=1 - 1e-2)

                if self.dis_loss_name == 'HingeLoss':
                    real_loss = self.dis_fn(real_preds, flag='real')
                    fake_loss = self.dis_fn(fake_preds, flag='fake')
                    # Combine losses and optimize
                    loss = self.dis_loss_weight * (real_loss + fake_loss) / 2
                elif self.dis_loss_name == 'DGMRLoss':
                    loss = self.dis_loss_weight * self.dis_fn(fake_preds, real_preds, flag='dis')
                else:
                    # Inference discriminator on real images
                    real_targets = torch.ones(target_y.size(0), 1).to(self.device)
                    real_loss = self.dis_fn(real_preds, real_targets)

                    fake_targets = torch.zeros(pre_y.size(0), 1).to(self.device)
                    fake_loss = self.dis_fn(fake_preds, fake_targets)

                    # Combine losses and optimize
                    loss = self.dis_loss_weight * (real_loss + fake_loss) / 2

        return loss.item(), real_preds, fake_preds

    def train_discriminator_once(self, target_y, pre_y):
        # self.dis_model.train()
        self.dis_optimizer.zero_grad()

        # Train discriminator on real images
        real_preds = self.dis_model(target_y)

        # Train discriminator on fake images
        # detach here is necessary. 需要中间变量，又不需要其求梯度的话最好加detach()
        fake_preds = self.dis_model(pre_y.detach())

        if self.dis_loss_name == 'WGANLoss':
            gp = compute_gradient_penalty(self.dis_model, target_y.data, pre_y.data)
            loss = self.dis_loss_weight * (self.dis_fn(fake_preds) - self.dis_fn(real_preds) + 10 * gp)

        else:
            real_preds = torch.clamp(real_preds, min=1e-2, max=1 - 1e-2)
            fake_preds = torch.clamp(fake_preds, min=1e-2, max=1 - 1e-2)

            if self.dis_loss_name == 'HingeLoss':
                real_loss = self.dis_fn(real_preds, flag='real')
                fake_loss = self.dis_fn(fake_preds, flag='fake')
                # Combine losses and update optimize
                loss = self.dis_loss_weight * (real_loss + fake_loss) / 2
            elif self.dis_loss_name == 'DGMRLoss':
                fake_preds = torch.clamp(fake_preds, min=1e-2, max=1 - 1e-2)
                loss = self.dis_loss_weight * self.dis_fn(fake_preds, real_preds, flag='dis')
            else:
                # Smooth Label.
                real_targets = torch.ones(target_y.size(0), 1).uniform_(0.8, 1.0).to(self.device)
                # 将真实样本标签反转
                real_targets = flip_lables(real_targets, p=0.05)
                real_loss = self.dis_fn(real_preds, real_targets)

                fake_targets = torch.zeros(pre_y.size(0), 1).uniform_(0.0, 0.2).to(self.device)
                fake_targets = flip_lables(fake_targets, p=0.05)
                fake_loss = self.dis_fn(fake_preds, fake_targets)

                # Combine losses and update optimize
                loss = self.dis_loss_weight * (real_loss + fake_loss) / 2
        # loss.backward(retain_graph=True)
        loss.backward()
        if self.gradient_clipping:
            clip_gradients(self.dis_model, clip_value=0.1)
            # nn.utils.clip_grad_norm_(self.dis_model.parameters(), self.clipping_threshold)
        self.dis_optimizer.step()

        return loss.item(), real_preds, fake_preds

    def train_generator_once(self, input_x, input_y, decoder_input=None,
                             index=0, update_gen_model=True):
        '''
        func: train one iteration and return loss.
        Parameter
        ---------
        index: int
            the specific step of the whole iterations.
        update_gen_model: bool
            if update the generator model
        '''
        self.gen_model.train()
        self.optimizer.zero_grad()
        if isinstance(input_x, list):
            input_x = [x.float().to(self.device) for x in input_x]
        else:
            input_x = input_x.float().to(self.device)

        if isinstance(input_y, list):
            input_y = [y.float().to(self.device) for y in input_y]
            if self.if_mask:
                target = input_y[0]
                mask = input_y[1]
            else:
                target = input_y[0]
        else:
            target = input_y.float().to(self.device)

        # add decoder_input into the model.
        if decoder_input is not None:
            decoder_input = decoder_input.to(self.device)
            output = self.gen_model(input_x, decoder_input)
        else:
            output = self.gen_model(input_x)

        if index % self.display_interval == 0:
            self.logger.info(f'y_pre --- min:{output.min():.5f} max: {output.max():.5f} mean: {output.mean():.5f}')
            self.logger.info(f'y_true --- min:{target.min():.5f} max: {target.max():.5f} mean: {target.mean():.5f}')

        if update_gen_model:
            # gen_loss1: the loss between the pred and label
            # if have any mask. using mask.
            if isinstance(input_y, list) and self.if_mask:
                if self.mask_weight_dict is not None:
                    mask = self.weight_mask_value(mask)
                gen_loss1 = self.loss_fn(output * mask, target * mask)
            else:
                gen_loss1 = self.loss_fn(output, target)
            gen_loss1 = self.gen_loss_weight * gen_loss1

            # gen_loss_2: Train generator to fool discriminator by maximizing log(D(G(z)))
            # self.dis_model.eval()
            preds = self.dis_model(output)
            if self.dis_loss_name == 'WGANLoss':
                gen_loss2 = -self.dis_loss_weight * self.dis_fn(preds)
            else:
                preds = torch.clamp(preds, min=1e-2, max=1 - 1e-2)
                targets = torch.ones(output.size(0), 1).to(self.device)
                if self.dis_loss_name == 'HingeLoss':
                    gen_loss2 = self.dis_loss_weight * self.dis_fn(preds, flag='real')
                elif self.dis_loss_name == 'DGMRLoss':
                    gen_loss2 = self.dis_loss_weight * self.dis_fn(preds, targets, flag='gen')
                else:
                    gen_loss2 = self.dis_loss_weight * self.dis_fn(preds, targets)
            loss = gen_loss1 + gen_loss2
            loss.backward()

            if self.gradient_clipping:
                nn.utils.clip_grad_norm_(self.gen_model.parameters(), self.clipping_threshold)

            self.optimizer.step()
            return output, target, loss.item(), gen_loss1.item(), gen_loss2.item()
        else:
            return output, target

    def train_epoch(self, dataloader_train, epoch=0):
        '''
        Expect the output of dataloader_train are: [index, x, y, decoder_input].
        '''
        one_epoch_all_gen_loss = []
        one_epoch_all_gen_loss1 = []
        one_epoch_all_gen_loss2 = []
        one_epoch_all_dis_loss = []
        for j, out in enumerate(dataloader_train):
            if len(out) == 2:
                input_x, input_y = out[0], out[1]
                decoder_input = None
            if len(out) == 3:
                input_x, input_y = out[1], out[2]
                decoder_input = None
            elif len(out) == 4:
                input_x, input_y, decoder_input = out[1], out[2], out[3]
            if j == 0:
                self.log_shape(input_x, input_y)
                if decoder_input is not None:
                    logger.info(f'decoder_input.shape: {decoder_input.shape}')

            # update generator model per self.gen_step
            if j % self.gen_step == 0:
                gen_output = self.train_generator_once(input_x, input_y,
                                                       decoder_input=decoder_input,
                                                       index=j, update_gen_model=True)
                pre_y, target_y, gen_loss_sum, gen_loss1, gen_loss2 = gen_output
                one_epoch_all_gen_loss.append(gen_loss_sum)
                one_epoch_all_gen_loss1.append(gen_loss1)
                one_epoch_all_gen_loss2.append(gen_loss2)
            else:
                gen_output = self.train_generator_once(input_x, input_y,
                                                       decoder_input=decoder_input,
                                                       index=j, update_gen_model=False)
                pre_y, target_y = gen_output

            # update discriminator model per self.dis_step
            if j % self.dis_step == 0:
                dis_loss, real_preds, fake_preds = self.train_discriminator_once(target_y, pre_y)
                one_epoch_all_dis_loss.append(dis_loss)

            if j % self.display_interval == 0:
                self.logger.info(f'Train: Epoch {epoch}/{self.num_epochs} step: {j} ---gen_loss: {gen_loss_sum:.4f}  '
                                 f'gen_loss1: {gen_loss1:.4f} gen_loss2: {gen_loss2:.4f} ---dis_loss: {dis_loss:.4f}')
                self.logger.info(f'Train: Epoch {epoch}/{self.num_epochs} step: {j} ---real_preds.mean(): {real_preds.mean():.4f}  '
                                 f'---fake_preds.mean(): {fake_preds.mean():.4f}')

                save_file = os.path.join(self.save_path, 'examples', 'train', f'epoch{epoch}_sample{j}.png')
                data = aggregate_obs_pre(obs=target_y.cpu().numpy(), pre=pre_y.cpu().numpy(), mu=0, scale=90)
                plot_rain(data,
                          title=f'sample {j}',
                          save_file=save_file
                          )

        return one_epoch_all_gen_loss, one_epoch_all_gen_loss1, one_epoch_all_gen_loss2, one_epoch_all_dis_loss

    def eval_once(self, input_x, input_y, decoder_input=None, index=0):
        '''
        func: eval one step.
        Parameter
        ---------
        index: int
            the specific step of the whole iterations.
        Return:
        -------
        loss:
            output_seq, input_y
        '''
        self.gen_model.eval()
        if isinstance(input_x, list):
            input_x = [x.float().to(self.device) for x in input_x]
        else:
            input_x = input_x.float().to(self.device)

        if isinstance(input_y, list):
            input_y = [y.float().to(self.device) for y in input_y]
            if self.if_mask:
                target = input_y[0]
                mask = input_y[1]
            else:
                target = input_y[0]
        else:
            target = input_y.float().to(self.device)

        with torch.no_grad():
            # add decoder_input into the model.
            if decoder_input is not None:
                decoder_input = decoder_input.to(self.device)
                output = self.gen_model(input_x, decoder_input)
            else:
                output = self.gen_model(input_x)

            if index % self.display_interval == 0:
                self.logger.info(f'y_pre --- min:{output.min():.5f} max: {output.max():.5f} mean: {output.mean():.5f}')
                self.logger.info(f'y_true --- min:{target.min():.5f} max: {target.max():.5f} mean: {target.mean():.5f}')

            # gen_loss1: the loss between the pred and label
            # if have any mask. using mask.
            if isinstance(input_y, list) and self.if_mask:
                if self.mask_weight_dict is not None:
                    mask = self.weight_mask_value(mask)
                gen_loss1 = self.loss_fn(output * mask, target * mask)
            else:
                gen_loss1 = self.loss_fn(output, target)
            gen_loss1 = self.gen_loss_weight * gen_loss1

            # gen_loss_2: Disciminator loss between the pred and 1.
            self.dis_model.eval()
            preds = self.dis_model(output)
            if self.dis_loss_name == 'WGANLoss':
                gen_loss2 = -self.dis_loss_weight * self.dis_fn(preds)
            else:
                preds = torch.clamp(preds, min=1e-2, max=1 - 1e-2)
                targets = torch.ones(output.size(0), 1).to(self.device)
                if self.dis_loss_name == 'HingeLoss':
                    gen_loss2 = self.dis_loss_weight * self.dis_fn(preds, flag='real')
                elif self.dis_loss_name == 'DGMRLoss':
                    gen_loss2 = self.dis_loss_weight * self.dis_fn(preds, targets, flag='gen')
                else:
                    targets = torch.ones(output.size(0), 1).to(self.device)
                    gen_loss2 = self.dis_loss_weight * self.dis_fn(preds, targets)

            loss = gen_loss1 + gen_loss2

        return output, target, loss.item(), gen_loss1.item(), gen_loss2.item()

    def eval_epoch(self, dataloader_valid, epoch=0):
        '''
        func: eval on the whole dataloader_eval.
        Expect the output of dataloader_train are: [index, x, y, decoder_input].
        '''
        one_epoch_all_gen_loss = []
        one_epoch_all_gen_loss1 = []
        one_epoch_all_gen_loss2 = []
        one_epoch_all_dis_loss = []

        self.gen_model.eval()
        for j, out in enumerate(dataloader_valid):
            if len(out) == 2:
                input_x, input_y = out[0], out[1]
                decoder_input = None
            if len(out) == 3:
                input_x, input_y = out[1], out[2]
                decoder_input = None
            elif len(out) == 4:
                input_x, input_y, decoder_input = out[1], out[2], out[3]
            if j == 0:
                self.log_shape(input_x, input_y)
                if decoder_input is not None:
                    logger.info(f'decoder_input.shape: {decoder_input.shape}')
            gen_output = self.eval_once(input_x, input_y,
                                        decoder_input=decoder_input,
                                        index=j)
            pre_y, target_y, gen_loss_sum, gen_loss1, gen_loss2 = gen_output

            dis_loss, real_preds, fake_preds = self.eval_discriminator_once(target_y, pre_y)

            one_epoch_all_gen_loss.append(gen_loss_sum)
            one_epoch_all_gen_loss1.append(gen_loss1)
            one_epoch_all_gen_loss2.append(gen_loss2)
            one_epoch_all_dis_loss.append(dis_loss)

            if j % self.display_interval == 0:
                self.logger.info(f'Eval: Epoch {epoch}/{self.num_epochs} step: {j} ---gen_loss: {gen_loss_sum:.4f}  '
                                 f'gen_loss1: {gen_loss1:.4f} gen_loss2: {gen_loss2:.4f} ---dis_loss: {dis_loss:.4f}')
                self.logger.info(f'Eval: Epoch {epoch}/{self.num_epochs} step: {j} ---real_preds.mean(): {real_preds.mean():.4f}  '
                                 f'---fake_preds.mean(): {fake_preds.mean():.4f}')

                save_file = os.path.join(self.save_path, 'examples', 'valid', f'epoch{epoch}_sample{j}.png')
                data = aggregate_obs_pre(obs=target_y.cpu().numpy(), pre=pre_y.cpu().numpy(), mu=0, scale=90)
                plot_rain(data,
                          title=f'sample {j}',
                          save_file=save_file
                          )

        return one_epoch_all_gen_loss, one_epoch_all_gen_loss1, one_epoch_all_gen_loss2, one_epoch_all_dis_loss

    def inference(self, dataloader_test):
        '''
        func: inference on the whole dataloader_test.
        '''
        best_gen_model_file = os.path.join(self.save_path, 'gen_checkpoint.chk')
        best_dis_model_file = os.path.join(self.save_path, 'dis_checkpoint.chk')
        logger.info(f'Loading best generator model: {best_gen_model_file}')
        logger.info(f'Loading best discriminator model: {best_dis_model_file}')
        self.gen_model, _ = self.load_model(self.gen_model, best_gen_model_file)
        self.dis_model, _ = self.load_model(self.dis_model, best_dis_model_file)
        self.gen_model.eval()
        self.dis_model.eval()
        one_epoch_gen_loss = []
        one_epoch_gen_loss1 = []
        one_epoch_gen_loss2 = []
        all_predict = []
        all_obs = []
        all_index = []
        for j, out in enumerate(dataloader_test):
            if len(out) == 2:
                index = j
                input_x, input_y = out[0], out[1]
                decoder_input = None
            if len(out) == 3:
                index = out[0].cpu().numpy()
                input_x, input_y = out[1], out[2]
                decoder_input = None
            elif len(out) == 4:
                index = out[0].cpu().numpy()
                input_x, input_y, decoder_input = out[1], out[2], out[3]
            if j == 0:
                self.log_shape(input_x, input_y)
                self.logger.info(f'index: {index}')
                if decoder_input is not None:
                    logger.info(f'decoder_input.shape: {decoder_input.shape}')
            eval_output = self.eval_once(input_x, input_y,
                                         decoder_input=decoder_input,
                                         index=j)
            predict, obs, loss_gen, loss_gen1, loss_gen2 = eval_output
            obs_value = obs[0] if isinstance(obs, list) else obs
            one_epoch_gen_loss.append(loss_gen)
            one_epoch_gen_loss1.append(loss_gen1)
            one_epoch_gen_loss2.append(loss_gen2)
            all_predict.append(predict.cpu().numpy())
            all_obs.append(obs_value.cpu().numpy())
            all_index.append(index)

        plot_several_lines([one_epoch_gen_loss, one_epoch_gen_loss1, one_epoch_gen_loss2],
                           label_list=['Gen Loss', 'Gen Loss1', 'Gen Loss2'],
                           x_label='Iterations',
                           y_label='Loss',
                           title='Test inference',
                           save_file=os.path.join(self.save_path, 'img', 'test_loss.png')
                           )

        avg_test_gen_loss = torch.mean(torch.tensor(one_epoch_gen_loss))
        avg_test_gen_loss1 = torch.mean(torch.tensor(one_epoch_gen_loss1))
        avg_test_gen_loss2 = torch.mean(torch.tensor(one_epoch_gen_loss2))

        self.logger.info(f'Test gen loss: {avg_test_gen_loss} '
                         f'gen_loss1: {avg_test_gen_loss1} '
                         f'gen_loss2: {avg_test_gen_loss2} '
                         )

        indexs = np.concatenate(all_index)
        # stack on batch dimension.
        obs_np = np.vstack(all_obs)
        predict_np = np.vstack(all_predict)
        logger.info(f'obs.shape: {obs_np.shape}')
        logger.info(f'pred.shape: {predict_np.shape}')

        batch = obs_np.shape[0]

        for i in range(0, batch, 16):
            if i >= 3000:
                break
            data = aggregate_obs_pre(obs=obs_np[i: i + 8], pre=predict_np[i: i + 8], mu=0, scale=90)
            plot_rain(data,
                      title='sample',
                      save_file=os.path.join(self.save_path, 'examples', 'test', f'sample{i}-{i+8}.png')
                      )

        return indexs, obs_np, predict_np

    def train(self, dataloader_train, dataloader_valid):
        '''
        func: training process. validation after each train epoch, and record the loss info.
        '''
        count = 0
        best = 1e12

        # Record train loss
        all_train_epoch_all_gen_loss = []
        all_train_epoch_mean_gen_loss = []

        all_train_epoch_all_gen_loss1 = []
        all_train_epoch_mean_gen_loss1 = []

        all_train_epoch_all_gen_loss2 = []
        all_train_epoch_mean_gen_loss2 = []

        all_train_epoch_all_dis_loss = []
        all_train_epoch_mean_dis_loss = []

        # gen_loss + dis_loss
        all_train_epoch_gen_dis_loss = []

        # Record valid loss
        all_valid_epoch_all_gen_loss = []
        all_valid_epoch_mean_gen_loss = []

        all_valid_epoch_all_gen_loss1 = []
        all_valid_epoch_mean_gen_loss1 = []

        all_valid_epoch_all_gen_loss2 = []
        all_valid_epoch_mean_gen_loss2 = []

        all_valid_epoch_all_dis_loss = []
        all_valid_epoch_mean_dis_loss = []

        # gen_loss + dis_loss
        all_valid_epoch_gen_dis_loss = []

        # Record learning rate
        all_epoch_lr = []
        all_epoch = []

        if self.if_initial:
            self.initial()
        for i in range(self.num_epochs):
            all_epoch.append(i)
            self.logger.info(f'Epoch: {i}/{self.num_epochs}')
            # use swa strategy!
            if i >= self.start_swa and self.if_swa:
                self.optimizer.swap_swa_sgd()
            train_outputs = self.train_epoch(dataloader_train, epoch=i)
            (one_epoch_train_gen_loss, one_epoch_train_gen_loss1, one_epoch_train_gen_loss2,
             one_epoch_train_dis_loss) = train_outputs
            # use swa strategy!
            if i >= self.start_swa and self.if_swa:
                self.optimizer.update_swa()
                self.optimizer.swap_swa_sgd()
                self.optimizer.bn_update(dataloader_train, self.gen_model, device=self.device)

            # plot train gen loss curve of each epoch.
            # save_file = os.path.join(self.save_path, 'img', f'train_generator_epoch_{i}.png')
            # plot_several_lines(
            #     [one_epoch_train_gen_loss, one_epoch_train_gen_loss1, one_epoch_train_gen_loss2],
            #     label_list=['Gen Loss', 'Gen Loss1', 'Gen Loss2'],
            #     x_label='Iterations',
            #     y_label='Loss',
            #     title=f'Train Generator Loss -- Epoch {i}',
            #     save_file=save_file
            # )

            # plot train dis loss curve of each epoch.
            # plot_scores_fig(one_epoch_train_dis_loss, x_label='Iterations', y_label='Loss',
            #                 title=f'Train Discriminator Loss -- Epoch {i}',
            #                 save_file=os.path.join(self.save_path, 'img', f'train_discriminator_epoch_{i}.png')
            #                 )

            # get the mean train loss of this epoch.
            avg_train_gen_loss = torch.mean(torch.tensor(one_epoch_train_gen_loss))
            all_train_epoch_mean_gen_loss.append(avg_train_gen_loss)
            all_train_epoch_all_gen_loss.append(one_epoch_train_gen_loss)

            avg_train_gen_loss1 = torch.mean(torch.tensor(one_epoch_train_gen_loss1))
            all_train_epoch_mean_gen_loss1.append(avg_train_gen_loss1)
            all_train_epoch_all_gen_loss1.append(one_epoch_train_gen_loss1)

            avg_train_gen_loss2 = torch.mean(torch.tensor(one_epoch_train_gen_loss2))
            all_train_epoch_mean_gen_loss2.append(avg_train_gen_loss2)
            all_train_epoch_all_gen_loss2.append(one_epoch_train_gen_loss2)

            avg_train_dis_loss = torch.mean(torch.tensor(one_epoch_train_dis_loss))
            all_train_epoch_mean_dis_loss.append(avg_train_dis_loss)
            all_train_epoch_all_dis_loss.append(one_epoch_train_dis_loss)

            avg_train_gen_dis_loss = avg_train_gen_loss + avg_train_dis_loss
            all_train_epoch_gen_dis_loss.append(avg_train_gen_dis_loss)

            self.logger.info(f'Train: Epoch {i}/{self.num_epochs}. gen_loss: {avg_train_gen_loss:.4f}  '
                             f'gen_loss1: {avg_train_gen_loss1:.4f}  '
                             f'gen_loss2: {avg_train_gen_loss2:.4f}  '
                             f'dis_loss: {avg_train_dis_loss:.4f} '
                             f'gen_loss + dis_loss: {avg_train_gen_dis_loss:.4f}'
                             )

            # get the mean valid loss of this epoch
            valid_outputs = self.eval_epoch(dataloader_valid, epoch=i)
            (one_epoch_valid_gen_loss, one_epoch_valid_gen_loss1, one_epoch_valid_gen_loss2,
             one_epoch_valid_dis_loss) = valid_outputs

            # plot valid gen loss curve of each epoch.
            # save_file = os.path.join(self.save_path, 'img', f'valid_generator_epoch_{i}.png')
            # plot_several_lines(
            #     [one_epoch_valid_gen_loss, one_epoch_valid_gen_loss1, one_epoch_valid_gen_loss2],
            #     label_list=['Gen Loss', 'Gen Loss1', 'Gen Loss2'],
            #     x_label='Iterations',
            #     y_label='Loss',
            #     title=f'Valid Generator Loss -- Epoch {i}',
            #     save_file=save_file
            # )

            # plot valid dis loss curve of each epoch.
            # plot_scores_fig(one_epoch_valid_dis_loss, x_label='Iterations', y_label='Loss',
            #                 title=f'Valid Discriminator Loss -- Epoch {i}',
            #                 save_file=os.path.join(self.save_path, 'img', f'valid_discriminator_epoch_{i}.png')
            #                 )

            # get the mean valid loss of this epoch.
            avg_valid_gen_loss = torch.mean(torch.tensor(one_epoch_valid_gen_loss))
            all_valid_epoch_mean_gen_loss.append(avg_valid_gen_loss)
            all_valid_epoch_all_gen_loss.append(one_epoch_valid_gen_loss)

            avg_valid_gen_loss1 = torch.mean(torch.tensor(one_epoch_valid_gen_loss1))
            all_valid_epoch_mean_gen_loss1.append(avg_valid_gen_loss1)
            all_valid_epoch_all_gen_loss1.append(one_epoch_valid_gen_loss1)

            avg_valid_gen_loss2 = torch.mean(torch.tensor(one_epoch_valid_gen_loss2))
            all_valid_epoch_mean_gen_loss2.append(avg_valid_gen_loss2)
            all_valid_epoch_all_gen_loss2.append(one_epoch_valid_gen_loss2)

            avg_valid_dis_loss = torch.mean(torch.tensor(one_epoch_valid_dis_loss))
            all_valid_epoch_mean_dis_loss.append(avg_valid_dis_loss)
            all_valid_epoch_all_dis_loss.append(one_epoch_valid_dis_loss)

            avg_valid_gen_dis_loss = avg_valid_gen_loss + avg_valid_dis_loss
            all_valid_epoch_gen_dis_loss.append(avg_valid_gen_dis_loss)

            self.logger.info(f'Valid: Epoch {i}/{self.num_epochs}. gen_loss: {avg_valid_gen_loss:.4f}  '
                             f'gen_loss1: {avg_valid_gen_loss1:.4f}  '
                             f'gen_loss2: {avg_valid_gen_loss2:.4f}  '
                             f'dis_loss: {avg_valid_dis_loss:.4f}  '
                             f'gen_loss + dis_loss: {avg_valid_gen_dis_loss:.4f}'
                             )

            # get the learning rate of each epoch.
            if self.lr_flag:
                lr = self.lr_scheduler.get_lr()
                all_epoch_lr.append(lr)
                lr_info = f'Epoch_{i} lr: {lr}'
                logger.info(lr_info)
                self.lr_scheduler.step()
            else:
                self.lr_scheduler.step(avg_valid_gen_loss1)

            if avg_valid_gen_loss1 >= best:
                count += 1
                self.logger.info(f'valid loss is not improved for {count} epoch. The best loss is: {best:.4f}')
            else:
                count = 0
                self.logger.info('valid loss improved from {:.5f} to {:.5f}, save model'.format(best,
                                                                                                avg_valid_gen_loss1)
                                 )
                best = avg_valid_gen_loss1
                self.save_model(epoch=i, flag='best')
            self.save_model(epoch=i, flag='other')

            if count == self.patience:
                self.logger.info('early stopping reached, best loss is {:5f}'.format(best))
                break

        # plot the loss curve of train and valid process
        plot_several_lines(
            [all_train_epoch_mean_gen_loss, all_train_epoch_mean_gen_loss1, all_train_epoch_mean_gen_loss2],
            label_list=['Gen Loss', 'Gen Loss1', 'Gen Loss2'],
            x_label='Epoch',
            y_label='Loss',
            title='Train Generator Loss',
            save_file=os.path.join(self.save_path, 'img', 'train_generator_loss.png')
        )

        plot_scores_fig(all_train_epoch_mean_dis_loss, x_label='Epoch',
                        y_label='Train loss', title='Train Discriminator Loss',
                        save_file=os.path.join(self.save_path, 'img', 'train_discriminator_loss.png')
                        )

        plot_several_lines(
            [all_train_epoch_mean_gen_loss, all_train_epoch_mean_dis_loss, all_train_epoch_gen_dis_loss],
            label_list=['Gen Loss', 'Dis Loss', 'Gen + Dis loss'],
            x_label='Epoch',
            y_label='Loss',
            title='Train Loss',
            save_file=os.path.join(self.save_path, 'img', 'train_loss.png')
        )

        plot_several_lines(
            [all_valid_epoch_mean_gen_loss, all_valid_epoch_mean_gen_loss1, all_valid_epoch_mean_gen_loss2],
            label_list=['Gen Loss', 'Gen Loss1', 'Gen Loss2'],
            x_label='Epoch',
            y_label='Loss',
            title='Valid Generator Loss',
            save_file=os.path.join(self.save_path, 'img', 'valid_generator_loss.png')
        )

        plot_scores_fig(all_valid_epoch_mean_dis_loss, x_label='Epoch',
                        y_label='Loss', title='Valid Discriminator Loss',
                        save_file=os.path.join(self.save_path, 'img', 'valid_discriminator_loss.png')
                        )

        plot_several_lines(
            [all_valid_epoch_mean_gen_loss, all_valid_epoch_mean_dis_loss, all_valid_epoch_gen_dis_loss],
            label_list=['Gen Loss', 'Dis Loss', 'Gen + Dis loss'],
            x_label='Epoch',
            y_label='Loss',
            title='Valid Loss',
            save_file=os.path.join(self.save_path, 'img', 'valid_loss.png')
        )

        # plot lr curve.
        if self.lr_flag:
            lr_title = 'lr-Epoch'
            plot_scores_fig(all_epoch_lr, x_label='Epoch',
                            y_label='Learning rate', title=lr_title,
                            save_file=os.path.join(self.save_path, 'img', 'lr-epoch.png')
                            )

        # save the loss info into .csv
        csv_save_path = os.path.join(self.save_path, 'csv')
        os.makedirs(csv_save_path, exist_ok=True)

        pd_all_train_epoch_mean_loss = pd.DataFrame()
        pd_all_train_epoch_mean_loss['epoch'] = all_epoch
        pd_all_train_epoch_mean_loss['gen_loss'] = all_train_epoch_mean_gen_loss
        pd_all_train_epoch_mean_loss['gen_loss1'] = all_train_epoch_mean_gen_loss1
        pd_all_train_epoch_mean_loss['gen_loss1'] = all_train_epoch_mean_gen_loss1
        pd_all_train_epoch_mean_loss['dis_loss'] = all_train_epoch_mean_dis_loss
        pd_all_train_epoch_mean_loss['gen_dis_loss'] = all_train_epoch_gen_dis_loss
        pd_all_train_epoch_mean_loss.to_csv(os.path.join(csv_save_path, 'train_loss_mean.csv'), index=None)

        pd_all_valid_epoch_mean_loss = pd.DataFrame()
        pd_all_valid_epoch_mean_loss['epoch'] = all_epoch
        pd_all_valid_epoch_mean_loss['gen_loss'] = all_valid_epoch_mean_gen_loss
        pd_all_valid_epoch_mean_loss['gen_loss1'] = all_valid_epoch_mean_gen_loss1
        pd_all_valid_epoch_mean_loss['gen_loss2'] = all_valid_epoch_mean_gen_loss2
        pd_all_valid_epoch_mean_loss['dis_loss'] = all_valid_epoch_mean_dis_loss
        pd_all_valid_epoch_mean_loss['gen_dis_loss'] = all_valid_epoch_gen_dis_loss
        pd_all_valid_epoch_mean_loss.to_csv(os.path.join(csv_save_path, 'valid_loss_mean.csv'), index=None)

        train_loss = np.array(all_train_epoch_all_gen_loss)
        train_columns = list(range(train_loss.shape[1]))
        pd_all_train_epoch_all_loss = pd.DataFrame(train_loss, columns=train_columns, index=all_epoch)
        pd_all_train_epoch_all_loss['epoch'] = all_epoch
        pd_all_train_epoch_all_loss.set_index('epoch', inplace=True)
        pd_all_train_epoch_all_loss.to_csv(os.path.join(csv_save_path, 'train_gen_loss_all.csv'))

        valid_loss = np.array(all_valid_epoch_all_gen_loss)
        valid_columns = list(range(valid_loss.shape[1]))
        pd_all_valid_epoch_all_loss = pd.DataFrame(valid_loss, columns=valid_columns, index=all_epoch)
        pd_all_valid_epoch_all_loss['epoch'] = all_epoch
        pd_all_valid_epoch_all_loss.set_index('epoch', inplace=True)
        pd_all_valid_epoch_all_loss.to_csv(os.path.join(csv_save_path, 'valid_gen_loss_all.csv'))

        return self.gen_model, self.logger

    def save_model(self, epoch, flag='best'):
        '''
        func: save model and optimizer. save path is self.save_path
        '''
        if flag == 'best':
            torch.save({'net': self.gen_model.state_dict(),
                        'optimizer': self.optimizer.state_dict()},
                       os.path.join(self.save_path, 'gen_checkpoint.chk')
                       )

            torch.save({'net': self.dis_model.state_dict(),
                        'optimizer': self.dis_optimizer.state_dict()},
                       os.path.join(self.save_path, 'dis_checkpoint.chk')
                       )
            if epoch > 0:
                epoch_save_path = os.path.join(self.save_path, 'model_chk', 'best')
                os.makedirs(epoch_save_path, exist_ok=True)
                torch.save({'net': self.gen_model.state_dict(),
                            'optimizer': self.optimizer.state_dict()},
                           os.path.join(epoch_save_path, f'epoch{epoch}_gen_checkpoint.chk'))
                torch.save({'net': self.dis_model.state_dict(),
                            'optimizer': self.dis_optimizer.state_dict()},
                           os.path.join(epoch_save_path, f'epoch{epoch}_dis_checkpoint.chk'))
        else:
            if epoch > 0:
                epoch_save_path = os.path.join(self.save_path, 'model_chk', 'epoch')
                os.makedirs(epoch_save_path, exist_ok=True)
                torch.save({'net': self.gen_model.state_dict(),
                            'optimizer': self.optimizer.state_dict()},
                           os.path.join(epoch_save_path, f'epoch{epoch}_gen_checkpoint.chk'))
                torch.save({'net': self.dis_model.state_dict(),
                            'optimizer': self.dis_optimizer.state_dict()},
                           os.path.join(epoch_save_path, f'epoch{epoch}_dis_checkpoint.chk'))

    def load_model(self, model, chk_path, optimizer=None):
        '''
        func: load_model.
        '''
        if self.device.type == 'cpu':
            checkpoint = torch.load(chk_path, map_location='cpu')
        else:
            checkpoint = torch.load(chk_path)
        model.load_state_dict(checkpoint['net'])
        if optimizer is not None:
            optimizer = optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer

    def initial(self):
        '''
        func: initialize the parameters.
        '''
        for m in self.gen_model.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.normal(m.weight.data)
                # nn.init.xavier_normal(m.weight.data)
                # nn.init.constant_(m.weight.data, val = 1)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                # m.bias.data.fill_(1)
            elif isinstance(m, nn.Linear):
                # nn.init.constant_(m.weight.data, val= 1)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')

        for m in self.dis_model.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.normal(m.weight.data)
                # nn.init.xavier_normal(m.weight.data)
                # nn.init.constant_(m.weight.data, val = 1)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                # m.bias.data.fill_(1)
            elif isinstance(m, nn.Linear):
                # nn.init.constant_(m.weight.data, val= 1)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')

    def weight_mask_value(self, mask):
        '''
        func: separate mask by its value. so we can give differen weight to different mask value.
        '''
        # step1: get the unique mask value. eg: [0, 1, 2]
        unique_value = list(torch.unique(mask))

        # step2: get all the value_mask of difference value.
        v_mask_list = []
        for v in unique_value:
            v_mask_list.append(mask == v)

        # step3: give value_weight to the specific value.
        # if the value did not match key of self.mask_weight_dict,
        # the weight is the value itself.
        weight_mask = mask.clone()
        for v, v_mask in zip(unique_value, v_mask_list):
            v_weight = self.mask_weight_dict.get(v, None)
            if v_weight is None:
                continue
            weight_mask[v_mask] = v_weight
        return weight_mask


def plot_scores_fig(scores, x_label=None, y_label=None, title=None, save_file=None):
    '''
    func: plot scores.
    '''
    scores = list(scores)
    f1 = plt.figure(figsize=(10, 6))
    plt.plot(scores)
    if x_label:
        plt.xlabel(x_label, fontsize=20)
    if y_label:
        plt.ylabel(y_label, fontsize=20)
    plt.xticks(fontsize=16)
    plt.xticks(fontsize=16)

    if title:
        plt.title(title, fontsize=20)

    if save_file is not None:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        plt.savefig(save_file, dpi=200, bbox_inches='tight')

    f1.clf()

    return None


def plot_several_lines(scores_list, label_list, x_label=None, y_label=None, title=None, save_file=None):

    assert len(scores_list) == len(label_list)

    f1 = plt.figure(figsize=(10, 6))
    nums = len(scores_list)
    for i in range(nums):
        plt.plot(scores_list[i], label=label_list[i])

    if x_label:
        plt.xlabel(x_label, fontsize=20)
    if y_label:
        plt.ylabel(y_label, fontsize=20)
    plt.xticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.legend()

    if title:
        plt.title(title, fontsize=20)

    if save_file is not None:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        plt.savefig(save_file, dpi=200, bbox_inches='tight')

    f1.clf()

    return None
