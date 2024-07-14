import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import logging
from gridforecast_v2.optim.swa import SWA
import numpy as np
import pandas as pd
from gridforecast_v2.src.utils import make_log, plot_rain
from gridforecast_v2.data_pipelines.utils import configure_logging


configure_logging()
logger = logging.getLogger(__name__)


class TrainTestModel(object):
    def __init__(self,
                 device,
                 save_path,
                 model,
                 optimizer,
                 scheduler=None,
                 loss_fn=nn.MSELoss(),
                 logger=logger,
                 checkpoint_path=None,
                 num_epochs=200,
                 patience=50,
                 display_interval=25,
                 gradient_clipping=False,
                 clipping_threshold=3,
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
        model: instance of the mode.
        save_path: str
            output dir of logger info. including the loss info.
            eg: /home/zuliang/model_train/test1
        optimizer: optimizer instance.
            instance of optimizer. such as torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler: lr_scheduler instance.
            instance of learning rate scheduler.
            eg: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-8).
        logger: logging.
            record the all the log info. if not. create a new log which saved into save_path.
        loss_fn: instance of loss function
            default: nn.MSELoss().
        num_epochs: int
            default 200, the max epochs the model trainging.
        patience: int
            default 5. earlystop epoch patience on valid dataset.
        display_interval: int
            default 25. the iterations interval to print the the loss infos.
        gradient_clipping: bool
            default False. using gradient_clipping or not.
        clipping_threshold: 3.
            default 3. valid when gradient_clipping=True.
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
        self.network = model.to(self.device)
        self.if_initial = if_initial

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            self.logger.info(f'load trained model from: {checkpoint_path}')
            self.network = self.load_model(checkpoint_path, load_optimizer=False)
            self.if_initial = False
        self.logger.info(f'initial model: {self.if_initial}')
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        self.loss_fn = loss_fn
        self.if_swa = if_swa

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

    def train_once(self, input_x, input_y, decoder_input=None, index=0):
        '''
        func: train one iteration and return loss.
        Parameter
        ---------
        index: int
            the specific step of the whole iterations.
        '''
        self.network.train()
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
            output = self.network(input_x, decoder_input)
        else:
            output = self.network(input_x)

        self.optimizer.zero_grad()
        if index % self.display_interval == 0:
            self.logger.info(f'y_pre --- min:{output.min():.5f} max: {output.max():.5f} mean: {output.mean():.5f}')
            self.logger.info(f'y_true --- min:{target.min():.5f} max: {target.max():.5f} mean: {target.mean():.5f}')

        # if have any mask. using mask.
        if isinstance(input_y, list) and self.if_mask:
            if self.mask_weight_dict is not None:
                mask = self.weight_mask_value(mask)
            loss = self.loss_fn(output * mask, target * mask)
        else:
            loss = self.loss_fn(output, target)

        loss.backward()
        if self.gradient_clipping:
            nn.utils.clip_grad_norm_(self.network.parameters(), self.clipping_threshold)
        self.optimizer.step()
        return loss.item(), output, target

    def train_epoch(self, dataloader_train, epoch=0):
        '''
        Expect the output of dataloader_train are: [index, x, y, decoder_input].
        '''
        one_epoch_all_loss = []
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
            loss_train, pre_y, target_y = self.train_once(input_x, input_y,
                                                          decoder_input=decoder_input,
                                                          index=j)
            one_epoch_all_loss.append(loss_train)
            if j % self.display_interval == 0:
                self.logger.info(f'Train: Epoch {epoch}/{self.num_epochs} step: {j} --- loss: {loss_train:.4f}')
                save_file = os.path.join(self.save_path, 'examples', 'train', f'epoch{epoch}_sample{j}.png')
                plot_rain(obs=target_y.detach().cpu().numpy(), pre=pre_y.detach().cpu().numpy(),
                          title=f'sample {j}',
                          save_file=save_file
                          )

        return one_epoch_all_loss

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
        self.network.eval()
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
                output = self.network(input_x, decoder_input)
            else:
                output = self.network(input_x)

            if index % self.display_interval == 0:
                self.logger.info(f'y_pre --- min:{output.min():.5f} max: {output.max():.5f} mean: {output.mean():.5f}')
                self.logger.info(f'y_true --- min:{target.min():.5f} max: {target.max():.5f} mean: {target.mean():.5f}')

            # if have any mask. using mask.
            if isinstance(input_y, list) and self.if_mask:
                if self.mask_weight_dict is not None:
                    mask = self.weight_mask_value(mask)
                loss = self.loss_fn(output * mask, target * mask)
            else:
                loss = self.loss_fn(output, target)

        return loss.item(), output, target

    def eval_epoch(self, dataloader_valid, epoch=0):
        '''
        func: eval on the whole dataloader_eval.
        Expect the output of dataloader_train are: [index, x, y, decoder_input].
        '''
        self.network.eval()
        one_epoch_all_loss = []
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
            loss_test, pre_y, target_y = self.eval_once(input_x, input_y,
                                                        decoder_input=decoder_input,
                                                        index=j)
            one_epoch_all_loss.append(loss_test)
            if j % self.display_interval == 0:
                logger.info(f'Eval: Epoch {epoch}/{self.num_epochs} step: {j} --- loss: {loss_test:.5f}')
                save_file = os.path.join(self.save_path, 'examples', 'valid', f'epoch{epoch}_sample{j}.png')
                plot_rain(obs=target_y.cpu().numpy(), pre=pre_y.cpu().numpy(),
                          title=f'sample {j}',
                          save_file=save_file
                          )

        return one_epoch_all_loss

    def inference(self, dataloader_test):
        '''
        func: inference on the whole dataloader_test.
        '''
        best_model_file = os.path.join(self.save_path, 'checkpoint.chk')
        logger.info(f'Starting best model: {best_model_file}')
        self.network = self.load_model(best_model_file)
        self.network.eval()
        one_epoch_all_loss = []
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
            loss_test, predict, obs = self.eval_once(input_x, input_y,
                                                     decoder_input=decoder_input,
                                                     index=j)
            obs_value = obs[0] if isinstance(obs, list) else obs
            one_epoch_all_loss.append(loss_test)
            all_predict.append(predict.cpu().numpy())
            all_obs.append(obs_value.cpu().numpy())
            all_index.append(index)

        plot_scores_fig(one_epoch_all_loss, x_label='Iterations',
                        y_label='Test loss', title='Test inference',
                        save_file=os.path.join(self.save_path, 'img', 'test_loss.png'))
        avg_test_loss = torch.mean(torch.tensor(one_epoch_all_loss))
        self.logger.info(f'Test loss: {avg_test_loss}')

        indexs = np.concatenate(all_index)
        # stack on batch dimension.
        obs_np = np.vstack(all_obs)
        predict_np = np.vstack(all_predict)
        logger.info(f'obs.shape: {obs_np.shape}')
        logger.info(f'pred.shape: {predict_np.shape}')

        return indexs, obs_np, predict_np

    def train(self, dataloader_train, dataloader_valid):
        '''
        func: training process. validation after each train epoch, and record the loss info.
        '''
        count = 0
        best = 1e12

        all_train_epoch_all_loss = []
        all_train_epoch_mean_loss = []
        all_valid_epoch_all_loss = []
        all_valid_epoch_mean_loss = []
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
            one_epoch_all_loss = self.train_epoch(dataloader_train, epoch=i)
            # use swa strategy!
            if i >= self.start_swa and self.if_swa:
                self.optimizer.update_swa()
                self.optimizer.swap_swa_sgd()
                self.optimizer.bn_update(dataloader_train, self.network, device=self.device)

            # plot loss curve of one epoch.
            save_file = os.path.join(self.save_path, 'img', f'train_epoch_{i}.png')
            plot_scores_fig(one_epoch_all_loss, x_label='Iterations',
                            y_label='Train loss', title=f'Train Epoch {i}',
                            save_file=save_file)

            # get the mean train loss of this epoch.
            avg_train_loss = torch.mean(torch.tensor(one_epoch_all_loss))
            all_train_epoch_mean_loss.append(avg_train_loss)
            self.logger.info(f'Train: Epoch {i}/{self.num_epochs}. loss: {avg_train_loss}')
            all_train_epoch_all_loss.append(one_epoch_all_loss)

            # get the mean valid loss of this epoch
            valid_one_epoch_all_loss = self.eval_epoch(dataloader_valid, epoch=i)

            # plot the valid loss change curve of this epoch
            save_file = os.path.join(self.save_path, 'img', f'valid_epoch_{i}.png')
            plot_scores_fig(valid_one_epoch_all_loss, x_label='Iterations',
                            y_label='Valid loss', title=f'Valid Epoch {i}',
                            save_file=save_file)

            avg_valid_loss = torch.mean(torch.tensor(valid_one_epoch_all_loss))
            all_valid_epoch_mean_loss.append(avg_valid_loss)
            self.logger.info(f'Valid: Epoch {i}/{self.num_epochs}. loss: {avg_valid_loss}')
            all_valid_epoch_all_loss.append(valid_one_epoch_all_loss)

            # get the learning rate of each epoch.
            if self.lr_flag:
                lr = self.lr_scheduler.get_lr()
                all_epoch_lr.append(lr)
                lr_info = f'Epoch_{i} lr: {lr}'
                logger.info(lr_info)
                self.lr_scheduler.step()
            else:
                self.lr_scheduler.step(avg_valid_loss)

            if avg_valid_loss >= best:
                count += 1
                self.logger.info(f'valid loss is not improved for {count} epoch')
            else:
                count = 0
                self.logger.info('valid loss improved from {:.5f} to {:.5f}, save model'.format(best, avg_valid_loss))
                self.save_model(epoch=i)
                best = avg_valid_loss

            if count == self.patience:
                self.logger.info('early stopping reached, best loss is {:5f}'.format(best))
                break

        # plot the loss curve of train and valid process
        train_title = 'Train Loss Epoch'
        valid_title = 'Valid Loss Epoch'

        plot_scores_fig(all_train_epoch_mean_loss, x_label='Epoch',
                        y_label='Train loss', title=train_title,
                        save_file=os.path.join(self.save_path, 'img', 'train_loss.png')
                        )
        plot_scores_fig(all_valid_epoch_mean_loss, x_label='Epoch',
                        y_label='Valid loss', title=valid_title,
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
        pd_all_train_epoch_mean_loss['loss'] = all_train_epoch_mean_loss
        pd_all_train_epoch_mean_loss.to_csv(os.path.join(csv_save_path, 'train_loss_mean.csv'), index=None)

        pd_all_valid_epoch_mean_loss = pd.DataFrame()
        pd_all_valid_epoch_mean_loss['epoch'] = all_epoch
        pd_all_valid_epoch_mean_loss['loss'] = all_valid_epoch_mean_loss
        pd_all_valid_epoch_mean_loss.to_csv(os.path.join(csv_save_path, 'valid_loss_mean.csv'), index=None)

        train_loss = np.array(all_train_epoch_all_loss)
        train_columns = list(range(train_loss.shape[1]))
        pd_all_train_epoch_all_loss = pd.DataFrame(train_loss, columns=train_columns, index=all_epoch)
        pd_all_train_epoch_all_loss['epoch'] = all_epoch
        pd_all_train_epoch_all_loss.set_index('epoch', inplace=True)
        pd_all_train_epoch_all_loss.to_csv(os.path.join(csv_save_path, 'train_loss_all.csv'))

        valid_loss = np.array(all_valid_epoch_all_loss)
        valid_columns = list(range(valid_loss.shape[1]))
        pd_all_valid_epoch_all_loss = pd.DataFrame(valid_loss, columns=valid_columns, index=all_epoch)
        pd_all_valid_epoch_all_loss['epoch'] = all_epoch
        pd_all_valid_epoch_all_loss.set_index('epoch', inplace=True)
        pd_all_valid_epoch_all_loss.to_csv(os.path.join(csv_save_path, 'valid_loss_all.csv'))

        return self.network, self.logger

    def save_model(self, epoch):
        '''
        func: save model and optimizer. save path is self.save_path
        '''
        torch.save({'net': self.network.state_dict(),
                    'optimizer': self.optimizer.state_dict()},
                   os.path.join(self.save_path, 'checkpoint.chk')
                   )

        if epoch > 0:
            epoch_save_path = os.path.join(self.save_path, 'model_chk')
            os.makedirs(epoch_save_path, exist_ok=True)
            torch.save({'net': self.network.state_dict(),
                        'optimizer': self.optimizer.state_dict()},
                       os.path.join(epoch_save_path, f'epoch{epoch}_checkpoint.chk'))

    def load_model(self, chk_path, load_optimizer=False):
        '''
        func: load_model.
        '''
        checkpoint = torch.load(chk_path)
        self.network.load_state_dict(checkpoint['net'])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        return self.network

    def initial(self):
        '''
        func: initialize the parameters.
        '''
        for m in self.network.modules():
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

    plt.show()
    f1.clf()

    return None
