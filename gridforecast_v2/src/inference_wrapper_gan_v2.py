import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import logging
from gridforecast_v2.src.utils import make_log
from gridforecast_v2.data_pipelines.utils import configure_logging


configure_logging()
logger = logging.getLogger(__name__)


class InferenceModel(object):
    def __init__(self,
                 device,
                 save_path,
                 gen_model,
                 dis_model,
                 gen_checkpoint_path=None,
                 dis_checkpoint_path=None,
                 gen_loss=nn.MSELoss(),
                 dis_loss=nn.BCELoss(),
                 gen_loss_weight=1,
                 dis_loss_weight=0.1,
                 logger=logger,
                 display_interval=25,
                 if_mask=False,
                 mask_weight_dict=None,
                 ):
        """
        func: train + test process.
        Parameter
        --------------------
        device: torch.device
            CPU or CUDA.
        gen_model: instance of the generator model.
        dis_model: instance of the discriminator model.
        gen_checkpoint_path: The checkpoint path of the trained generator model.
        dis_checkpoint_path: The checkpoint path of the trained discriminator model.
        save_path: str
            output dir of logger info. including the loss info.
            eg: /home/zuliang/model_train/test1
        logger: logging.
            record the all the log info. if not. create a new log which saved into save_path.
        gen_loss: instance of loss function used for generator model.
            default: nn.MSELoss().
        dis_loss: instance of loss function used for discriminator model.
            default: nn.BCELoss().
        dis_loss_weight: default 0.1
            sum_gen_loss = gen_loss + dis_loss_weight * dis_loss(preds, 1)
        gen_loss_weight: default 1
            sum_gen_loss = gen_loss_weight * gen_loss + dis_loss_weight * dis_loss
        display_interval: int
            default 25. the iterations interval to print the the loss infos.
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
            log_file = os.path.join(save_path, 'inference_log.txt')
            logger = make_log(log_file)

        self.logger = logger
        self.gen_model = gen_model.to(self.device)
        self.dis_model = dis_model.to(self.device)
        self.loss_fn = gen_loss
        self.dis_fn = dis_loss
        self.dis_loss_weight = dis_loss_weight
        self.gen_loss_weight = gen_loss_weight
        self.display_interval = display_interval

        self.if_mask = if_mask
        self.mask_weight_dict = mask_weight_dict
        if self.mask_weight_dict is not None:
            assert isinstance(self.mask_weight, dict)

        # Get the the name of discriminator loss. if dis_loss is a function, then we will get the function name
        self.dis_loss_name = dis_loss.__name__ if '__name__' in dis_loss.__dir__() else None
        if self.dis_loss_name is not None:
            self.logger.info(f'dis_loss_name: {self.dis_loss_name}')

        # Get the name of discriminator model. Eg: WGAN
        self.dis_model_name = self.dis_model.__name__() if '__name__' in self.dis_model.__dir__() else None
        if self.dis_model_name is not None:
            self.logger.info(f'dis_model_name: {self.dis_model_name}')

        # The directory of trained model
        if gen_checkpoint_path is not None and os.path.exists(gen_checkpoint_path):
            self.logger.info(f'load trained generator model from: {gen_checkpoint_path}')
            self.gen_model = self.load_model(self.gen_model, gen_checkpoint_path)
        if dis_checkpoint_path is not None and os.path.exists(dis_checkpoint_path):
            self.logger.info(f'load trained discriminator model from: {dis_checkpoint_path}')
            self.dis_model = self.load_model(self.dis_model, dis_checkpoint_path)

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

    def inference(self, dataloader_test):
        '''
        func: inference on the whole dataloader_test.
        '''
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
                           y_label='Test Loss',
                           title='Test inference',
                           save_file=os.path.join(self.save_path, 'img', 'test_loss.png')
                           )

        avg_test_gen_loss = torch.mean(torch.tensor(one_epoch_gen_loss))
        avg_test_gen_loss1 = torch.mean(torch.tensor(one_epoch_gen_loss1))
        avg_test_gen_loss2 = torch.mean(torch.tensor(one_epoch_gen_loss2))

        self.logger.info(f'Test gen loss: {avg_test_gen_loss:.4f}.  '
                         f'gen_loss1: {avg_test_gen_loss1:.4f}  '
                         f'gen_loss2: {avg_test_gen_loss2:.4f}  '
                         )

        indexs = np.concatenate(all_index)
        # stack on batch dimension.
        obs_np = np.vstack(all_obs)
        predict_np = np.vstack(all_predict)
        logger.info(f'obs.shape: {obs_np.shape}')
        logger.info(f'pred.shape: {predict_np.shape}')

        return indexs, obs_np, predict_np

    def load_model(self, model, chk_path):
        '''
        func: load_model.
        '''
        if self.device.type == 'cpu':
            checkpoint = torch.load(chk_path, map_location='cpu')
        else:
            checkpoint = torch.load(chk_path)
        model.load_state_dict(checkpoint['net'])
        return model

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
