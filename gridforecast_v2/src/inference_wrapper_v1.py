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
                 model,
                 checkpoint_path=None,
                 loss_fn=nn.MSELoss(),
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
        model: instance of the mode.
        save_path: str
            output dir of logger info. including the loss info.
            eg: /home/zuliang/model_train/test1
        logger: logging.
            record the all the log info. if not. create a new log which saved into save_path.
        loss_fn: instance of loss function
            default: nn.MSELoss().
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
        self.network = model.to(self.device)
        self.loss_fn = loss_fn
        self.display_interval = display_interval

        self.if_mask = if_mask
        self.mask_weight_dict = mask_weight_dict
        if self.mask_weight_dict is not None:
            assert isinstance(self.mask_weight, dict)

        # The directory of trained model
        self.checkpoint_path = checkpoint_path
        if self.checkpoint_path is not None:
            logger.info(f'Load trained model.Checkpoint path is: {self.checkpoint_path}')
            self.network = self.load_model(self.checkpoint_path)

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
                self.logger.info(f'y_pre --- min:{output.min()} max: {output.max()} mean: {output.mean()}')
                self.logger.info(f'y_true --- min:{target.min()} max: {target.max()} mean: {target.mean()}')

            # if have any mask. using mask.
            if isinstance(input_y, list) and self.if_mask:
                if self.mask_weight_dict is not None:
                    mask = self.weight_mask_value(mask)
                loss = self.loss_fn(output * mask, target * mask)
            else:
                loss = self.loss_fn(output, target)

        return loss.item(), output, input_y

    def inference(self, dataloader_test):
        '''
        func: inference on the whole dataloader_test.
        '''
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

    def load_model(self, chk_path):
        '''
        func: load_model.
        '''
        if self.device.type == 'cpu':
            checkpoint = torch.load(chk_path, map_location='cpu')
        else:
            checkpoint = torch.load(chk_path)
        self.network.load_state_dict(checkpoint['net'])
        return self.network

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
