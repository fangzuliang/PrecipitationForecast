# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:11:52 2020
'''
输入测试数据的Dataloader + 加载了参数的模型 ：输出在某些特定评价指标下的模型得分
'''
@author: fzl
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import confusion_matrix

try:
    from skimage.measure import compare_ssim  
except Exception as e:
    import skimage.metrics.structural_similarity as compare_ssim

import Radar_utils

class Model_Evaluate():
    
    '''
    func: 在指定dataloader_test上对模型适用某种评价指标进行评估.
    Parameters
    ----------
    model: instance of class
        加载好的模型
    dataloader_test: instance of class
        加载dataloader,其batch_size 必须为1
    device: torch.device
        cuda 或者 cpu
    metric: str
        评价指标,可选用['Multi_HSS','MSE','MAE','SSIM','TS','MAR','FAR','POD','BIAS','F1','ACC','Single_HSS']
        当metric in ['MSE','MAE','SSIM']时，scale必须等于1
        其中Multi_HSS: 分为多个DBZ阈值综合考虑obs 和 pre之间的相似度; scale = 75
        当metric in ['TS','MAR','FAR','POD','BIAS','F1','ACC','Single_HSS']时，scale = 75
    scale: int
        由于模型的输出为[0,1]之间,为方便比对需要 *scale, 数值扩充到[0,scale]范围
        
    threshold: int
        当metric in ['TS','MAR','FAR','POD','BIAS','F1','ACC','Single_HSS'] 时,threshold才其作用
        当metric in ['MSE','MAE']时，threshold不起作用
        数值范围在[0,75]之间，通常取值[1,5,10,15,20,25,30,35,40,45,50,55,60]
    '''
    
    def __init__(self,model, dataloader_test, 
                 device, metric='TS',
                 scale=75, threshold=10,
                 y_mask=False):

        self.device = device
        
        self.model = model
        self.dataloader_test = dataloader_test
        
        self.metric = metric
        self.scale = scale
        self.threshold = threshold
        self.y_mask = y_mask

    def plot_compare_obs_pre(self,obs, pre,scale = None,save_path = None):
        '''
        func: 对比Dataloader中输出的 obs 和 模型对应预测的pre，图形展示
        parameters
        ################
        obs: Tensor , 4D or 5D
            真实值 
        pre: Tensor
            预测值
        scale: int
            default 75
            由于模型的输出为[0,1]之间,为方便比对需要 *scale, 数值扩充到[0,scale]范围
        save_path: None or str
            default None.则不保存obs 和 pre的比较图, 
            when str. 则为雷达图的保存位置. eg: '/mnt/nvme1/fzl/radar_img'
        '''
        
        if scale is None:
            scale = self.scale
        
        obs = obs.cpu().detach().numpy() * scale
        pre = pre.cpu().detach().numpy() * scale
        
        assert len(obs.shape) in [4,5]
    
        if len(obs.shape) == 5:
            batch,seq, channel, height,width = obs.shape
            index = np.random.randint(0,batch)
            for k in range(seq):
                obs_img = obs[index,k,0,:,:]
                pre_img = pre[index,k,0,:,:]
                save_filepath = os.path.join(save_path,str(k) + '.png') if save_path else None
                Radar_utils.plot_two_radar_img(obs_img, pre_img, index = k,save_filepath = save_filepath)
                plt.show()
                
        if len(obs.shape) == 4:
            batch,channel,height,width = obs.shape
            index = np.random.randint(0,batch)
            for k in range(channel):
                obs_img = obs[index,k,:,:]
                pre_img = pre[index,k,:,:]
                save_filepath = os.path.join(save_path,str(k) + '.png') if save_path else None
                Radar_utils.plot_two_radar_img(obs_img, pre_img, index = k,save_filepath = save_filepath)
                plt.show()
            
        return None

    def one_batch_metric(self, obs, pre, metric=None, scale=None, threshold=None, if_metric = False):               
        '''
        func: 对一个batch内obs和pre进行评价，默认输入的batch维为1
        Parameter
        ---------
        obs: tensor
            真实值
        pre: tensor
            预测值
        metric: str or None or all
            when None, 则metric = self.metric
        scale: int or None
            when None, 则 scale = self.scale
        threshold: int or None
            when None, 则 threshold = self.threshold
        if_metric: bool
            default False.当metric in  ['TS','MAR','FAR','POD','BIAS','F1','ACC','Single_HSS'] 时，是否返回对应的score.
            when False.不返回这个one_batch的得分，只是返回对应的 [TN, FN, FP, TP],即[correctnegatives, misses, falsealarms, hits] 
            when True. 则返回这个batch下的 对应的metric的得分
        '''
        
        if metric is None:
            metric = self.metric

        if scale is None:
            scale = self.scale

        if threshold is None:
            threshold = self.threshold
        
        obs = obs.cpu().detach().numpy() 
        pre = pre.cpu().detach().numpy() 
        
        assert obs.shape == pre.shape
        assert len(obs.shape) in [4,5]
        
        #如果维度为5，则保证channel为1
        if len(obs.shape) == 5:
            assert obs.shape[2] == 1 #channel通道为1
            obs = obs[:,:,0,:,:]
            pre = pre[:,:,0,:,:]
            
        batch, seq, height,width = obs.shape
        
        ####
        #step1: 先获取逐帧的评分
        #step2: 或者所有帧(完整的样本序列)的一个综合得分
        
        scores = [] #得分列表
        if metric == 'Multi_HSS':
            for i in range(seq):
                obs_img = obs[:,i,:,:]*scale
                pre_img = pre[:,i,:,:]*scale
                hss_score = Radar_utils.HSS(obs_img,pre_img)           
                scores.append(hss_score)
            
            #样本序列的综合得分
            if seq > 1:
                sample_score = Radar_utils.HSS(obs, pre)
                scores.append(sample_score) 

        if metric in ['MSE','MAE']:
            scale = 1
            for i in range(seq):
                obs_img = obs[:,i,:,:]*scale
                pre_img = pre[:,i,:,:]*scale    
                
                score = Radar_utils.MSE(obs_img,pre_img) if metric == 'MSE' else Radar_utils.MAE(obs_img,pre_img)
                scores.append(score)
                
            #样本序列的综合得分
            if seq > 1:
                sample_score = Radar_utils.MSE(obs,pre) if metric == 'MSE' else Radar_utils.MAE(obs,pre)
                scores.append(sample_score)

        if metric == 'SSIM':
            scale == self.scale
            for i in range(seq):
                obs_img = obs[:,i,:,:]*scale
                pre_img = pre[:,i,:,:]*scale   
                
                #计算SSIM得分时，size --> [height,width, channels]
                score = compare_ssim(obs_img.transpose(1,2,0),pre_img.transpose(1,2,0),multichannel = True)
                scores.append(score)
            
            if seq > 1:
                #shape = [height,width,batch_size * seq]
                obs = obs.reshape(-1,height, width).transpose(1,2,0)
                pre = pre.reshape(-1,height, width).transpose(1,2,0)
                sample_score = compare_ssim(obs,pre)
                scores.append(sample_score)
            
        if metric in ['TS','MAR','FAR','POD','BIAS','F1','ACC','Single_HSS']:
            scale = self.scale            
            obs = obs * scale
            pre = pre * scale

            # 根据给定的阈值进行 0-1化
            obs = np.where(obs >= threshold,1,0)
            pre = np.where(pre >= threshold,1,0)

            for i in range(seq + 1):

                # 当序列长度 > 1时，需要计算所有seq的综合得分(综合得分放到score_list的最后)
                if seq > 1:
                    if i == seq:
                        obs_img = obs
                        pre_img = pre
                    else:
                        obs_img = obs[:,i,:,:]
                        pre_img = pre[:,i,:,:]   
                
                #当seq = 1,则只计算一次
                elif seq == 1:
                    if i == 0:
                        continue
                    obs_img = obs
                    pre_img = pre
                    
                #计算混淆矩阵，得到TN,FN,FP,TP
                matrix = confusion_matrix(obs_img.ravel(), pre_img.ravel())
                
                #当obs 和 pre全为0时，则对应的matrix只有 TN, shape = (1,1)
                if len(matrix) == 1:
                    TN = matrix[0,0]
                    FN, FP, TP = 0,0,0
                else:
                    #分别对应：correctnegatives, misses, falsealarms, hits
                    TN, FN, FP, TP = matrix[0,0], matrix[1,0], matrix[0,1],matrix[1,1]
                    
                TN = np.float64(TN)
                FN = np.float64(FN)
                FP = np.float64(FP)
                TP = np.float64(TP)
                
                #如果需要计算一个batch内的得分(if_metric = True)，
                if if_metric: 
                    #correctnegatives, misses, falsealarms, hits = TN, FN, FP, TP
                    if metric == 'TS':
                        score = TP/(TP + FN + FP) 
                    elif metric == 'MAR':
                        score = FN / (TP + FN)
                    elif metric == 'FAR':
                        score = FP / (TP + FP)
                    elif metric == 'POD':
                        score = TP / (TP + FN)
                    elif metric == 'BIAS':
                        score = (TP + FP) / (TP + FN)
                    elif metric == 'F1':
                        rec_score = TP / (TP + FN)
                        pre_score = TP / (TP + FP)
                        score = 2 * ((pre_score * rec_score) / (pre_score + rec_score))
                    elif metric == 'ACC':
                        score = (TP + TN) / (TP + TN + FP + FN) 
                    elif metric == 'Single_HSS':
                        HSS_num = 2 * (TP * TN - FN * FN)
                        HSS_den = (FN**2 + FP**2 + 2*TP*TN + (FN + FP)*(TP + TN))
                        score = HSS_num/ HSS_den
                    else: 
                        print('No such metric!')
                    scores.append(score)
                
                #如果 if_metric = False,只是返回所有的seq的 [TN,FN,FP,TP]
                else:
                    scores.append([TN,FN,FP,TP])
               
        return scores
    
    def one_batch_more_thresholds_metric(self, obs, pre, metric=None, scale=None, thresholds = [15,20,25,30,35,40,45,50]):
        '''
        func: 对一个batch内obs和pre进行多个阈值的评价，默认输入的batch维为1 
        Parameter
        ---------
        obs: tensor
            真实值
        pre: tensor
            预测值
        metric: str or None or all
            when None, 则metric = self.metric
            metic in ['TS','MAR','FAR','POD','BIAS','F1','ACC','Single_HSS']
        scale: int or None
            when None, 则 scale = self.scale
        thresholds: list or int
            default [15,20,25,30,35,40,45,50]
        
        Return
        ------
        
        '''
        if metric == None:
            metric = self.metric
        if scale == None:
            scale = self.scale
        
        all_thresholds_score = []

        for threshold in thresholds:
            scores = self.one_batch_metric(obs,pre, metric = metric, scale=scale, threshold = threshold, if_metric = True)
            print('threshold:{},scores:{}'.format(threshold, scores))
            all_thresholds_score.append(scores)

        return all_thresholds_score
    
    def plot_more_threshold_scores_curve(self, more_thresholds_score, title = None,thresholds = [15,20,25,30,35,40,45,50]):
        '''
        func: 画出某个metric下的多个阈值下的scores变化
        Parameter
        ---------
        more_thresholds_score: list or list
            每个threshold对应一个list ---> 每一帧的评分
            多个list组成的list
        title: None or str
            图的title
            when None. title = self.metric
        thresholds: list
            
        '''
        assert len(more_thresholds_score) == len(thresholds)
        
        if title == None:
            title = self.metric
        
        f1 = plt.figure(figsize = (10,7))
        
        for threshold,scores in zip(thresholds,more_thresholds_score):
        
            seq_scores = scores[0:-1] 
            last_score = scores[-1] 
            
            if len(scores) > 1:
                seq = len(scores) - 1
                line = plt.plot(np.arange(seq),seq_scores,'-*',label = str(threshold))
                plt.plot(seq,last_score,'o',color = line[0].get_color())
            else:
                plt.plot(last_score,'o',label = 'mean')
        
        
        plt.xlabel('Time', fontsize = 14)
        plt.ylabel('Score', fontsize = 14)
        plt.legend()
        plt.grid()
        
        xticks = np.arange(len(scores))
        xtickslabel = list(np.arange(len(scores))) + ['mean'] if len(scores) > 1 else ['mean']
        
        plt.xticks(xticks,xtickslabel,fontsize = 14)
        plt.yticks(fontsize = 14)
        
        plt.title(title,fontsize = 20)
        
        if title != 'BIAS':
            plt.ylim((0,1))
        
        plt.show()    

    
    def plot_one_threshold_scores_curve(self,scores, title = None):
        '''
        func: 画出某个metric下的单个阈值下的scores变化
        Parameter
        ---------
        scores: list or array 
            if array: 一维数组
        title: None or str
            图的title 
            when None. title = self.metric
        '''
        
        if title == None:
            title = self.metric
        
        f1 = plt.figure(figsize = (10,7))
        
        seq_scores = scores[0:-1] 
        last_score = scores[-1] 
        
        if len(scores) > 1:
            seq = len(scores) - 1
            line = plt.plot(np.arange(seq),seq_scores,'-*',label = 'seq')
            plt.plot(seq,last_score,'o',color = line[0].get_color(),label = 'mean')
        else:
            plt.plot(last_score,'o',label = 'mean')
        
        plt.xlabel('Time', fontsize = 14)
        plt.ylabel('Score', fontsize = 14)
        plt.legend()
        plt.grid()
        
        xticks = np.arange(len(scores))
        xtickslabel = list(np.arange(len(scores))) + ['mean'] if len(scores) > 1 else ['mean']
        
        plt.xticks(xticks,xtickslabel,fontsize = 14)
        plt.yticks(fontsize = 14)
        
        if title not in ['Multi_HSS','MSE','MAE','SSIM']:
            plt.title(title + ':' + str(self.threshold),fontsize = 20)
        else: 
            plt.title(title, fontsize = 20)
            
        if title not in ['MSE','MAE','BIAS']:
            plt.ylim((0,1))
        
        plt.show()
        

    def plot_which_index(self,which_batch_plot=[0,1,2], show_metric=False, metric = None, 
                         threshold = None, save_path = None
                         ):
        '''
        func: 画出某个batch下的y_true 和 y_pre的比对,同时可以计算metric得分;
        Parameter
        ---------
        which_batch_plot: list
            default [0,1,2],Dataloader中的所有iterter中，哪些iter的输入和预测是需要被可视化对比的
        show_metric: bool
            是否在输出对比图的同时，也输出metric，即对应的评价信息
        metric: None or str
            当show_metrci为True时才使用
            when None. 则默认使用 self.metric
            when str,可选['Multi_HSS','MSE','MAE','SSIM','TS','MAR','FAR','POD','BIAS','F1','ACC','Single_HSS']
        threshold: None or int or list
            when show_metric = True, 且metric in ['TS','MAR','FAR','POD','BIAS','F1','ACC','Single_HSS']时,才使用
            when None.则默认使用 self.thrshold
            when int: 即计算哪个阈值的metric得分；
            when list, eg: [20,30,40],则同时计算这3个阈值下的评分
        save_path: None or str
            default None.则不保存obs 和 pre的比较图, 
            when str. 则为雷达图的保存位置. eg: '/mnt/nvme1/fzl/radar_img'
        '''
        max_batch_index = max(which_batch_plot)
        
        for i, out in enumerate(self.dataloader_test):
            test_x = out[0]
            test_y = out[1]
            if self.y_mask:
                target = test_y[0].to(self.device)
                mask = test_y[1].to(self.device)
                target = target * mask
            else:
                target = test_y.to(self.device)

            if i in which_batch_plot:
                pre_y = self.predict(test_x)
                save_path_batch = os.path.join(save_path, str(i)) if save_path else None
                if save_path_batch is not None:
                    os.makedirs(save_path_batch, exist_ok=True)
                self.plot_compare_obs_pre(target, pre_y,scale = 75,save_path = save_path_batch)
                
                if show_metric:
                    
                    if metric == None:
                        metric = self.metric
                    if threshold == None:
                        threshold = self.threshold
                        
                    if isinstance(threshold, list):                        
                        more_thresholds_score = self.one_batch_more_thresholds_metric(input_y, pre_y, metric = metric,thresholds = threshold)                             
                        self.plot_more_threshold_scores_curve(more_thresholds_score, title = metric,thresholds = threshold)
                        
                    else:
                        score = self.one_batch_metric(input_y,pre_y, metric = metric, threshold = threshold, if_metric = True)
                        self.plot_one_threshold_scores_curve(scores = score, title = metric)
                        print('index:{}-- metric:{}, threshold:{} ----- score:{}'.format(i,metric,threshold,score))  
         
                print()
                
            if i >= max_batch_index:
                break
            
        return test_x, target, pre_y
    

    def plot_which_index_DL_SMS_EC(self,which_batch_plot=[0,1,2], show_metric=False, metric=None, 
                                   threshold=None, save_path = None,
                                   levels=[0,0.1,1,2,5,10,15,20,25,30,50,90],
                                   metric_compare=['DL', 'SMS', 'EC'],
                                    ):
        '''
        func: 画出指定batch下的['OBS', 'DL', 'SMS', 'EC']的降水分布
        Parameter
        ---------
        which_batch_plot: list
            default [0,1,2],Dataloader中的所有iterter中，哪些iter的输入和预测是需要被可视化对比的
        show_metric: bool
            是否在输出对比图的同时，也输出metric，即对应的评价信息
        metric: None or str
            当show_metrci为True时才使用
            when None. 则默认使用 self.metric
            when str,可选['Multi_HSS','MSE','MAE','SSIM','TS','MAR','FAR','POD','BIAS','F1','ACC','Single_HSS']
        threshold: None or int or list
            when show_metric = True, 且metric in ['TS','MAR','FAR','POD','BIAS','F1','ACC','Single_HSS']时,才使用
            when None.则默认使用 self.thrshold
            when int: 即计算哪个阈值的metric得分；
            when list, eg: [20,30,40],则同时计算这3个阈值下的评分
        save_path: None or str
            default None.则不保存图片
            when str. 则为图的保存位置. eg: '/mnt/nvme1/fzl/radar_img'
        metric_compare: str
            除了['OBS']外，还画出和metric哪些数据，可选为['DL', 'SMS', 'EC'].
        '''
        max_batch_index = max(which_batch_plot)
        
        for i, out in enumerate(self.dataloader_test):
            test_x = out[0]
            test_y = out[1]
            SMS_r3, EC_r3 = out[2]
            if self.y_mask:
                target = test_y[0].to(self.device)
                mask = test_y[1].to(self.device)
                target = target * mask
            else:
                target = test_y.to(self.device)

            if i in which_batch_plot:
                pre_y = self.predict(test_x)
                save_path_batch = os.path.join(save_path, str(i)) if save_path else None
                if save_path_batch is not None:
                    os.makedirs(save_path_batch, exist_ok=True)
                
                four_titles = ['OBS', 'DL', 'SMS', 'EC']
                four_data = [target, pre_y, SMS_r3, EC_r3]
                for data, title in zip(four_data, four_titles):
                    if title not in metric_compare + ['OBS']:
                        continue 
                    data = data.cpu().detach().numpy()
                    data = data[0,0] * self.scale
                    save_filepath=os.path.join(save_path_batch, f'{title}.png') if save_path_batch is not None else None
                    Radar_utils.plot_rain_single(data,
                                                title=title,
                                                levels=levels,
                                                save_filepath=save_filepath
                                                )

                # self.plot_compare_obs_pre(target, pre_y, scale=self.scale, save_path=save_path_batch)

                if show_metric:
                    if metric == None:
                        metric = self.metric
                    if threshold == None:
                        threshold = self.threshold
                    if isinstance(threshold, list):
                        for title in metric_compare:
                            title_index = four_titles.index(title)
                            data = four_data[title_index]
                            more_thresholds_score = self.one_batch_more_thresholds_metric(target, data, metric = metric,thresholds = threshold)                             
                            self.plot_more_threshold_scores_curve(more_thresholds_score, title=f'{title}_{metric}', thresholds=threshold)
                        
                    else:
                        for title in metric_compare:
                            title_index = four_titles.index(title)
                            data = four_data[title_index]
                            score = self.one_batch_metric(target, data, metric = metric, threshold = threshold, if_metric = True)
                            self.plot_one_threshold_scores_curve(scores=score, title=f'{title}_{metric}')
                            print('index:{}-- metric:{}, threshold:{} ----- score:{}'.format(i,metric,threshold,score))  
         
                print()
                
            if i >= max_batch_index:
                break
            
        return test_x, target, pre_y, SMS_r3, EC_r3

    
    def plot_input_x(self,input_x,
                     save_path=None,
                     x_angle_list=['d_15','05','15','24','34','43','60'],
                     ):
        '''
        func: 画出某个iterration下的模型的输入: input_x
        Returns
        -------
        input_x: 4D or 5D Tensor 
            模型的输入信息
        save_path: str
            画出的雷达图的保存路径，eg: 'D:/radar/input/img'
        x_angle_list: list
            输入的x的仰角信息。default: ['d_15','05','15','24','34','43','60']
            可选 ['d_15','05','15','24','34','43','60']. 默认这些仰角都取. 默认保留去噪的15仰角雷达观测
        '''
        input_x = input_x.cpu().detach().numpy()
        angle_num = len(x_angle_list)
        
        input_x_list = []
    
        #单仰角输入
        if angle_num == 1 and len(input_x.shape()) == 4:
            input_x_list.append(input_x)
        
        #如果存在多仰角输入，
        if angle_num > 1:
            print(input_x.shape)
            if len(input_x.shape) == 5: 
                
                batch, angles,channels, height, width = input_x.shape
                print(angles,angle_num)
                assert angles == angle_num
                
                for i in range(angle_num):
                    input_x_list.append(input_x[:,i])
                    
            elif len(input_x.shape) == 4:
                
                batch, channels, height, width = input_x.shape
                
                #确定单个仰角的输入channel数,
                angle_seq = channels // angle_num
                
                for i in range(angle_num):
                    single_x = input_x[:,i*angle_seq: (i + 1)* angle_seq, :,:]
                    input_x_list.append(single_x)
                    
        
        #确定每个仰角输入几帧,即输入历史的几个时刻的雷达图
        input_seq = input_x_list[0].shape[1]
        
        #画出历史时次的多个仰角的雷达图
        for t in range(input_seq):
            
            t_title = 'T-' + str(input_seq - t - 1) 
    
            for angle, x in zip(x_angle_list,input_x_list):
                
                save_filepath = os.path.join(save_path, angle + '_' + t_title + '.png')
                Radar_utils.plot_single_radar_img(x[0,t,:,:]*self.scale,
                                                  title = angle + '_' + t_title,
                                                  save_filepath = save_filepath)
        
        return None
            
                
    def predict(self, test_x):
        '''
        func: 只迭代预测一次，不做参数更新
        parameters
        ---------------
        test_x: np.array
            输入的测试的样本
        ---------------
        return
            tensor. 模型对test_x 的预测结果
        '''
        self.model.eval()
        
        if isinstance(test_x,list):
            test_x = [x.float().to(self.device) for x in test_x]
        else:
            test_x = test_x.float().to(self.device)

        pre_y = None
        with torch.no_grad():
            pre_y = self.model(test_x)
                        
        return pre_y
    

    def dataloader_metric(self, metric=None, scale=None, threshold=None, if_metric=False):
        '''
        func: 对整个Dataloader进行综合评分
        Parameter
        ---------
        metric: str or None
            评价指标,可选用['Multi_HSS','MSE','MAE','SSIM','TS','MAR','FAR','POD','BIAS','F1','ACC','Single_HSS']
            when None, 则用默认的使用 self.metric
        scale str or None
            当metric in ['MSE','MAE','SSIM']时，scale必须等于1
            其中Multi_HSS: 分为多个DBZ阈值综合考虑obs 和 pre之间的相似度; scale = 75
                Single_HSS:只考虑单个阈值下，obs 和 pre的相似度, scale = 75
            当metric in ['TS','MAR','FAR','POD','BIAS','F1','ACC']时，scale = 75
        threshold: int or None
            when None, 则 threshold = self.threshold    
        if_metric: bool
            default True.当metric in  ['TS','MAR','FAR','POD','BIAS','F1','ACC','Single_HSS'] 时，是否返回整个数据集上对应的score.
            when False.不返回这个all_batch的得分，只是返回对应的整个数据集上的[TN, FN, FP, TP],即[correctnegatives, misses, falsealarms, hits] 
            when True. 则返回这个batch下的对应的metric的得分            
        '''
        if metric is None:
            metric = self.metric
        if scale is None:
            scale = self.scale
        if threshold is None:
            threshold = self.threshold        
        all_scores = []
        for i, out in enumerate(self.dataloader_test):
            test_x = out[0]
            test_y = out[1]
            pre_y = self.predict(test_x)
            if self.y_mask:
                target = test_y[0].to(self.device)
                mask = test_y[1].to(self.device)
                target = target * mask
            else:
                target = test_y.to(self.device)
            scores = self.one_batch_metric(target, pre_y, 
                                           metric=metric, 
                                           scale=scale, 
                                           threshold=threshold,
                                           if_metric=False)
            
            all_scores.append(scores)

        if metric in ['Multi_HSS','MSE','MAE','SSIM']:
            score_mean = self.score_mean_with_frame(all_scores)
            return score_mean
        
        elif metric in ['TS','MAR','FAR','POD','BIAS','F1','ACC','Single_HSS']:
            
            sample_nums = len(all_scores) 
            seq = len(all_scores[0])
            
            #eg: all_scores = [[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
                                #[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]]
                                
            all_scores = np.array(all_scores) #shape --> (sample_nums,seq,4)
            all_scores = np.sum(all_scores,axis = 0) #对所有样本对应的位置的上的值累加 --> shape = (seq,4)
            
            #[TN,FN,FP,TP]
            TN = all_scores[:,0] #shape = (seq,)
            FN = all_scores[:,1]
            FP = all_scores[:,2]
            TP = all_scores[:,3]
            
            if not if_metric:  
                return [TN, FN,FP, TP]
            else:
                if metric == 'TS':
                    score = TP/(TP + FN + FP) 
                elif metric == 'MAR':
                    score = FN / (TP + FN)
                elif metric == 'FAR':
                    score = FP / (TP + FP)
                elif metric == 'POD':
                    score = TP / (TP + FN)
                elif metric == 'BIAS':
                    score = (TP + FP) / (TP + FN)
                elif metric == 'F1':
                    rec_score = TP / (TP + FN)
                    pre_score = TP / (TP + FP)
                    score = 2 * ((pre_score * rec_score) / (pre_score + rec_score))
                elif metric == 'ACC':
                    score = (TP + TN) / (TP + TN + FP + FN) 
                elif metric == 'Single_HSS':
                    HSS_num = 2 * (TP * TN - FN * FN)
                    HSS_den = (FN**2 + FP**2 + 2*TP*TN + (FN + FP)*(TP + TN))
                    score = HSS_num/ HSS_den
                    
                return list(score)


    def score_mean_with_frame(self, scores, plot = False):
        '''
        func: 得到预测的n_frames帧的逐帧的评分
        Parameter:
        ----------
        scores: list --> 多个子list组成的list
            len为整个dataloader的iteration数量,每个子list为[seq帧的得分,所有seq帧的综合得分],len为 seq + 1
        plot: bool
            是否画出得分情况，默认画出
        '''
        pd_scores = pd.DataFrame(scores)
        score_mean = pd_scores.mean(axis = 0).values
        score_mean = list(score_mean)
        
        if plot:
            self.plot_scores_curve(score_mean)
        return score_mean

    def save_all_metric(self, metric_save_path, 
                        thresholds=[0.1,1,5,10,15,20,25,30,35,40,45,50,55,60],
                        all_metrics = ['TS','MAR','FAR','POD',
                                       'BIAS','F1','ACC','Single_HSS',
                                       'SSIM','MSE','MAE',
                                       'Multi_HSS'],
                        ):
        '''
        func: 将模型在指定评价指标上的评分和图保存到save_path + '/metric'路径下
            文件名用str(metric)表示
        Parameter
        ---------
        save_path: str
            评估结果保存路径
        all_metrics: list
            default ['TS','MAR','FAR','POD','BIAS','F1','ACC','Single_HSS','SSIM','MSE','MAE', 'Multi_HSS']
            即使用哪些评价指标进行评估                
        '''
        os.makedirs(metric_save_path, exist_ok=True)

        metrics_1 = ['TS','MAR','FAR','POD','BIAS','F1','ACC','Single_HSS']
        metrics_2 = ['MSE','MAE','Multi_HSS','SSIM']

        # 先获取所有阈值下的对应的TN,FN,FP,TP
        pd_threshold = pd.DataFrame(index=thresholds, columns=['TN', 'FN', 'FP', 'TP'])

        all_threshold_metric1 = []
        for i, threshold in enumerate(thresholds):
            TN, FN, FP, TP = self.dataloader_metric(metric='TS', scale=self.scale, threshold=threshold, if_metric=False)
            pd_threshold.iloc[i] = [TN[0], FN[0], FP[0], TP[0]] 
            print('threshold:{}'.format(threshold))
            print('TN seq:{}'.format(TN))
            print('FN seq:{}'.format(FN))
            print('FP seq:{}'.format(FP))
            print('TP seq:{}'.format(TP))
            print()
            all_threshold_metric1.append([TN,FN,FP,TP])

        print(f'all threshold binary: {pd_threshold}')
        pd_threshold.to_csv(os.path.join(metric_save_path, 'binary_TN_FN_FP_TP.csv'))

        for metric in all_metrics[0:]:
            if metric in metrics_1:
                print(metric)
                f2 = plt.figure(figsize=(10,6))
                all_scores = []
                for index,threshold in enumerate(thresholds):
                    TN, FN, FP, TP = all_threshold_metric1[index]
                    if metric == 'TS':
                        mean_scores = TP/(TP + FN + FP) 
                    elif metric == 'MAR':
                        mean_scores = FN / (TP + FN)
                    elif metric == 'FAR':
                        mean_scores = FP / (TP + FP)
                    elif metric == 'POD':
                        mean_scores = TP / (TP + FN)
                    elif metric == 'BIAS':
                        mean_scores = (TP + FP) / (TP + FN)
                    elif metric == 'F1':
                        rec_score = TP / (TP + FN)
                        pre_score = TP / (TP + FP)
                        mean_scores = 2 * ((pre_score * rec_score) / (pre_score + rec_score))
                    elif metric == 'ACC':
                        mean_scores = (TP + TN) / (TP + TN + FP + FN) 
                    elif metric == 'Single_HSS':
                        HSS_num = 2 * (TP * TN - FN * FN)
                        HSS_den = (FN**2 + FP**2 + 2*TP*TN + (FN + FP)*(TP + TN))
                        mean_scores = HSS_num/ HSS_den
                    all_scores.append(list(mean_scores))
                    
                    seq = len(mean_scores) 
                    if seq > 1:
                        seq = seq - 1
                        
                    if seq > 1:
                        #保持两者颜色一致
                        line = plt.plot(np.arange(seq),mean_scores[0:seq],'-*',label = str(threshold))
                        plt.plot(seq,mean_scores[seq],'o',color = line[0].get_color())
                    elif seq == 1:
                        plt.plot(mean_scores,'o',label = str(threshold))

                plt.xlabel('Time', fontsize = 14)
                plt.ylabel('Score', fontsize = 14)
                xticks = np.arange(len(mean_scores))
                
                xtickslabel = list(np.arange(seq)) + ['mean'] if seq > 1 else ['mean']
                    
                plt.xticks(xticks,xtickslabel,fontsize = 16)
                plt.yticks(fontsize = 16)
                if metric != 'BIAS':
                    plt.ylim((0,1))
                plt.legend()
            
                plt.title(metric,fontsize = 16)
                plt.grid()
                plt.savefig(os.path.join(metric_save_path,metric + '.png'), dpi = 200)
                plt.show()
                
                columns = list(range(seq)) + ['mean'] if seq > 1 else ['mean']
                pd_scores = pd.DataFrame(all_scores,index = thresholds, columns = columns)
                pd_scores.to_csv(os.path.join(metric_save_path,metric + '.csv'))
                    
                f2.clf()

            elif metric in metrics_2:
                print(metric)
                if metric in ['MSE','MAE','SSIM']:
                    scale = 1
                else:
                    scale = self.scale
                TN, FN,FP, TP = self.dataloader_metric(metric ='TS', scale=self.scale, threshold = threshold, if_metric = False)

                seq = len(mean_scores) 
                if seq > 1:
                    seq = seq - 1
                
                f2 = plt.figure(figsize=(10,6))
                
                if seq > 1:
                    line = plt.plot(np.arange(seq),mean_scores[0:seq],'-*',label = metric)
                    plt.plot(seq,mean_scores[seq],'o',color = line[0].get_color())
                elif seq == 1:
                    plt.plot(mean_scores,'o',label = metric)
                
                plt.xlabel('Time', fontsize = 14)
                plt.ylabel('Score', fontsize = 14)
                
                xticks = np.arange(len(mean_scores))
                xtickslabel = list(np.arange(seq)) + ['mean'] if seq > 1 else ['mean']
                plt.xticks(xticks,xtickslabel,fontsize = 16)
                plt.yticks(fontsize = 16)
                plt.legend()
                plt.grid()
                
                if metric in ['Multi_HSS','SSIM']:
                    plt.ylim((0,1))
                
                plt.title(metric,fontsize = 16)
                plt.savefig(os.path.join(metric_save_path,metric + '.png'),dpi = 200)
                plt.show()
                
                columns = list(range(seq)) + ['mean'] if seq > 1 else ['mean']
                pd_scores = pd.DataFrame(np.array(mean_scores).reshape(1,-1),columns = columns)
                pd_scores.to_csv(os.path.join(metric_save_path, metric + '.csv'))   
            
                f2.clf()

        return None
