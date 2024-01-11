import os
import json
import time
import torch

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model.Signal_Token_transformer import STT
from model.classifier import Projector, Predictor, ReconstructionHead

from utils.saver import Saver
from utils.helper import make_description
from utils.summary import TensorboardSummary
from utils.multi_task_loss import AutomaticWeightedLoss
from utils.schedulers import WarmupStepLRSchedule, WarmupConstantSchedule, WarmupExponetialSchedule

from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix


class SWISS_trainer(object):
    def __init__(self, args, pretrain_loader, save_path: str):

        self.args = args
        
        # Define save directory
        self.save_path = os.path.join(save_path, f'/pretrain')
        self.check_path = os.path.join(self.save_path, 'pretrain_ckpt')
        self.summary_path = os.path.join(self.save_path, 'pretrain_runs')

        # Define Dataloader
        self.pretrain_loader = pretrain_loader
        
        # Define saver
        self.saver = Saver(self.check_path)

        # Denfine Tensorboard Summary
        self.summary = TensorboardSummary(self.summary_path)
        self.writer = self.summary.create_summary()

        # Define Transformer online network and target network
        self.online_encoder = STT(args).to(args.cuda)
        self.online_projector = Projector(args).to(args.cuda)
        self.online_predictor = Predictor(args).to(args.cuda)
        
        self.target_encoder = STT(args).to(args.cuda)
        self.target_projector = Projector(args).to(args.cuda)
        self.set_required_grid(self.target_encoder, False)
        self.set_required_grid(self.target_projector, False)
        
        self.recon_head = ReconstructionHead(args.num_signals, args.emb_dim, args.feature_dim).to(args.cuda)
        
        self.m = args.m
        
        self.criterion = nn.MSELoss().to(args.cuda)
        
        # Learnable loss parameter
        self.auto_loss = AutomaticWeightedLoss().to(args.cuda)
        # Define optimizer & Criterion
        optim_params = [{'params': list(self.online_encoder.parameters())+list(self.online_projector.parameters())\
                        +list(self.online_predictor.parameters())+list(self.recon_head.parameters())},
                        {'params': self.auto_loss.parameters(), 'weight_decay':0}]
       
        if args.pre_optimizer == 'adam':
            self.optimizer = optim.Adam(optim_params, lr=args.pre_lr)
        
        elif args.pre_optimizer == 'adamw':
            self.optimizer = optim.AdamW(optim_params, lr=args.pre_lr)
        
        if args.pre_scheduler == 'constant':
            self.scheduler = WarmupConstantSchedule(self.optimizer, warmup_steps=args.pre_warm_up)
            
        elif args.pre_scheduler == 'exp':
            self.scheduler = WarmupExponetialSchedule(self.optimizer, warmup_steps=args.pre_warm_up, gamma=0.95)
        
        elif args.pre_scheduler == 'step':
            self.scheduler = WarmupStepLRSchedule(self.optimizer, warmup_steps=args.pre_warm_up, gamma=0.8)

        args.online_encoder_parameters = self.count_parameters(self.online_encoder, val=True)
        print(f'Number of online Network Parameters : {args.online_encoder_parameters}')
        
        # Save argparser
        with open(os.path.join(self.save_path, 'arg_parser.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        
    def set_required_grid(self, model, val):
        for p in model.parameters():
            p.requires_grad = val
        
    @torch.no_grad()
    def _update_target_encoder_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            
    @torch.no_grad()
    def _update_target_projector_parameters(self):
        """
        Momentum update of the key projector
        """
        for param_q, param_k in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            
    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
        for param_q, param_k in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
    def feature_level_loss_fn(self, x, y):
        # x : B, proj_dim
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        
        return 2 - 2 * (x * y).sum(dim=-1)
    
    def data_level_loss_fn(self, x, y):
        # x : B, S, T
        loss = self.criterion(x,y)
            
        return loss
    
    def adjust_mm(self, step):
        self.m = 1 - (1 - self.args.m) * (np.cos(np.pi * step / self.args.pre_epochs) + 1) / 2
        
    def run(self, epochs):
        loss_dict = {'total_loss':[],'feature_level_loss':[],'data_level_loss':[]}
        
        # initialize target network
        self.initializes_target_network()
        
        for epoch in range(1, epochs+1):
            start_time = time.time()
            # pretraining
            train_history = self.train(self.pretrain_loader, current_epoch=epoch)
            end_time = round(time.time()-start_time, 2)
            # epoch_history (reconstruction loss)
            epoch_history = {
                'Loss': {
                    'total_loss': train_history.get('total_loss'),
                    'feature_level_loss': train_history.get('feature_level_loss'),
                    'data_level_loss': train_history.get('data_level_loss')
                    }
                }
            
            for k in loss_dict:
                loss_dict[k].append(train_history[k])
                
            print('\n')
            make_description(['K-fold','Dataset'], self.args.exp_fold, self.args.dataset)
            make_description(['Epoch', 'Required time', 'w1', 'w2'], epoch, end_time, 
                                round((0.5/self.auto_loss.params[0].item()**2),4), round((0.5/self.auto_loss.params[1].item()**2),4))
                
            make_description( ['Total Loss', 'Feature Level', 'Data Level'],
                             round(train_history["total_loss"],4), 
                             round(train_history["feature_level_loss"],4),
                             round(train_history["data_level_loss"],4))
            
            # Tensorboard summary
            for metric_name, metric_dict in epoch_history.items():
                self.writer.add_scalars(
                    main_tag=metric_name,
                    tag_scalar_dict=metric_dict,
                    global_step=epoch
                )

        self.saver.checkpoint(f'target_encoder', self.target_encoder, is_best = False)
        self.saver.checkpoint(f'online_encoder', self.online_encoder, is_best = False)
        self.saver.checkpoint(f'recon_head', self.recon_head, is_best = False)
        
        return epoch_history

    def train(self, pretrain_loader, current_epoch: int):
        self.online_encoder.train()
        self.online_projector.train()
        self.online_predictor.train()
        self.recon_head.train()

        feature_level_loss, data_level_loss, total_scalar_loss = .0, .0, .0
        
        if self.args.adjust_m:
            self.adjust_mm(current_epoch-1)
        
        for i, (crt_x, _, masks, _) in enumerate(pretrain_loader):
            """
            x_1 & x_2 : (B, S, L)
            masks : (B, S, L)
            subjects : (B,)"""
            b, _, _ = crt_x.shape
            
            x = crt_x.to(device=self.args.cuda)
            masks = masks.to(device=self.args.cuda)
            masked_x = x.masked_fill(masks, value=0)

            # features : {'enc_out':enc_out, 'mean_out':mean_out}
            features = self.online_encoder(masked_x)
            online_enc_out = features['enc_out'] # shape : batch, signals, emb_dim
            global_pred = self.online_predictor(self.online_projector(online_enc_out.view(b,-1))) # shape : batch, proj_dim
            
            recon_out = self.recon_head(online_enc_out) # recon_out : batch, signals, feature_dim
            
            with torch.no_grad():
                features = self.target_encoder(x)
                target_enc_out = features['enc_out'] # shape : batch, signals, emb_dim
                global_gt = self.target_projector(target_enc_out.view(b,-1)) # shape : batch, proj_dim

            # Calculate feature level reconstruction Loss
            feature_level_loss_sum = self.feature_level_loss_fn(global_pred, global_gt)
            feature_level_loss = feature_level_loss_sum.mean()
            feature_level_loss += feature_level_loss.item()
            
            # Calculate signal level reconstruction Loss
            data_level_loss = self.data_level_loss_fn(recon_out, x)
            data_level_loss += data_level_loss.item()
            
            total_loss = self.auto_loss(feature_level_loss, data_level_loss)

            total_scalar_loss += total_loss.item()
            
            # Update online network
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Update target network
            self._update_target_encoder_parameters()
            self._update_target_projector_parameters()
            
        self.scheduler.step()
        feature_level_loss /= (i+1)
        data_level_loss /= (i+1)
        total_scalar_loss /= (i+1)
        
        total_train_history = {
            'total_loss' : total_scalar_loss,
            'feature_level_loss' : feature_level_loss,
            'data_level_loss' : data_level_loss
            }

        return total_train_history

    def matrics(self, y_true, y_pred):
        acc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        confusion = pd.DataFrame(confusion_matrix(y_true, y_pred))

        return acc, f1, confusion
    
    def count_parameters(self, model, val):
        return sum(p.numel() for p in model.parameters() if p.requires_grad == val)