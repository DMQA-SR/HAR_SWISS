import os
import json
import time
import torch

import numpy as np
import torch.optim as optim
import torch.nn as nn

from model.classifier import MLPHead
from model.Signal_Token_transformer import STT

from utils.saver import Saver
from utils.schedulers import WarmupStepLRSchedule, WarmupConstantSchedule, WarmupExponetialSchedule


from utils.summary import TensorboardSummary
from sklearn.metrics import balanced_accuracy_score, f1_score


class HAR_trainer(object):
    def __init__(self, args, train_loader, valid_loader, save_path):

        self.args = args
        self.args = args
        
        # Define save directory
        self.check_path = os.path.join(save_path, f'/ckpt')
        self.summary_path = os.path.join(save_path, f'/runs')
        pretrained_path = os.path.join(save_path, f'/pretrain/pretrain_ckpt')
        
        # Define Dataloader
        self.train_loader, self.valid_loader = train_loader, valid_loader
        
        # Define saver
        self.saver = Saver(self.check_path)

        # Denfine Tensorboard Summary
        self.summary = TensorboardSummary(self.summary_path)
        self.writer = self.summary.create_summary()

        # Define model
        encoder = STT(args)
        
        # load pretrained model and freeze for linear evaluations
        enc_weight = torch.load(os.path.join(pretrained_path, f'online_encoder.pt'))
        encoder.load_state_dict(enc_weight)
        for param in encoder.parameters():
            param.requires_grad = False
        
        classifier = MLPHead(args, args.emb_dim*args.num_signals, args.num_classes)
        
        self.encoder = encoder.to(args.cuda)
        self.classifier = classifier.to(args.cuda)
        
        args.encoder_parameters = self.count_parameters(self.encoder, grad=False)
        args.mlp_parameters = self.count_parameters(self.classifier, grad=True)
        print(f'Number of Freezed Parameters(Transformer) : {args.encoder_parameters}')
        print(f'Number of Trainable Parameters(MLP) : {args.mlp_parameters}')

        # Save argparser
        with open(os.path.join(save_path, f'{args.crt_fold}_fold/arg_parser.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        # Define optimizer & Criterion
        optim_params = self.classifier.parameters()
       
        if args.optimizer == 'adam':
            self.optimizer = optim.Adam(optim_params, lr=args.lr)
        
        elif args.optimizer == 'adamw':
            self.optimizer = optim.AdamW(optim_params, lr=args.lr,)
            
        elif args.optimizer == 'SGD':
            self.optimizer = optim.SGD(optim_params, lr=args.lr)
        
        if args.scheduler == 'constant':
            self.scheduler = WarmupConstantSchedule(self.optimizer, warmup_steps=args.warm_up)
            
        elif args.scheduler == 'exp':
            self.scheduler = WarmupExponetialSchedule(self.optimizer, warmup_steps=args.warm_up, gamma=0.9)
        
        elif args.scheduler == 'step':
            self.scheduler = WarmupStepLRSchedule(self.optimizer, warmup_steps=args.warm_up, gamma=0.8)
        
        self.criterion = nn.CrossEntropyLoss().to(self.args.cuda)

    def run(self, epochs):
        train_loss_list, valid_loss_list = [], []
        for epoch in range(1, epochs+1):
            start_time = time.time()
            # Train & Validation
            train_history = self.train(self.train_loader, current_epoch=epoch)
            valid_history = self.validation(self.valid_loader, current_epoch=epoch)
            end_time = round(time.time()-start_time, 2)
            
            # epoch_history (loss & matrics)
            epoch_history = {
                'Loss': {
                    'train': train_history.get('train_loss'),
                    'valid': valid_history.get('valid_loss'),
                },
                'Accuracy': {
                    'train': train_history.get('train_acc'),
                    'valid': valid_history.get('valid_acc'),
                },
                'F1_Score': {
                    'train': train_history.get('train_f1'),
                    'valid': valid_history.get('valid_f1'),
                }
            }
            
            print(f'\nK-Fold : {self.args.exp_fold} | Dataset : {self.args.dataset} | Seed : {self.args.seed}')
            print(f'Epoch : {epoch}\t | \t Required time : {end_time}s')
            print(f'Train Loss     : {train_history["train_loss"]:.4f}\t | \tTrain F1-Score     : {train_history["train_f1"]:2.4f}')
            print(f'Valid Loss     : {valid_history["valid_loss"]:.4f}\t | \tValid F1-Score     : {valid_history["valid_f1"]:2.4f}')
            
            # Append loss
            train_loss_list.append(epoch_history['Loss']['train'])
            valid_loss_list.append(epoch_history['Loss']['valid'])
            
            # Tensorboard summary
            for metric_name, metric_dict in epoch_history.items():
                self.writer.add_scalars(
                    main_tag=metric_name,
                    tag_scalar_dict=metric_dict,
                    global_step=epoch
                )

        # Save last epoch weights
        self.saver.checkpoint(f'classifier', self.classifier, is_best = False)
        self.saver.checkpoint(f'encoder', self.encoder, is_best = False)
        
        return epoch_history

    def train(self, train_loader):
        self.encoder.train()
        self.classifier.train()

        train_loss = .0
        y_true_list, y_pred_list = [], []
       
        for i, (crt_x, crt_y) in enumerate(train_loader):
            """
            x : (batch, # of sensors, feature dim)
            y : (batch,)"""

            x = crt_x.to(device=self.args.cuda)
            y = crt_y.to(device=self.args.cuda, dtype=torch.int64)

            self.optimizer.zero_grad()

            # features : {'enc_out':enc_out, 'mean_out':mean_out}
            features = self.encoder(x)
            mlp_input = features['enc_out'].view(x.shape[0], -1) # shape : batch, signals*emb_dim

            pred = self.classifier(mlp_input)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            # Calculate metric
            y_pred = torch.argmax(pred, dim=-1).detach().cpu().numpy() 
            y_true = y.detach().cpu().numpy()
            
            y_true_list += list(y_true)
            y_pred_list += list(y_pred)

        self.scheduler.step()
        
        train_loss /= (i+1)
        train_acc, train_f1 = self.matrics(y_pred_list, y_true_list)
        total_train_history = {
            'train_loss' : train_loss,
            'train_acc' : train_acc,
            'train_f1' : train_f1
            }

        return total_train_history

    @torch.no_grad()
    def validation(self, valid_loader):
        self.encoder.eval()
        self.classifier.eval()

        valid_loss = .0
        y_true_list, y_pred_list = [], []
        
        for i, (crt_x, crt_y) in enumerate(valid_loader):
            """
            x : (batch, # of sensors, feature dim)
            y : (batch,)"""

            x = crt_x.to(device=self.args.cuda)
            y = crt_y.to(device=self.args.cuda, dtype=torch.int64)

            # features : {'enc_out':enc_out, 'mean_out':mean_out}
            features = self.encoder(x)
            dec_input = features['enc_out'].view(x.shape[0], -1) # shape : batch, signals*emb_dim

            pred = self.classifier(dec_input)
            loss = self.criterion(pred, y)

            valid_loss += loss.item()

            # Calculate metric
            y_pred = torch.argmax(pred, dim=-1).detach().cpu().numpy() 
            y_true = y.detach().cpu().numpy()
            
            y_true_list += list(y_true)
            y_pred_list += list(y_pred)

        valid_loss /= (i+1)
        valid_acc, valid_f1 = self.matrics(y_pred_list, y_true_list)

        total_valid_history = {
            'valid_loss' : valid_loss,
            'valid_acc' : valid_acc,
            'valid_f1' : valid_f1
            }

        return total_valid_history

    def matrics(self, y_true, y_pred):
        acc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')

        return acc, f1
    
    def count_parameters(self, model, grad = True):
        return sum(p.numel() for p in model.parameters() if p.requires_grad == grad)
