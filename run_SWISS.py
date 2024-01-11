import os
import torch
import random
import pickle
import argparse
import numpy as np
from sklearn.model_selection import KFold

from tasks.Finetuning_tester import HAR_tester
from tasks.Finetuning_trainer import HAR_trainer
from tasks.Pretraining_trainer import SWISS_trainer

from dataloaders import make_data_loader

import warnings
warnings.filterwarnings(action='ignore')

def main():
    
    parser = argparse.ArgumentParser(description='SWISS argparser')
    
    # Settings
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--cuda', type = str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='usc_had', 
                        choices=['uci_har','mobiact','motion_sense','usc_had'])
    parser.add_argument('--k-fold', type=int, default=5)
    parser.add_argument('--exp-fold', type=int, default=1)
    parser.add_argument('--feature_dim', type=int, default=33)
    
    
    """Pretraining parsers"""
    # Pretrain optimizer parameters
    parser.add_argument('--pre-optimizer', type=str, default='adam')
    parser.add_argument('--pre-lr', type=float, default=3e-4)
    parser.add_argument('--pre-weight_decay', type=float, default=5e-4)
    parser.add_argument('--pre-scheduler', type=str, default='step', choices=['exp','step','constant'])
    parser.add_argument('--pre-warm-up', type=int, default=25)
    parser.add_argument('--pre-step-period', type=int, default=25)
    parser.add_argument('--pre-lr-decay', type=float, default=.8)

    # pre training hyperparameters
    parser.add_argument('--pre-epochs', type=int, default=150)
    parser.add_argument('--pre-batch_size', type=int, default=512)
    parser.add_argument('--adjust_m', action='store', default=False)
    parser.add_argument('--m', type=float, default=0.9)
    parser.add_argument('--proj_hiddim', type=int, default=512)
    parser.add_argument('--proj_dim', type=int, default=256)
    
    # Masking parameters
    parser.add_argument('--mask-type', type=str, default='random', choices=['random','geom'])
    parser.add_argument('--masking_rate', type=float, default=.1)
    
    # Agmentation parameters
    parser.add_argument('--p', type=float, default=.5)
    parser.add_argument('--jitter_sigma', type=float, default=.03)
    parser.add_argument('--scaling_sigma', type=float, default=.1)
    
    """Finetuning parsers"""
    # training hyperparameters
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=512)
    
    # optimizer parameters
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'SGD'])
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    
    # scheduler
    parser.add_argument('--scheduler', type=str, default='step', choices=['exp','step','constant'])
    parser.add_argument('--warm-up', type=int, default=25)
    parser.add_argument('--step-period', type=int, default=25)
    parser.add_argument('--lr-decay', type=float, default=.8)
    
    """Model parsers"""
    # Classification MLP Head
    parser.add_argument('--mlp-hidden', type=int, default=256)
    parser.add_argument('--mlp-dropout', type=float, default=.2)
    
    # Transformer
    parser.add_argument('--dropout', type=float, default=.2)
    parser.add_argument('--emb_dropout', type=float, default=.0)
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--heads', type=int, default=2)
    
    # GRU
    parser.add_argument('--bidirectional', action='store', default=True)
    parser.add_argument('--gru_input_size', type=int, default=11)
    parser.add_argument('--gru_emb_dim', type=int, default=32)
    parser.add_argument('--gru_hid_dim', type=int, default=32)
    parser.add_argument('--gru_layers', type=int, default=1)
    parser.add_argument('--gru_dropout', type=float, default=.2)
    
    args = parser.parse_args()
    
    data_path = f'./dataset/{args.dataset}'
    subj_list = np.array(os.listdir(data_path))
    
    ## Transformer hyperparameters
    if args.dataset == 'mobiact':
        args.num_classes, args.num_signals = 11, 9
    
    elif args.dataset =='motion_sense':
        args.num_classes, args.num_signals = 6, 12
                
    elif args.dataset == 'uci_har':
        args.num_classes, args.num_signals = 6, 9
    
    elif args.dataset == 'usc_had':
        args.num_classes, args.num_signals = 12, 6
    else:
        raise Exception(f"Dataset Error : {args.dataset} dose not exist.")
    
    args.gru_hid_dim = args.emb_dim
    args.gru_emb_dim = args.emb_dim
    args.head_dim = int(args.emb_dim/args.heads)
    args.transformer_mlp_dim = args.emb_dim*4
    
    if args.gru_layers == 1:
        args.gru_dropout = 0
    
    # Create save path
    save_path = f'./results/{args.dataset}/fold_{args.exp_fold}'
    os.makedirs(save_path, exist_ok=True)
    
    # Fix seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Split data using K-Fold
    kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
    kf.get_n_splits(subj_list)
    
    for i, (tn, ts) in enumerate(kf.split(subj_list)):
        if i+1 == args.exp_fold:
            crt_tn = tn
            crt_ts = ts
            break
        
    train_len = int(np.ceil(len(crt_tn)*0.8))
    subj_dict = {'train':subj_list[crt_tn][:train_len],
                'valid':subj_list[crt_tn][train_len:],
                'test':subj_list[crt_ts]}
    
    with open(os.path.join(save_path, f'{args.exp_fold}_fold_split.pkl'), 'wb') as fw:
        pickle.dump(subj_dict, fw)
    
    # Define each loaders
    pretrain_loader, train_loader, valid_loader, test_loader = make_data_loader(args, subj_dict)
    
    # Pretraining
    print('-------------------Pretraining Start-------------------')
    pretrain_trainer = SWISS_trainer(args, pretrain_loader, save_path)
    pretrain_history = pretrain_trainer.run(epochs=args.pre_epochs)
        
    # Finetuning each pretrain epochs
    print(f'-------------------Finetuning with encoder Start-------------------')
    finetune_trainer = HAR_trainer(args, train_loader, valid_loader, save_path, args.pretrained)
    train_history = finetune_trainer.run(epochs=args.epochs)

    # Test
    print(f'-------------------Test-------------------')
    epoch_tester = HAR_tester(args, test_loader, save_path, best=False, pretrained=args.pretrained)
    epoch_tester.test()
    
if __name__ == '__main__':

    main()