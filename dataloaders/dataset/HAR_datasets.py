import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
sys.path.append(os.getcwd())

from utils.helper import custom_norm_transform, custom_normalize, geom_noise_mask_single

def load_dataset(args, subj_dict):
    """
    x shape : (# of obs, # of signals, length)
    y shape : (# of obs,)
    dataset type : uci_har or mhealth or hhar
    """
    train_list, valid_list, test_list = subj_dict['train'], subj_dict['valid'], subj_dict['test']
    
    train_data, tr_obs, tr_v = dataset_loader(args, train_list, 'train')
    valid_data, v_obs, v_v  = dataset_loader(args, valid_list, 'valid')
    test_data, te_obs, te_v = dataset_loader(args, test_list, 'test')
    
    train_data['x'], means, stds = custom_normalize(train_data['x'])
    valid_data['x'] = custom_norm_transform(valid_data['x'], means, stds)
    test_data['x'] = custom_norm_transform(test_data['x'], means, stds)
        
    total_obs = tr_obs+v_obs+te_obs
    
    print('-'*30,f'{args.dataset}','-'*30)
    print(f'train / valid / test split : {len(train_list)} volunteers / {len(valid_list)} volunteers / {len(test_list)} volunteers')
    print(f"Total obs : {total_obs} / Train : Valid : Test = {tr_obs} : {v_obs} : {te_obs}")
    
    print(f'train volunteers : {tr_v}')
    print(f'valid volunteers : {v_v}')
    print(f'test volunteers : {te_v}')
            
    return train_data, valid_data, test_data


def dataset_loader(args, subject_list, mode):
    
    volunteers = []
    data_path = f'./dataset/{args.dataset}'
    x, y, sub = torch.empty(0), torch.empty(0), torch.empty(0)

    for crt_path in subject_list:
        
        volunteer = crt_path.split('.')[0].replace('subject', '')
        crt = os.path.join(data_path, crt_path)
        crt_df = np.load(crt)
        crt_x, crt_y, crt_sub = crt_df['x'], crt_df['y'], crt_df['sub']

        crt_x = torch.FloatTensor(crt_x)
        crt_y = torch.Tensor(crt_y)
        crt_sub = torch.Tensor(crt_sub)
        
        x = torch.cat([x, crt_x])
        y = torch.cat([y, crt_y])
        sub = torch.cat([sub, crt_sub])
        
        volunteers.append(volunteer)

    print(f'[{mode} Loader] {mode} observations : {x.shape[0]}')
    
    data_dict = {'x':x, 'y':y, 'sub':sub}
    obs = x.shape[0]
    
    """
    x shape : # of obs, # of signals, 128
    y shape : # of obs,
    """
    return data_dict, obs, volunteers
    

class Downstream_dataset(Dataset):
    def __init__(self, data_dict, args):
        """
        x shape : (# of obs, # of signals, length)
        y shape : (# of obs,)
        """
        self.args = args
        self.x, self.y = data_dict['x'], data_dict['y']

    def __getitem__(self, index):        

        input = self.x[index]
        target = self.y[index]
            
        return input, target

    def __len__(self):
        return self.x.shape[0]
    
    
class SWISS_dataset(Dataset):
    def __init__(self, data_dict, args):
        """
        x shape : (# of obs, # of signals, length)
        y shape : (# of obs,)
        sub shape : (# of obs,)
        """
        self.args = args
        self.x, self.sub = data_dict['x'], data_dict['sub']
        self.aug_list = ['jitter', 'scaling', 'rotation', 'permutation', 'channel_shuffle']

    def __getitem__(self, index):        

        input = self.x[index]
        subj = self.sub[index]
            
        mask = self.creat_mask(input.shape, self.args.mask_type, self.args.masking_rate)

        x_1, x_2 = input, input
            
        return x_1, x_2, mask, subj

    def __len__(self):
        return self.x.shape[0]
    
    def creat_mask(self, mask_shape, mask_type, p):
        """
        state 0 means not masking, 1 means masking """
        s, l = mask_shape
        mask_l = int(np.ceil(l*p))
        if mask_type == 'random':
            mask = torch.rand(mask_shape).ge(1-p)
        
        elif mask_type == 'geom':
            mask = torch.zeros(mask_shape, dtype=torch.bool)
            for i in range(s):
                mask[i, :] = geom_noise_mask_single(l, lm=mask_l, masking_ratio=p)
                
        elif mask_type == 'poisson':
            mask = torch.zeros(mask_shape, dtype=torch.bool)
            for i in range(s):
                try:
                    crt_l = np.random.poisson(mask_l)
                    crt_point = np.random.randint(0,l-crt_l+1)
                    mask[i,crt_point:crt_point+crt_l] = 1
                except:
                    mask[i,:] = 1
                
        elif mask_type == 'same':
            mask = torch.zeros(mask_shape, dtype=torch.bool)
            for i in range(s):
                crt_point = np.random.randint(0,l-mask_l+1)
                mask[i,crt_point:crt_point+mask_l] = 1
        
        else:
            raise Exception(f'mask type error: there is no {mask_type} mask type')
        
        return mask