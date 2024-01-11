import torch

import numpy as np


def sliding_window(data, window_size, step_size):
    for x in range(0, data.shape[1]-window_size, step_size):
        yield data.iloc[:,x:x+window_size]
   
def custom_normalize(train_x: torch.FloatTensor):
    """z-normalization

    Args:
        train_x (torch.FloatTensor): [shape : N, S, T]
    """
    N,S,T = train_x.shape
    normalized_data = torch.zeros((N,S,T))
    
    means, stds = [], []
    for i in range(S):
        crt_signal = train_x[:,i,:]
        mean = crt_signal.mean()# mean of signal i
        std = crt_signal.std()# std of signal i
        means.append(mean)
        stds.append(std)
        normalized_data[:,i,:] = (crt_signal - mean) / std
    
    return normalized_data, means, stds

def custom_norm_transform(data, means, stds):
    N,S,T = data.shape
    normalized_data = torch.zeros((N,S,T))
    
    for i in range(S):
        crt_signal = data[:,i,:]
        mean = means[i]
        std = stds[i]
        normalized_data[:,i,:] = (crt_signal-mean) / std
    return normalized_data

def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked
    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.zeros(L, dtype=bool)

    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    if L == 33:
        p_u = (p_m * masking_ratio / (1 - masking_ratio))  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    else:
        p_u = (p_m * masking_ratio / (1 - masking_ratio))
        
    p = [p_u, p_m]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() < masking_ratio)  # state 0 means not masking, 1 means masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return torch.tensor(keep_mask)

def jitter(x, sigma=0.03):
    """x : 1, signals, time length"""
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling(x, sigma=0.1):
    """x : 1, signals, time length"""
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[2]))
    return np.multiply(x, factor[:,np.newaxis,:])

def rotation(x):
    """x : 1, signals, time length"""
    flip = np.random.choice([-1, 1], size=(x.shape[0],x.shape[2]))
    rotate_axis = np.arange(x.shape[2])
    np.random.shuffle(rotate_axis)    
    return flip[:,np.newaxis,:] * x[:,:,rotate_axis]

def permutation(x, max_segments=5, seg_mode="equal"):
    """x : 1, signals, time length"""
    orig_steps = np.arange(x.shape[1])
    
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1]-2, num_segs[i]-1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret
    
def channel_shuffle(x: np.array):
    """x : 1, signals, time length"""
    perm = np.arange(x.shape[1])
    np.random.shuffle(perm)
    return x[:,perm, :]
    
def make_description(args_name :list, *args):
    split = '| '
    strFormat = '%-20s'*len(args_name)
    
    str_list = list()
    for i, (arg, arg_name) in enumerate(zip(args, args_name)):
        if i==0:
            crt_str = arg_name+' : '+str(arg)
            str_list.append(crt_str)
        
        else:
            crt_str = split+arg_name+' : '+str(arg)
            str_list.append(crt_str)
        
    strFormat = '%-25s'*len(args_name)
    strout= strFormat % tuple(str_list)
    
    return print(strout)