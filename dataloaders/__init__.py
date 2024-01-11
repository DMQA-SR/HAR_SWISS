import os
import sys
import argparse

sys.path.append(os.getcwd())
from torch.utils.data import DataLoader
from dataloaders.dataset.HAR_datasets import load_dataset, Downstream_dataset, SWISS_dataset

def make_data_loader(args: argparse, subj_dict: dict, **kwargs: dict):

    # Get dataset list
    train_data, valid_data, test_data = load_dataset(args, subj_dict)

    # Define dataset
    pretrain_set = SWISS_dataset(train_data, args)
    train_set = Downstream_dataset(train_data, args)
    valid_set = Downstream_dataset(valid_data, args)
    test_set = Downstream_dataset(test_data, args)

    pretrain_loader = DataLoader(pretrain_set,
                              batch_size=args.pre_batch_size, shuffle=True,
                              **kwargs)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size, shuffle=True,
                              **kwargs)

    valid_loader = DataLoader(valid_set,
                              batch_size=args.batch_size, shuffle=False,
                              **kwargs)

    test_loader = DataLoader(test_set,
                              batch_size=args.batch_size, shuffle=False,
                              **kwargs)

    return pretrain_loader, train_loader, valid_loader, test_loader