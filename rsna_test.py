
from types import SimpleNamespace
from ct_loader import *
import scipy.io as sio
import sys
import os
import pdb
import numpy as np

arguments = {'gpu': 0,
             'resume': None,
             'evaluate': False,
             'root_dir_2d': '../../data/rsna/stage_1_train_images',
             'label_file_2d': '../../data/rsna/stage_1_train.csv',
             'root_dir_3d': '../../data/cq500',
             'label_file_3d': '../../data/cq500/reads.csv',
             'group': 128,
             'num_val': 1000,
             'batch_size': 1,
             'workers': 0,
             'resume': '',
             'print_freq': 1,
             'num_classes': 6,
             'warmup_epochs': 0,
             'lr': 0.0001,
             'momentum': 0.9,
             'weight_decay': 1e-4,
             'dummy': False,
             'start_epoch': 0,
             'epochs': 100,
             'create_preprocessed': False}

args = SimpleNamespace(**arguments)      
dir_path = '../../data/rsna/preprocessed_stage_1_train_images'

train_dataset = CTLoader(args.root_dir_2d, args.label_file_2d, args.root_dir_3d, args.label_file_3d,
                                       groups=args.group, split='train', num_val=args.num_val, sample=False)


print('length - ', len(train_dataset))

for i in range(train_dataset.num_2d):
    try:
        scans, labels, filenames = train_dataset[i]
        print('scan shape - ', scans.shape)
        print('labels shape - ', labels.shape)
        sys.stdout.flush()
        if args.create_preprocessed:
            matname = str(i) + '.mat'
            sio.savemat(os.path.join(dir_path, matname), {'x': scans.numpy(), 'y': labels.numpy(), 'filenames': filenames})
    except:
        print('Failed')
