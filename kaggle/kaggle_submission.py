# Training
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# python imports
import argparse
import sys
import os
import time
import math
import random
from types import SimpleNamespace
from collections import OrderedDict

sys.path.append('../')

# debugging
import pdb

# numpy imports
import numpy as np
from scipy.special import expit
# from sklearn.metrics import confusion_matrix
# import sklearn.metrics as metrics
import matplotlib.pyplot as plt

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

# pretrained models
import torchvision

# for visualization
from torch.utils.tensorboard import SummaryWriter

# our code
from losses import *
from ct_loader import *
from models import *
from models import CTNetworkBranch
from utils import *


# tensorboard writer
writer = SummaryWriter('./logs')

# model
from network import CTNetwork2d


""" Generates a submission file (.csv) for the Kaggle Competition - 
https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/overview"""


import os
import torch
from torch.utils import data
import numpy as np
import pandas as pd
import pdb
import random
import pydicom
import scipy
import scipy.ndimage
import scipy.io as sio
import random
import matplotlib.pyplot as plt
from skimage import exposure
import pickle

glabels2d = None
glabels3d = None
gct3d_desc = None


def is_dcmdir(directory, count_thresh=150):
    files = os.listdir(directory)
    num_dcm = len([1 for file in files if file[-4:] == '.dcm'])
    #     print('len - ', num_dcm)
    if num_dcm > count_thresh:
        return True
    else:
        return False


class KaggleTest(data.Dataset):
    """
    Dataloader for CT Image Analysis
    """

    def __init__(self,
                 data_dir=None,
                 submission_csv=None,
                 num_classes=6,
                 sample=0,
                 transforms=None,
                 verify_data=False,
                 device=None):
        """
        data_dir: should contain 'data' sub dir. 'cache' and 'preprocessed' sub directories may be created.
        submission_file: path to template csv file
        """
        self.data_dir = data_dir
        self.submission_csv = submission_csv
        self.n_classes = num_classes
        self.sample = sample
        self.verify_data = verify_data
        self.device = device
        self.cache = True
        self.submission_df = pd.read_csv(self.submission_csv)
        self.abnormality_dict = {'any':5, 'epidural':0, 'intraparenchymal':1, 'intraventricular':2, 'subarachnoid':3, 'subdural':4}

        # create a dictionary that maps every file in the sample submission to its index in dataframe
        self.submission_dict = {}
        self.submission_dflen = len(self.submission_df)
        for i in range(self.submission_dflen):
            self.submission_dict[self.submission_df.iloc[i,0]] =  i

        self.raw_dir = os.path.join(self.data_dir, 'data')
        self.preprocessed_dir = os.path.join(self.data_dir, 'preprocessed')
        self.cache_dir = os.path.join(self.data_dir, 'cache')

        self.excess_files = []

        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
            self.cache = False

        if not os.path.exists(self.preprocessed_dir):
            os.mkdir(self.preprocessed_dir)
            # read all the files in the test directory and verify that there is a file corresponding to each entry in submission file
            self.dcm_files = os.listdir(self.raw_dir)
            
            if self.verify_data:
                tmp_submission_dict = self.submission_dict
                for dcm_file in self.dcm_files:
                    file_name = dcm_file.split('.')[0]
                    abnormalities = self.abnormality_dict.keys()
                    for abnormality in abnormalities:
                        submission_entry = file_name + '_' + abnormality
                        if submission_entry in tmp_submission_dict:
                            del tmp_submission_dict[submission_entry]
                        else:
                            self.excess_files.append(dcm_file)

                if len(tmp_submission_dict) != 0:
                    print('Not all entries in the submission file have a corresponding file, excess entries - ', len(tmp_submission_dict))
                    # print('Entries with no corresponding file - ', tmp_submission_dict.keys())
                    assert 'Non matching submission template and test set'

            # convert each file to corresponding preprocessed matfile
            print('Writing Preprocessed files')
            new_spacing = [1,1,1]
            for dcm_file in self.dcm_files:
                ds = pydicom.dcmread(os.path.join(self.raw_dir, dcm_file))
                intercept = ds.RescaleIntercept
                slope = ds.RescaleSlope
                spacing = ds.PixelSpacing
                image = np.array(ds.pixel_array, dtype=np.int16)
                image[image == -2000] = 0
                # Convert to Hounsfield units (HU)
                if slope != 1:
                    image = slope * image.astype(np.float64)
                    image = image.astype(np.int16)
                image += np.int16(intercept)
                image = np.array(image, dtype=np.int16)
                # resample
                new_spacing = [1,1,1]
                new_spacing = np.array(new_spacing)
                spacing = np.array([1] + list(map(float, spacing)))
                resize_factor = spacing / new_spacing
                new_real_shape = image.shape * resize_factor[1:]
                new_shape = np.round(new_real_shape)
                real_resize_factor = new_shape / image.shape
                new_spacing = spacing[1:] / real_resize_factor

                image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
                image = self.normalize(image)
                image = self.crop_centre(image)
                sio.savemat(os.path.join(self.preprocessed_dir ,dcm_file.split('.')[0]+'.mat'), {'img':image})

        self.preprocessed_files = sorted(os.listdir(self.preprocessed_dir))
        # pdb.set_trace()
        assert (len(self.submission_df) // 6) == len(self.preprocessed_files) 

    def __len__(self):
        return len(self.submission_df) // 6

    def __getitem__(self, index):
        # load img and label
        img = sio.loadmat(os.path.join(self.preprocessed_dir, self.preprocessed_files[index]))
        img = img['img']
        img = torch.tensor(img, dtype=torch.float32)
        img = img.to(self.device)
        return img

    def crop_centre(self, image, out_shape=[200, 200]):
        out_shape = out_shape
        image_shape = image.shape
        startx = image_shape[0] // 2 - (out_shape[0] // 2)
        starty = image_shape[1] // 2 - (out_shape[1] // 2)
        return image[startx:startx + out_shape[0], starty:starty + out_shape[1]]

    def normalize(self, image, min_bound=-1000.0, max_bound=400.0, mean=0.25):
        image = (image - min_bound) / (max_bound - min_bound)
        image[image > 1] = 1
        image[image < 0] = 0
        image -= mean
        return image


def histogram_equalize(img):
    img_cdf, bin_centers = exposure.cumulative_distribution(img)
    return np.interp(img, bin_centers, img_cdf)


def main(args):
    # parse args
    print('args - ', args)
    # pdb.set_trace()
    best_acc1 = 0.0

    if args.gpu >= 0:
        print("Use GPU: {}".format(args.gpu))
    else:
        print('You are using CPU for computing!',
              'Yet we assume you are using a GPU.',
              'You will NOT be able to switch between CPU and GPU training!')

    # fix the random seeds (the best we can)
    fixed_random_seed = 2019
    torch.manual_seed(fixed_random_seed)
    np.random.seed(fixed_random_seed)
    random.seed(fixed_random_seed)

    # set up the model + loss
    inchannels = args.group
    num_classes = 6
    pretrained = True
    model = CTNetwork2d(lossfn=args.loss_2d, inchannels=inchannels, num_classes=num_classes, pretrained=pretrained)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # put everthing to gpu
    if args.gpu >= 0:
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            if not args.resume:
                model = nn.DataParallel(model)
                model.to(device)
                args.batch_size = args.batch_size * torch.cuda.device_count()
        else:
            model = model.cuda(args.gpu)

    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            # TODO: Update the condition HERE
            if False:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)

            if args.gpu < 0:
                model = model.cpu()
            else:
                if False: # torch.cuda.device_count() > 1:
                    print("Using", torch.cuda.device_count(), "GPUs!")
                    model = nn.DataParallel(model)
                    model.to(device)
                    args.batch_size = args.batch_size * torch.cuda.device_count()
                else:
                    model = model.cuda(args.gpu)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    #setup basemodel
    # pdb.set_trace()
    basemodel = model.base_network

    # setup dataset and dataloader
    val_dataset = KaggleTest(args.data_dir, args.submission_file, device=device)
    print('Val dataset length - ', len(val_dataset))

    # clean the dataset and get the results
    preprocessed_set = set(val_dataset.preprocessed_files)
    act_fn = nn.Softmax().to(device)
    target_df = val_dataset.submission_df
    for i, input in enumerate(val_dataset):
        try:
            print('filename - ', val_dataset.preprocessed_files[i], ' input shape - ', input.shape)
            if input.shape[0] != 200 and input.shape[1]!=200:
                print('removing element - ', val_dataset.preprocessed_files[i])
                preprocessed_set.remove(val_dataset.preprocessed_files[i])
                # files failed preprocessing stage get low probability
                basename = val_dataset.preprocessed_files[i].split('.')[0] + '_epidural'
                base_index = val_dataset.submission_dict[basename]
                for i in range(5):
                    target_df.iloc[base_index+i, 1] = 0.01
                target_df.iloc[base_index+5, 1] = 0.01
            else:
                # pdb.set_trace()
                output = basemodel(input.reshape(1,1,200,200))
                output = act_fn(output)
                #find the base index in the target dataframe
                basename = val_dataset.preprocessed_files[i].split('.')[0] + '_epidural'
                base_index = val_dataset.submission_dict[basename]
                for i in range(5):
                    target_df.iloc[base_index+i, 1] = output[0, i+1].item()
                target_df.iloc[base_index+5, 1] = output[0, 0].item()
        except:
            print('Exception raised - ', val_dataset.preprocessed_files[i])
            print('removing element - ', val_dataset.preprocessed_files[i])
            preprocessed_set.remove(val_dataset.preprocessed_files[i])
            # files failed preprocessing stage get low probability
            basename = val_dataset.preprocessed_files[i].split('.')[0] + '_epidural'
            base_index = val_dataset.submission_dict[basename]
            for i in range(5):
                target_df.iloc[base_index+i, 1] = 0.01
            target_df.iloc[base_index+5, 1] = 0.01
    
    print('Saving Results!')
    target_df.to_csv(args.target, index = False)
    print('Number of removed - ', len(val_dataset.preprocessed_files)-len(preprocessed_set))
    return

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)

    criterion = None
    # evaluation
    if args.resume and args.evaluate:
        print("Testing the model ...")
        cudnn.deterministic = True
        # validate(val_loader, model, -1, args, device)
        for i, input in enumerate(val_loader):
            print(input.shape)

if __name__ == '__main__':
    # the arg parser
    parser = argparse.ArgumentParser(description='3D CT Image Analysis - Kaggle submission generation')

    # data director
    parser.add_argument('--data_dir', default='../../../data/rsna_stage2/data_dir/',
                        type=str, metavar='DATA_DIR', help='path to the test files directory')

    # model path
    parser.add_argument('--model', default='../logs/focal_loss_2d_02/models/model_best.pth.tar',
                        type=str, metavar='MODEL', help='path saved model')

    # submission file
    parser.add_argument('--submission_file', default='stage_2_sample_submission.csv',
                        type=str, metavar='SUB_FILE', help='path to submission file template')

    parser.add_argument('--target', default='./predictions.csv',
                        type=str, metavar='SUB_FILE', help='name of the target file')

    parser.add_argument('--loss_2d',  default='focal',
                        type=str, metavar='Loss', help='Loss function for 2D Images (ce or focal) ')

    parser.add_argument('--resume', default='../logs/focal_loss_2d_02/models/model_best.pth.tar', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')        

    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU IDs to use.')

    parser.add_argument('--group', default=128, type=int, metavar='G',
                        help='number of total epochs to run')

    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        metavar='N',
                        help='mini-batch size (default: 1)')

    parser.add_argument('-j', '--workers', default=0, type=int, metavar='W',
                        help='number of data loading workers (default: 0)')

    args = parser.parse_args()

    main(args)





