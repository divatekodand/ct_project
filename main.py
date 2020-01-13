# Training
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# python imports
import argparse
import os
import time
import math
import random
from types import SimpleNamespace
from collections import OrderedDict

#debugging
import pdb

# numpy imports
import numpy as np
from scipy.special import expit
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
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
from utils import AverageMeter
from losses import *
from ct_loader import *
from models import *
from models import CTNetworkBranch
from utils import *

import pdb
import sys



# tensorboard writer
writer = SummaryWriter('./logs')

# model
from network import CTNetwork2d

# main function for training and testing
def main(args):
    # parse args
    print('args - ', args)

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
    width_per_group = 8
    if args.model == "CTNetwork2d":
        pretrained = True
        model  = CTNetwork2d(lossfn=args.loss_2d, inchannels=inchannels, num_classes=num_classes, pretrained=pretrained)
    else:
        model = CTNetworkBranch(lossfn=args.loss_2d, groups=inchannels, num_classes=num_classes, width_per_group=width_per_group)
    
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

    # setup the optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

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
                if torch.cuda.device_count() > 1:
                    print("Using", torch.cuda.device_count(), "GPUs!")
                    model = nn.DataParallel(model)
                    model.to(device)
                    args.batch_size = args.batch_size * torch.cuda.device_count()
                else:
                    model = model.cuda(args.gpu)

            # only load the optimizer if necessary
            if not args.evaluate:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}, acc1 {})"
                  .format(args.resume, checkpoint['epoch'], best_acc1))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # setup dataset and dataloader
    if args.random_data:
        train_dataset = CTLoader_dummy(args.root_dir_2d, args.label_file_2d, args.root_dir_3d, args.label_file_3d,
                                       groups=args.group, split='train', num_val=args.num_val, sample=True, modalities=args.modalities)
        val_dataset = CTLoader_dummy(args.root_dir_2d, args.label_file_2d, args.root_dir_3d, args.label_file_3d,
                                     groups=args.group, split='val', num_val=args.num_val, sample=True, modalities=args.modalities)
    else:
        train_dataset = CTLoader(args.root_dir_2d, args.label_file_2d, args.root_dir_3d, args.label_file_3d,
                                 groups=args.group, split='train', num_val=args.num_val, sample=0, preprocessed2d=args.preprocessed2d)
        val_dataset = CTLoader(args.root_dir_2d, args.label_file_2d, args.root_dir_3d, args.label_file_3d,
                               groups=args.group, split='val', num_val=args.num_val, sample=0, preprocessed2d=args.preprocessed2d)
        
        print('Train dataset length - ', len(train_dataset))
        print('Val dataset length - ', len(val_dataset))
 

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)

    criterion = None
    # evaluation
    if args.resume and args.evaluate:
        print("Testing the model ...")
        cudnn.deterministic = True
        validate(val_loader, model, -1, args, device)
        return

    # enable cudnn benchmark
    cudnn.enabled = True
    cudnn.benchmark = True

    # warmup the training
    if (args.start_epoch == 0) and (args.warmup_epochs > 0):
        print("Warmup the training ...")
        for epoch in range(0, args.warmup_epochs):
            train(train_loader, model, criterion, optimizer, epoch, "warmup", args, device)

    # start the training
    print("Training the model ...")
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, "train", args, device)

        # evaluate on validation set
        # pdb.set_trace()
        acc1 = validate(val_loader, model, epoch, args, device)

        # remember best acc@1 and save checkpoint
        acc1_any = acc1[0].item()
        is_best = acc1_any > best_acc1
        best_acc1 = max(acc1_any, best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'model_arch': 'model_arch',
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)


def save_checkpoint(state, is_best,
                    file_folder="./models/", filename='checkpoint.pth.tar'):
    """save checkpoint"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    filename = str(state['epoch']) + '__' + filename
    torch.save(state, os.path.join(file_folder, filename))
    if is_best:
        # skip the optimization state
        #state.pop('optimizer', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


def train(train_loader, model, criterion, optimizer, epoch, stage, args, device):
    """Training the model"""
    assert stage in ["train", "warmup"]
    # adjust the learning rate
    num_iters = len(train_loader)
    lr = 0.0

    # set up meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = []
    for i in range(args.num_classes):
        accuracies.append(AverageMeter())

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # adjust the learning rate
        # input shape N,128,200,200
        # target shape N,128,6
        if stage == "warmup":
            # warmup: linear scaling
            lr = (epoch * num_iters + i) / float(
                args.warmup_epochs * num_iters) * args.lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                param_group['weight_decay'] = 0.0
        else:
            # cosine learning rate decay
            lr = 0.5 * args.lr * (1 + math.cos(
                (epoch * num_iters + i) / float(args.epochs * num_iters) * math.pi))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                param_group['weight_decay'] = args.weight_decay

        #pdb.set_trace()
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu >= 0:
            # input = input.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)

            input = input.to(device)
            target = target.to(device)

        # compute output
        # print('bias - ', model.fc.bias)
        output, output_map = model(input)
        # print('output -', output)
        # print('target-', target)
        # print(output_map)
        if True:
            if args.loss_2d == 'ce':
                loss = groupBCELoss(output, target, args)
            else:
                loss = focal_loss(output, target, args)
        else:
            loss = set_loss(output, target, args) + kld_loss(output_map, args)

        #print('loss - ', loss)
        # measure accuracy and record loss
        # pdb.set_trace()
        acc = accuracy(output, target, args.num_classes)
        losses.update(loss.item(), input.size(0))
        for j in range(len(accuracies)):
            accuracies[j].update(acc[j], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        output_types = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
        # printing
        # pdb.set_trace()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'
                  'Any {accuracies0.val:.2f} ({accuracies0.avg:.2f})\t'
                  'epidural {accuracies1.val:.2f} ({accuracies1.avg:.2f})\t'
                  'intraparenchymal {accuracies2.val:.2f} ({accuracies2.avg:.2f})\t'
                  'intraventricular {accuracies3.val:.2f} ({accuracies3.avg:.2f})\t'
                  'subarachnoid {accuracies4.val:.2f} ({accuracies4.avg:.2f})\t'
                  'subdural {accuracies5.val:.2f} ({accuracies5.avg:.2f})'.format(
                epoch + 1, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, accuracies0=accuracies[0],
                accuracies1=accuracies[1], accuracies2=accuracies[2],
                accuracies3=accuracies[3], accuracies4=accuracies[4], accuracies5=accuracies[5]))

            sys.stdout.flush()

            # log loss / lr
            if stage == "train":
                writer.add_scalar('data/training_loss',
                                  losses.val, epoch * num_iters + i)
                writer.add_scalar('data/learning_rate',
                                  lr, epoch * num_iters + i)

    # print the learning rate
    print("[Stage {:s}]: Epoch {:d} finished with lr={:f}".format(
        stage, epoch + 1, lr))

    # log top-1/5 acc
    for j in range(len(output_types)):
        writer.add_scalars('data/' + output_types[j],
                           {"train": accuracies[j].val}, epoch + 1)


def validate(val_loader, model, epoch, args, device):
    """Test the model on the validation set"""
    batch_time = AverageMeter()
    accuracies = []
    for _ in range(args.num_classes):
        accuracies.append(AverageMeter())

    # switch to evaluate mode (autograd will still track the graph!)
    model.eval()

    # disable/enable gradients
    grad_flag = False
    with torch.set_grad_enabled(grad_flag):
        end = time.time()
        predicted_list = []
        target_list = []
        # loop over validation set
        for i, (input, target) in enumerate(val_loader):
            if args.gpu >= 0:
                # input = input.cuda(args.gpu, non_blocking=True)
                # target = target.cuda(args.gpu, non_blocking=True)

                input = input.to(device)
                target = target.to(device)

            # forward the model
            output, output_map = model(input)
            #TODO: Reshape the tensor?
            #pdb.set_trace()
            predicted_list.extend(output.reshape(output.shape[0], args.group, -1)[:,:,0].reshape(-1).tolist())
            target_list.extend(target.reshape(target.shape[0], args.group, -1)[:,:,0].reshape(-1).tolist())
            # target_list.extend(target[:, :, 0].reshape(-1).tolist())

            # measure accuracy and record loss
            # pdb.set_trace()
            acc = accuracy(output.reshape(output.shape[0], -1), target.reshape(target.shape[0], -1), args.num_classes)
            for j in range(len(accuracies)):
                accuracies[j].update(acc[j], input.size(0))

            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()

            output_types = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
            # printing
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Any {accuracies0.val:.2f} ({accuracies0.avg:.2f})\t'
                      'epidural {accuracies1.val:.2f} ({accuracies1.avg:.2f})\t'
                      'intraparenchymal {accuracies2.val:.2f} ({accuracies2.avg:.2f})\t'
                      'intraventricular {accuracies3.val:.2f} ({accuracies3.avg:.2f})\t'
                      'subarachnoid {accuracies4.val:.2f} ({accuracies4.avg:.2f})\t'
                      'subdural {accuracies5.val:.2f} ({accuracies5.avg:.2f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    accuracies0=accuracies[0], accuracies1=accuracies[1], accuracies2=accuracies[2],
                    accuracies3=accuracies[3], accuracies4=accuracies[4], accuracies5=accuracies[5]))
                sys.stdout.flush()
    # pdb.set_trace()
    metrics_dict, plt = get_metrics(predicted_list, target_list)
    plt.savefig('epoch_' + str(epoch) + '.png')
    print('metrics - ', metrics_dict)

    print('******Any {acc0.avg:.3f} epidural {acc1.avg:.3f} intraparenchymal {acc2.avg:.3f} \
        intraventricular {acc3.avg:.3f} subarachnoid {acc4.avg:.3f} subdural {acc5.avg:.3f}'.format(
        acc0=accuracies[0], acc1=accuracies[1], acc2=accuracies[2],
        acc3=accuracies[3], acc4=accuracies[4], acc5=accuracies[5]))
    sys.stdout.flush()

    if (not args.evaluate):
        for j in range(len(output_types)):
            writer.add_scalars('data/' + output_types[j],
                               {"val": accuracies[j].val}, epoch + 1)
    # pdb.set_trace()
    return [accuracy.val for accuracy in accuracies]


def get_metrics(predicted, target):
    predicted = np.array(predicted)
    target = np.array(target)
    prob = scipy.special.expit(predicted)
    p_label = prob > 0.5
    p_label = p_label.astype(float)

    all_metrics = {}
    # pdb.set_trace()
    tn, fp, fn, tp = confusion_matrix(target, p_label).ravel()
    confusion_mat = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
    all_metrics['confusion_mat'] = confusion_mat
    # pdb.set_trace()
    all_metrics['precision'] = (tp * 1.0) / (tp + fp)
    all_metrics['recall'] = tp / (tp + fn)
    all_metrics['f1'] = 2 * (all_metrics['precision'] * all_metrics['recall']) / (all_metrics['precision'] + all_metrics['recall'])
    all_metrics['accuracy'] = (tp + tn) / (tp + fp + tn + fn)

    fpr, tpr, thresholds = metrics.roc_curve(target, prob, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    # all_metrics['fpr'] = fpr
    # all_metrics['tpr'] = tpr
    # all_metrics['thresholds'] = thresholds
    all_metrics['roc_auc'] = roc_auc

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    # plt.show()
    return all_metrics, plt


def accuracy(output, target, num_classes):
    """
    output = batch x (group x num_classes)
    target = batch x (group x num_classes)
    """
    # pdb.set_trace()
    is3d = False
    with torch.no_grad():
        if output.size() != target.size():
            is3d = True
            output = output.view(-1, num_classes)
            output, _ = torch.max(output, 0, keepdims=True)
            # output = output.view(-1, num_classes)
        output = (output > 0.5).float()
        correct = (output == target).float()
        correct = correct.view(-1, num_classes)
        accuracy = (torch.sum(correct, axis=0) / correct.size(0)) * 100.0
    return accuracy



if __name__ == '__main__':
    # the arg parser
    parser = argparse.ArgumentParser(description='3D CT Image Analysis')
    parser.add_argument('--root_dir_2d',  default='/UserData/CT_2D_project/data/rsna/stage_1_train_images', 
                        type=str, metavar='2D_DIR', help='path to 2D labelled dicom images')
    parser.add_argument('--label_file_2d',  default='/UserData/CT_2D_project/data/rsna/stage_1_train.csv', 
                        type=str, metavar='2D_Labels', help='2D labels')
    parser.add_argument('--preprocessed2d',  default='/UserData/CT_2D_project/data/rsna/preprocessed_stage_1_train_images', 
                        type=str, metavar='2D_Preprocessed', help='2D preprocessed data folder')
    parser.add_argument('--root_dir_3d',  default='/UserData/CT_2D_project/data/cq500', 
                        type=str, metavar='3D_DIR', help='path to 3D labelled dicom images')
    parser.add_argument('--label_file_3d',  default='/UserData/CT_2D_project/data/cq500/reads.csv', 
                        type=str, metavar='3D_Labels', help='3D labels')
    parser.add_argument('--num_val', default=1000, type=int,
                        help='Number of Images in val set')
    parser.add_argument('-m', '--modalities', default=1, type=int, metavar='W',
                        help='Number of modalities of data 1 - 2d labelled only, \
                             2 - 2d and 3d labelled, 3 - 2d labelled, 3d labelled and 3d unlabelled')

    parser.add_argument('-j', '--workers', default=0, type=int, metavar='W',
                        help='number of data loading workers (default: 0)')

    parser.add_argument('-c', '--num_classes', default=6, type=int, metavar='W',
                        help='Number of outputs from the neural network (default 6)')

    parser.add_argument('-r', '--random_data', dest='random_data', action='store_true',
                        help='Use randomly generated data to verify all tensor dims')

    parser.add_argument('--model',  default='CTNetwork2d', 
                        type=str, metavar='model', help='Model to use')
    parser.add_argument('--group', default=128, type=int, metavar='G',
                        help='number of total epochs to run')
    parser.add_argument('--classes', default=6, type=int, metavar='V',
                        help='Number of Output Classes')

    parser.add_argument('--loss_2d',  default='focal',
                        type=str, metavar='Loss', help='Loss function for 2D Images (ce or focal) ')

    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--warmup-epochs', default=0, type=int,
                        help='number of epochs for warmup')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        metavar='N',
                        help='mini-batch size (default: 1)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU IDs to use.')
    
    args = parser.parse_args()

    main(args)





