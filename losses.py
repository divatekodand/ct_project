# Loss functions
import pdb
import torch
import torch.nn as nn


def groupBCELoss(predicted, target, args, num_classes=6, groups=128):
    # predicted shape  = batch_size X (groups*6)
    # target = batch_size X (groups*6)
    # [torch.Size([1, 900]), torch.Size([1, 900, 13, 13])]
    loss = nn.BCEWithLogitsLoss()
    if args.gpu >= 0:
        loss = loss.cuda(args.gpu)
    output = loss(predicted, target)
    return output


def focal_loss(predicted, target, args, num_classes=6, groups=128, alpha=0.25, gamma=2):
    loss = nn.BCEWithLogitsLoss(reduction='none')
    act_fn = nn.Sigmoid()
    if args.gpu >= 0:
        loss = loss.cuda(args.gpu)
        act_fn = act_fn.cuda(args.gpu)
    c_loss = loss(predicted, target)
    probs = act_fn(predicted)
    f_loss = c_loss * (alpha * torch.pow(1 - probs, gamma) * target +
                                (1-alpha) * torch.pow(probs, gamma) * (1 - target))
    f_loss = torch.sum(f_loss)
    return f_loss


def kld_loss(group_maps, args, num_classes=6):
    group_maps = group_maps.view(-1, num_classes, group_maps.shape[2], group_maps.shape[3])
    activation_fn = nn.Sigmoid()
    log_activation_fn = torch.nn.LogSigmoid()
    activation = activation_fn(group_maps)
    logits = log_activation_fn(group_maps)
    loss = torch.nn.KLDivLoss()
    if args.gpu >= 0:
        loss = loss.cuda(args.gpu)
    size0 = group_maps.shape[0]
    shift1_map = logits[1:size0, :, :, :]
    shift2_map = activation[0:size0 - 1, :, :, :]
    output = loss(shift1_map, shift2_map)
    return output


def set_loss(predicted, target, args, num_classes=6, grpups=128):
    predicted = predicted.view(-1, num_classes)
    predicted, _ = torch.max(predicted, 0, keepdims=True)
    return groupBCELoss(predicted, target, args)
