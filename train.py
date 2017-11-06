#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data

from data import COCOroot, make_dataloader_coco 
from data import VOCroot, make_dataloader_voc 
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import SSD300

import numpy as np
import os, time, argparse


def train(args):

    ssd_dim = 300
    means = (104, 117, 123)
    max_iter = 120000
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuID
    if args.cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    if args.tensorboard:
        from tensorboard import SummaryWriter
        writer = SummaryWriter()

    # data
    #data_loader = make_dataloader_coco(args.coco_root, Augmentation(size),
    #                                   args.batch_size, args.num_workers, pin_memory=False)
    #num_classes = 81 # coco

    train_sets = [('2007', 'trainval'), ('2012', 'train')]
    val_sets = [('2012', 'val')]
    data_loader_train = make_dataloader_voc(args.voc_root, train_sets, SSDAugmentation(ssd_dim, means),
                                      args.batch_size, args.num_workers, pin_memory=False)
    data_loader_val = make_dataloader_voc(args.voc_root, val_sets, SSDAugmentation(ssd_dim, means),
                                      args.batch_size, args.num_workers, pin_memory=False)
    num_classes = 21 

    # model
    net = SSD300(num_classes, 'train', pretrain=True)
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        net.load_state_dict(torch.load(args.resume))
    if args.cuda:
        net = net.cuda()
    net.train() 

    # optimizer
    optimizer = optim.SGD(net.parameters(), 
                          lr=args.lr,
                          momentum=args.momentum, 
                          weight_decay=args.weight_decay)
    lr_sche = lr_scheduler.MultiStepLR(optimizer, milestones=[80000, 100000], gamma=args.gamma)
    #lr_sche = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # loss
    criterion = MultiBoxLoss(
            num_classes, 
            overlap_thresh = 0.5, 
            prior_for_matching = True, 
            bkg_label = 0, 
            neg_mining = True, 
            neg_pos = 3, 
            neg_overlap = 0.5, 
            encode_target = False, 
            use_gpu = args.cuda)

    loc_loss = 0  
    conf_loss = 0
    num_epochs = int(max_iter / len(data_loader_train)) + 1
    for epoch in range(num_epochs):
        for i, (images, targets) in enumerate(data_loader_train):
            iteration = i + epoch * len(data_loader_train)
            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            else:
                images = Variable(images)
                targets = [Variable(anno, volatile=True) for anno in targets]

            # forward
            t0 = time.time()
            out = net(images)

            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()

            # logging
            loc_loss += loss_l.data[0]
            conf_loss += loss_c.data[0]
            if iteration % 10 == 0:
                lr = list(map(lambda group: group['lr'], optimizer.param_groups))[0]
                print('iter ' + repr(iteration) + ' || train loss: %.4f || ' % (loss.data[0]) + 'lr: %.4f' % lr)
            if args.tensorboard:
                writer.add_scalar('train_loss/cls', loss_c.data[0], iteration)
                writer.add_scalar('train_loss/loc', loss_l.data[0], iteration)

            # val and save model
            if False: # No enough memory
                print('validating...')
                val_loss_l = 0
                val_loss_c = 0
                val_sum = 1000
                for enum, (images2, targets2) in enumerate(data_loader_val):
                    if args.cuda:
                        images2 = Variable(images2.cuda())
                        targets2 = [Variable(anno.cuda(), volatile=True) for anno in targets2]
                    else:
                        images2 = Variable(images2)
                        targets2 = [Variable(anno, volatile=True) for anno in targets2]
                    out = net(images2)
                    loss_l, loss_c = criterion(out, targets2)
                    val_loss_l += loss_l.data[0]
                    val_loss_c += loss_c.data[0]
                    if enum > val_sum:
                        break
                val_loss_l /= val_sum 
                val_loss_c /= val_sum 
                val_loss = val_loss_l + val_loss_c
                print('val loss: %.4f' % (val_loss))
                lr_sche.step(val_loss)
                if args.tensorboard:
                    writer.add_scalar('val_loss/cls', val_loss_c, iteration)
                    writer.add_scalar('val_loss/loc', val_loss_l, iteration)

            if iteration % 5000 == 0 and iteration != 0:
                print('Saving state, iter:', iteration)
                torch.save(net.state_dict(), os.path.join(args.save_folder, 'ssd300_0712_{}.pth'.format(iteration))) 

            if iteration >= max_iter:
                break
        if iteration >= max_iter:
            break
    torch.save(net.state_dict(), os.path.join(args.save_folder, 'ssd300_0712_final.pth')) 
    if args.tensorboard:
        writer.close()

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
    parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
    parser.add_argument('--gpuID', default='0', type=str, help='Use which gpu to train')

    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')

    parser.add_argument('--tensorboard', default=True, type=str2bool, help='Use tensorboard for loss visualization')

    parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
    parser.add_argument('--coco_root', default=COCOroot, help='Location of OCT root directory')
    parser.add_argument('--voc_root', default=VOCroot, help='Location of OCT root directory')
    args = parser.parse_args()
    print(args)
    train(args)
