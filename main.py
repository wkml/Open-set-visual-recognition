import argparse
import os
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler 

from utils.transforms import get_train_test_set
from utils.load_pretrain_model import load_pretrain_model
from utils.metrics import AveragePrecisionMeter, voc12_mAP
from models import SSGRL

global best_prec1
best_prec1 = 0

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', metavar='DATASET',
                        help='path to train dataset')
    parser.add_argument('--train_data', metavar='DIR',
                        help='path to train dataset')
    parser.add_argument('--test_data', metavar='DIR',
                        help='path to test dataset')
    parser.add_argument('--train_list', metavar='DIR',
                        help='path to train list')
    parser.add_argument('--test_list', metavar='DIR',
                        help='path to test list')
    parser.add_argument('--pretrain_model', default='', type=str, metavar='PATH',
                        help='path to latest pretrained_model (default: none)')
    parser.add_argument('--train_label', default='', type=str, metavar='PATH',
                        help='path to train label (default: none)')
    parser.add_argument('--graph_file', default='', type=str, metavar='PATH',
                       help='path to graph (default: none)')
    parser.add_argument('--word_file', default='', type=str, metavar='PATH',
                       help='path to word feature')
    parser.add_argument('--test_label', default='', type=str, metavar='PATH',
                        help='path to test label (default: none)')
    parser.add_argument('--print_freq', '-p', default=100, type=int, metavar='N',
                        help='number of print_freq (default: 100)')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--step_epoch', default=30, type=int, metavar='N',
                        help='decend the lr in epoch number')
    parser.add_argument('--batch_size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained', type=int,default=0,
                        help='use pre-trained model')
    parser.add_argument('--crop_size', dest='crop_size',default=224, type=int,
                        help='crop size')
    parser.add_argument('--scale_size', dest = 'scale_size',default=448, type=int,
                        help='the size of the rescale image')
    parser.add_argument('--evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--post', dest='post', type=str,default='',
                         help='postname of save model')
    parser.add_argument('--num_classes', default=80, type=int, metavar='N',
                        help='number of classes (default: 80)')
    args = parser.parse_args()
    return args

def print_args(args):
    print("==========================================")
    print("==========       CONFIG      =============")
    print("==========================================")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("\n")


def main():
    global best_prec1
    args = arg_parse()
    print_args(args)

    # Create dataloader
    print("==> Creating dataloader...")
    train_data_dir = args.train_data
    test_data_dir = args.test_data
    train_list = args.train_list
    test_list = args.test_list
    train_label = args.train_label
    test_label = args.test_label
    train_loader, test_loader = get_train_test_set(train_data_dir,test_data_dir,train_list,test_list,train_label, test_label,args)

    # load the network
    print("==> Loading the network ...")

    model = SSGRL(image_feature_dim=2048,
                  output_dim=2048, time_step=3,
                  adjacency_matrix=args.graph_file,
                  word_features=args.word_file,
                  num_classes=args.num_classes)

    if args.pretrained:
        model = load_pretrain_model(model, args)
    model.cuda()

    criterion = nn.BCEWithLogitsLoss(reduce=True, size_average=True).cuda()
    for p in model.resnet_101.parameters():
        p.requires_grad=False
    for p in model.resnet_101.layer4.parameters():
        p.requires_grad=True
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=args.lr)
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_epoch, gamma=0.1)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_mAP']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        with torch.no_grad():
            validate(test_loader, model, criterion, 0, args)
        return

    for epoch in range(args.start_epoch,args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        with torch.no_grad():
            mAP = validate(test_loader, model, criterion, epoch, args)
        
        scheduler.step()
        
        # remember best prec@1 and save checkpoint
        is_best = mAP > best_prec1
        best_prec1 = max(mAP, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mAP': mAP,
        }, is_best,args)

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()
    model.resnet_101.eval()
    model.resnet_101.layer4.train()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = torch.tensor(target).cuda()
        input_var = torch.tensor(input).cuda()
        # compute output

        output = model(input_var)
        target = target.float()
        target = target.cuda()
        target_var = torch.autograd.Variable(target)
        loss = criterion(output, target_var)
        losses.update(loss.data, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                  epoch, i, len(train_loader), batch_time=batch_time,
                  loss=losses))

def validate(val_loader, model, criterion,epoch,args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    apMeter = AveragePrecisionMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    x=[]
    for i, (input, target) in enumerate(val_loader):
        target = torch.tensor(target).cuda()
        input_var = torch.tensor(input).cuda()
        output = model(input_var)
        target = target.float()
        target = target.cuda()
        target_var = torch.autograd.Variable(target)
        loss = criterion(output, target_var)
        losses.update(loss.data, input.size(0))

        apMeter.add(output, target)

        mask = (target > 0).float()
        v = torch.cat((output, mask),1)
        x.append(v)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses))
    x = torch.cat(x,0)
    x = x.cpu().detach().numpy()
    print(x.shape)
    np.savetxt(args.post+'_score', x)
    mAP=voc12_mAP(args.post+'_score', args.num_classes)
    print(' * mAP {mAP:.3f}'.format(mAP=mAP))

    averageAP = apMeter.value().mean()
    OP, OR, OF1, CP, CR, CF1 = apMeter.overall()
    OP_K, OR_K, OF1_K, CP_K, CR_K, CF1_K = apMeter.overall_topk(3)

    print('[Test] mAP: {mAP:.3f}, averageAP: {averageAP:.3f}\n'
                '\t\t\t\t\t(Compute with all label) OP: {OP:.3f}, OR: {OR:.3f}, OF1: {OF1:.3f}, CP: {CP:.3f}, CR: {CR:.3f}, CF1:{CF1:.3f}\n'
                '\t\t\t\t\t(Compute with top-3 label) OP: {OP_K:.3f}, OR: {OR_K:.3f}, OF1: {OF1_K:.3f}, CP: {CP_K:.3f}, CR: {CR_K:.3f}, CF1: {CF1_K:.3f}'.format(
                mAP=mAP, averageAP=averageAP,
                OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1, OP_K=OP_K, OR_K=OR_K, OF1_K=OF1_K, CP_K=CP_K, CR_K=CR_K, CF1_K=CF1_K))


    return mAP

def save_checkpoint(state, is_best, args,filename='checkpoint.pth.tar'):
    filename = 'checkpoint_{}.pth.tar'.format(args.post)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_{}.pth.tar'.format(args.post))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__=="__main__":
    main()

