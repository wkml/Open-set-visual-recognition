import sys
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
from utils.metrics import AveragePrecisionMeter, voc12_mAP
from model.models import SSGRL
from utils.checkpoint import save_code_file, save_checkpoint

from tensorboardX import SummaryWriter
import logging
from config import arg_parse, logger, show_args

global best_prec1
best_prec1 = 0


def main():
    global best_prec1
    args = arg_parse()

    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    file_path = 'exp/log/{}.log'.format(args.post)
    file_handler = logging.FileHandler(file_path)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Show Argument
    show_args(args)

    # Save Code File
    # save_code_file(args)

    # Create dataloader
    logger.info("==> Creating dataloader...")
    train_data_dir = args.train_data
    test_data_dir = args.test_data
    train_list = args.train_list
    test_list = args.test_list
    train_label = args.train_label
    test_label = args.test_label
    train_loader, test_loader = get_train_test_set(train_data_dir,test_data_dir,train_list,test_list,train_label, test_label,args)
    logger.info("==> Done!\n")

    # load the network
    logger.info("==> Loading the network ...")

    model = SSGRL(args=args,
                    adjacency_matrix=args.graph_file,
                    word_features=args.word_file,
                    num_classes=args.num_classes,
                    image_feature_dim=2048,
                    output_dim=2048,
                    time_step=3,
                    )
    model.cuda()

    for p in model.parameters():
        p.requires_grad = True
    for p in model.clip_model.parameters():
        p.requires_grad = False

    criterion = nn.BCEWithLogitsLoss(reduce=True, size_average=True).cuda()
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_epoch, gamma=0.1)

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_mAP']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    logger.info("==> Done!\n")

    if args.evaluate:
        with torch.no_grad():
            validate(test_loader, model, criterion, 0, args)
        return

    logger.info("Run Experiment...")
    writer = SummaryWriter('{}/{}'.format('exp/summary/', args.post))

    for epoch in range(args.start_epoch,args.epochs):
        # model train mode
        train(train_loader, model, criterion, optimizer, epoch, args, writer)

        # evaluate on validation set
        with torch.no_grad():
            # model eval mode
            mAP = validate(test_loader, model, criterion, epoch, args)
        
        scheduler.step()
        writer.add_scalar('mAP', mAP, epoch)
        
        # remember best prec@1 and save checkpoint
        is_best = mAP > best_prec1
        best_prec1 = max(mAP, best_prec1)

        save_checkpoint(args, {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_mAP': mAP,
        }, is_best)

        if is_best:
            logger.info('[Best] [Epoch {0}]: Best mAP is {1:.3f}'.format(epoch, best_prec1))
            
    writer.close()

def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    model.train()
    end = time.time()
    model.clip_model.eval()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input, target = input.cuda(), target.float().cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)
        losses.update(loss.data, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                         epoch, i, len(train_loader), batch_time=batch_time,
                         loss=losses))
    writer.add_scalar('Loss', losses.avg, epoch)

def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    apMeter = AveragePrecisionMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    x=[]
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.float().cuda()

        output = model(input)

        loss = criterion(output, target)
        losses.update(loss.data, input.size(0))

        apMeter.add(output, target)

        mask = (target > 0).float()
        v = torch.cat((output, mask),1)
        x.append(v)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses))
    x = torch.cat(x,0)
    x = x.cpu().detach().numpy()
    np.savetxt(args.post+'_score', x)
    mAP=voc12_mAP(args.post+'_score', args.num_classes)

    averageAP = apMeter.value().mean()
    OP, OR, OF1, CP, CR, CF1 = apMeter.overall()
    OP_K, OR_K, OF1_K, CP_K, CR_K, CF1_K = apMeter.overall_topk(3)

    logger.info('[Test] mAP: {mAP:.3f}, averageAP: {averageAP:.3f}\n'
                '\t\t\t\t\t(Compute with all label) OP: {OP:.3f}, OR: {OR:.3f}, OF1: {OF1:.3f}, CP: {CP:.3f}, CR: {CR:.3f}, CF1:{CF1:.3f}\n'
                '\t\t\t\t\t(Compute with top-3 label) OP: {OP_K:.3f}, OR: {OR_K:.3f}, OF1: {OF1_K:.3f}, CP: {CP_K:.3f}, CR: {CR_K:.3f}, CF1: {CF1_K:.3f}'.format(
                mAP=mAP, averageAP=averageAP,
                OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1, OP_K=OP_K, OR_K=OR_K, OF1_K=OF1_K, CP_K=CP_K, CR_K=CR_K, CF1_K=CF1_K))

    return mAP

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

