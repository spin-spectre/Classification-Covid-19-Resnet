import time

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from model import resnet
from myDataset import CovidDataset

import os
import torch
from torch.utils.data import DataLoader

best_prec1=0

MAX_EPOCH = 20
BATCH_SIZE = 32
LR = 0.01
log_interval = 10
val_interval = 1

def main():
    global  best_prec1
    arch = 'resnet50'
    batch_size=128
    epochs=400

    lr=0.1
    momentum=0.9
    workers=4  # workers 是啥
    weight_decay=1e-4
    start_epoch = 0 # 为什么是0 那lr_scheduler那边不是-1？
    print_freq=10
    save_every=10

    save_dir='temp'
    model = torch.nn.DataParallel(resnet.__dict__[arch]())
    model.cuda()
    cudnn.benchmark = True

    split_dir = os.path.join(".", "data", "covid_split")
    train_dir = os.path.join(split_dir, "train")
    valid_dir = os.path.join(split_dir, "valid")

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # 构建MyDataset实例
    train_data = CovidDataset(data_dir=train_dir, transform=train_transform)
    valid_data = CovidDataset(data_dir=valid_dir, transform=valid_transform)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=start_epoch - 1)
    for epoch in range(start_epoch, epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

    ########################### train
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        model.train()
        for i, (input, target) in enumerate(train_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target
            output = model(input_var)
            loss = criterion(output, target_var)
            optimizer.zero_grad() #
            loss.backward()#
            optimizer.step()   #
            output = output.float()
            loss = loss.float()
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, top1=top1))

        ###########################
        lr_scheduler.step()

        # validate
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(save_dir, 'checkpoint.th'))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(save_dir, 'model.th'))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    print_freq=10
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()


            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

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


if __name__ == '__main__':
    main()
