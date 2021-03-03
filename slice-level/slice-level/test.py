import time
import torch.nn.parallel
import torch.optim
import torch.utils.data
from model import resnet
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from myDataset import CovidDataset



def main():
    arch = 'resnet50'
    BATCH_SIZE = 16
    split_dir = os.path.join(".", "data", "covid_split")
    valid_dir = os.path.join(split_dir, "test")

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    valid_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # 构建MyDataset实例
    valid_data = CovidDataset(data_dir=valid_dir, transform=valid_transform)

    # 构建DataLoder
    val_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    criterion = nn.CrossEntropyLoss().cuda()
    model = torch.nn.DataParallel(resnet.__dict__[arch]())
    PATH="./save_temp/resnet/model.th"
    model.load_state_dict(torch.load(PATH)["state_dict"])
    prec1 = validate(val_loader, model, criterion)



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
