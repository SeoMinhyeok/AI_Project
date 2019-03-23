# -*- coding: utf-8 -*-
from torch.utils.data.dataset import Dataset
import torch
from cnn_model import CnnModel
from my_dataset import NkDataSet
from tensorboardX import SummaryWriter
import set_variable
import argparse
import time
import os
import torchvision.datasets as mdatset
import torchvision.transforms as transforms
from custom_model import CustomModel


#argparse 커맨드로 부터 인자를 받아서 수행 할 수 있는 기능. fault로 기본값을 지정할 수 도 있다.

parser = argparse.ArgumentParser(description='PyTorch Custom Training')
parser.add_argument('--print_freq', '--p', default=50, type=int, metavar='N',
                    help='number of data loading workers (default: 4')
parser.add_argument('--save-import torch.nn.init as initdir',dest='save_dir',
                    help='The directory used to save the trained models',default='save_layer_load',type=str)
#metavar 는 nickname 같은 역할이라고 생각을 하면 된다.

#dest 입력되는 값이 저장되는 변수

args = parser.parse_args()

#save checkpoint

def save_checkpoint(state,filename='checkpoint.pth.bar'):
    torch.save(state,filename)

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
class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def test(my_dataset_loader, model, model_2, model_3, criterion, epoch, test_writer):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    model_2.eval()
    model_3.eval()
    # model_4.eval()
    # model_5.eval()

    batch_time = AverageMeter()
    end = time.time()

    for i, data in enumerate(my_dataset_loader, 0):
        # Forward pass: Compute predicted y by passing x to the model

        # fc 구조 이기 때문에 일렬로 쫙피는 작업이 필요하다.
        images, label = data

        # 그냥 images 를 하면 에러가 난다. 데이터 shape 이 일치하지 않아서
        y_pred = model(images)
        model_2_pred = model_2(images)
        model_3_pred = model_3(images)
        # model_4_pred = model_4(images)
        # model_5_pred = model_5(images)

        y_pred = (y_pred + model_2_pred + model_3_pred)

        # Compute and print loss
        loss = criterion(y_pred, label)

        output = y_pred.float()
        loss = loss.float()

        prec1 = accuracy(output.data, label)[0]

        # print("prec1", (prec1))
        # print("prec2", (prec_2))

        # print('loss.item', loss)
        # print('real loss item ', loss.item())

        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        #중간 과정을 보는 것
        if i % args.print_freq == 0:
            print('Test : [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                i,len(my_dataset_loader),batch_time=batch_time,loss=losses,top1=top1
            ))
    print('*, epoch : {epoch:.2f} Prec@1 {top1.avg:.3f}'.format(epoch=epoch,top1=top1))


    test_writer.add_scalar('Test/loss', losses.avg, epoch)
    test_writer.add_scalar('Test/accuaracy', top1.avg, epoch)

#Data_Load
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

root = "./"

train_set = mdatset.MNIST(root=root, train=True, transform=trans, download=True)
test_set = mdatset.MNIST(root=root, train=False, transform=trans, download=True)

batch_size = 100

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

# test data set 만들어 줘야 한다.
# Model Load
#input , hiddn, output size

model = CustomModel()
model_2 = CustomModel()
model_3 = CustomModel()
# model_4 = CustomModel()
# model_5 = CustomModel()

checkpoint = torch.load("save_dir/checkpoint_0.tar")
model.load_state_dict(checkpoint["state_dict"])

checkpoint = torch.load('save_dir/checkpoint_3.tar')
model_2.load_state_dict(checkpoint['state_dict'])

checkpoint = torch.load('save_dir/checkpoint_2.tar')
model_3.load_state_dict(checkpoint['state_dict'])

# checkpoint = torch.load("save_dir/checkpoint_3.tar")
# model_4.load_state_dict(checkpoint["state_dict"])

# checkpoint = torch.load("save_dir/checkpoint_4.tar")
# model_5.load_state_dict(checkpoint["state_dict"])

#CrossEntropyLoss 를 사용
criterion = torch.nn.CrossEntropyLoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

writer = SummaryWriter('./log')
test_writer = SummaryWriter('./log/test')

args.save_dir = 'save_dir' # save_dir_2

for epoch in range(500):
    test(test_loader, model, model_2, model_3, criterion, epoch, test_writer)

    save_checkpoint({'epoch': epoch + 1,
                     'state_dict': model.state_dict()
                     }, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch)))
# 의미없음 중복이기 때문에
"""class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count"""