from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import datetime

from networks import *
from torch.autograd import Variable
from summary import summary

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--wd', default=5e-4, type=float, help='weight_decay')
parser.add_argument('--g', default=1, type=int, help='groups in group conv')
parser.add_argument('--k', default=1, type=int, help='kernel size for re1d')
parser.add_argument('--bnr', '-bnr', action='store_true', help='fine-grain batch norm relu')
parser.add_argument('--order', '-o', action='store_true', help='order striding like original')
parser.add_argument('--all', '-a', action='store_true', help='apply re1d to all')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
parser.add_argument('--cosine', '-cos', action='store_true', help='Run cosine learning decay')
args = parser.parse_args()

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
#use_cuda = False
best_acc = 0
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
]) # meanstd transformation

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

if(args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 10
elif(args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 100

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Return network & file name
def getNetwork(args):
    if (args.net_type == 'lenet'):
        net = LeNet(num_classes)
        file_name = 'lenet'
    elif (args.net_type == 'vggnet'):
        net = VGG(args.depth, num_classes)
        file_name = 'vgg-'+str(args.depth)
    elif (args.net_type == 'resnet'):
        net = ResNet(args.depth, num_classes)
        file_name = 'resnet-'+str(args.depth)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor) + \
                    '-drop-'+str(args.dropout)+'-wd-'+str(args.wd)+'-nest-'+str(args.nesterov)
    elif (args.net_type == 'wide-resnet-2d'):
        net = Wide_ResNet_2D(args.depth, args.widen_factor, args.dropout, num_classes)
        file_name = 'wide-resnet-2d-'+str(args.depth)+'x'+str(args.widen_factor) + \
                    '-drop-'+str(args.dropout)+'-wd-'+str(args.wd)+'-nest-'+str(args.nesterov)
    elif (args.net_type == 'wide-resnet-2d-g'):
        net = Wide_ResNet_2D_G(args.depth, args.widen_factor, args.dropout, num_classes, args.g)
        file_name = 'wide-resnet-2d-g-'+str(args.depth)+'x'+str(args.widen_factor) + \
                    '-g'+str(args.g)+'-drop-'+str(args.dropout)+'-wd-'+str(args.wd)+'-nest-'+str(args.nesterov)
    elif (args.net_type == 'wide-resnet-2d-normal'):
        net = Wide_ResNet_2D_Normal(args.depth, args.widen_factor, args.dropout, num_classes)
        file_name = 'wide-resnet-2d-normal-'+str(args.depth)+'x'+str(args.widen_factor) + \
                    '-drop-'+str(args.dropout)+'-wd-'+str(args.wd)+'-nest-'+str(args.nesterov)
    elif (args.net_type == 'wide-resnet-2d-const'):
        net = Wide_ResNet_2D_Const(args.depth, args.widen_factor, args.dropout, num_classes)
        file_name = 'wide-resnet-2d-const-'+str(args.depth)+'x'+str(args.widen_factor) + \
                    '-drop-'+str(args.dropout)+'-wd-'+str(args.wd)+'-nest-'+str(args.nesterov)
    elif (args.net_type == 'wide-resnet-2d-resize-norm'):
        net = Wide_ResNet_2D_Resize_Norm(args.depth, args.widen_factor, args.dropout, num_classes)
        file_name = 'wide-resnet-2d-resize-norm-'+str(args.depth)+'x'+str(args.widen_factor) + \
                    '-drop-'+str(args.dropout)+'-wd-'+str(args.wd)+'-nest-'+str(args.nesterov)
    elif (args.net_type == 'wide-resnet-2d-resize-avg'):
        net = Wide_ResNet_2D_Resize_Avg(args.depth, args.widen_factor, args.dropout, num_classes)
        file_name = 'wide-resnet-2d-resize-avg-'+str(args.depth)+'x'+str(args.widen_factor) + \
                    '-drop-'+str(args.dropout)+'-wd-'+str(args.wd)+'-nest-'+str(args.nesterov)
    elif (args.net_type == 'wide-resnet-re1d'):
        net = Wide_ResNet_RE1D(args.depth, args.widen_factor, args.dropout, num_classes, args.k, args.bnr,
            args.order, args.all, 32, 32)
        file_name = 'wide-resnet-re1d-'+str(args.depth)+'x'+str(args.widen_factor)+\
                    '-drop-'+str(args.dropout)+'-wd-'+str(args.wd)+'-nest-'+str(args.nesterov)+\
                    '-kernel-'+str(args.k)+'-bnr-'+str(args.bnr)+'-order-'+str(args.order)+'-all-'+str(args.all)
    elif (args.net_type == 'wide-resnet-re1d3'):
        net = Wide_ResNet_RE1D3(args.depth, args.widen_factor, args.dropout, num_classes, 32, 32)
        file_name = 'wide-resnet-re1d3-'+str(args.depth)+'x'+str(args.widen_factor) + \
                    '-drop-'+str(args.dropout)+'-wd-'+str(args.wd)+'-nest-'+str(args.nesterov)
    elif (args.net_type == 'wide-resnet-re1d3-slow'):
        net = Wide_ResNet_RE1D3_Slow(args.depth, args.widen_factor, args.dropout, num_classes, 32, 32)
        file_name = 'wide-resnet-re1d3-slow-'+str(args.depth)+'x'+str(args.widen_factor) + \
                    '-drop-'+str(args.dropout)+'-wd-'+str(args.wd)+'-nest-'+str(args.nesterov)
    elif (args.net_type == 'wide-resnet-re1d-slow'):
        net = Wide_ResNet_RE1D_Slow(args.depth, args.widen_factor, args.dropout, num_classes, args.k, 32, 32)
        file_name = 'wide-resnet-re1d-slow-'+str(args.depth)+'x'+str(args.widen_factor) + \
                    '-drop-'+str(args.dropout)+'-wd-'+str(args.wd)+'-nest-'+str(args.nesterov)+\
                    '-kernel-'+str(args.k)
    elif (args.net_type == 'wide-resnet-1d-resize-avg'):
        net = Wide_ResNet_1D_Resize_Avg(args.depth, args.widen_factor, args.dropout, num_classes, 32, 32)
        file_name = 'wide-resnet-1d-resize-avg-'+str(args.depth)+'x'+str(args.widen_factor) + \
                    '-drop-'+str(args.dropout)+'-wd-'+str(args.wd)+'-nest-'+str(args.nesterov)
    elif (args.net_type == 'wide-resnet-1d-resize-avg2'):
        net = Wide_ResNet_1D_Resize_Avg2(args.depth, args.widen_factor, args.dropout, num_classes, 32, 32)
        file_name = 'wide-resnet-1d-resize-avg2-'+str(args.depth)+'x'+str(args.widen_factor) + \
                    '-drop-'+str(args.dropout)+'-wd-'+str(args.wd)+'-nest-'+str(args.nesterov)
    elif (args.net_type == 'wide-resnet-2d-g-resize-avg'):
        net = Wide_ResNet_2D_G_Resize_Avg(args.depth, args.widen_factor, args.dropout, num_classes, args.g)
        file_name = 'wide-resnet-2d-g-resize-avg-'+str(args.depth)+'x'+str(args.widen_factor) + \
                    '-g'+str(args.g)+'-drop-'+str(args.dropout)+'-wd-'+str(args.wd)+'-nest-'+str(args.nesterov)
    elif (args.net_type == 'wide-resnet-2d-g-resize-avg-shuffle'):
        net = Wide_ResNet_2D_G_Resize_Avg_Shuffle(args.depth, args.widen_factor, args.dropout, num_classes, args.g)
        file_name = 'wide-resnet-2d-g-resize-avg-shuffle-'+str(args.depth)+'x'+str(args.widen_factor) + \
                    '-g'+str(args.g)+'-drop-'+str(args.dropout)+'-wd-'+str(args.wd)+'-nest-'+str(args.nesterov)
    elif (args.net_type == 'wide-resnet-2d-g-resize-avg-noshuffle'):
        net = Wide_ResNet_2D_G_Resize_Avg_NoShuffle(args.depth, args.widen_factor, args.dropout, num_classes, args.g)
        file_name = 'wide-resnet-2d-g-resize-avg-noshuffle-'+str(args.depth)+'x'+str(args.widen_factor) + \
                    '-g'+str(args.g)+'-drop-'+str(args.dropout)+'-wd-'+str(args.wd)+'-nest-'+str(args.nesterov)
    elif (args.net_type == 'wide-resnet-avg'):
        net = Wide_ResNet_Avg(args.depth, args.widen_factor, args.dropout, num_classes)
        file_name = 'wide-resnet-avg-'+str(args.depth)+'x'+str(args.widen_factor) + \
                    '-drop-'+str(args.dropout)+'-wd-'+str(args.wd)+'-nest-'+str(args.nesterov)
    elif (args.net_type == 'wide-resnet-2d-dp-resize-avg'):
        net = Wide_ResNet_2D_DP_Resize_Avg(args.depth, args.widen_factor, args.dropout, num_classes)
        file_name = 'wide-resnet-2d-dp-resize-avg-'+str(args.depth)+'x'+str(args.widen_factor) + \
                    '-drop-'+str(args.dropout)+'-wd-'+str(args.wd)+'-nest-'+str(args.nesterov)
    elif (args.net_type == 'wide-resnet-2d-pd-resize-avg'):
        net = Wide_ResNet_2D_PD_Resize_Avg(args.depth, args.widen_factor, args.dropout, num_classes)
        file_name = 'wide-resnet-2d-pd-resize-avg-'+str(args.depth)+'x'+str(args.widen_factor) + \
                    '-drop-'+str(args.dropout)+'-wd-'+str(args.wd)+'-nest-'+str(args.nesterov)
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet / ... ')
        sys.exit(0)

    return net, file_name

# Test only option
if (args.testOnly):
    print('\n[Test Phase] : Model setup')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
    net = checkpoint['net']

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model
    acc = 100.*correct/total
    print("| Test Result\tAcc@1: %.2f%%" %(acc))

    sys.exit(0)

# Model
print('\n[Phase 2] : Model setup')
if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('| Building net type [' + args.net_type + ']...')
    net, file_name = getNetwork(args)
    #print(globals())
    module1 = globals()[net.__module__.replace('networks.','')]    
    #print(module1)
    net.apply(getattr(module1, 'conv_init'))

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

print('\n| Model Information Below... ')
test_in, _ = testset[0]
summary(list(test_in.size()), net)

# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if args.cosine:
        lr = cf.learning_rate_cos(args.lr, epoch)
    else:
        lr = cf.learning_rate(args.lr, epoch)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, 
                            nesterov = args.nesterov, weight_decay=args.wd)

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, lr))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        #with torch.autograd.profiler.profile() as prof:
        #with torch.autograd.profiler.emit_nvtx() as prof:
	if use_cuda:
	    inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
	optimizer.zero_grad()
	inputs, targets = Variable(inputs), Variable(targets)
	outputs = net(inputs)               # Forward Propagation
	loss = criterion(outputs, targets)  # Loss
	loss.backward()  # Backward Propagation
	optimizer.step() # Optimizer update
	
	train_loss += loss.data[0]
	_, predicted = torch.max(outputs.data, 1)
	total += targets.size(0)
	correct += predicted.eq(targets.data).cpu().sum()
	
	sys.stdout.write('\r')
	sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
	%(epoch, num_epochs, batch_idx+1,
	(len(trainset)//batch_size)+1, loss.data[0], 100.*correct/total))
	sys.stdout.flush()
        #print(prof)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model
    acc = 100.*correct/total
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.data[0], acc))

    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
        state = {
                'net':net.module if use_cuda else net,
                'acc':acc,
                'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/'+args.dataset+os.sep
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point+file_name+'.t7')
        best_acc = acc

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))

elapsed_time = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()

    train(epoch)
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))

print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = %.2f%%' %(best_acc))
