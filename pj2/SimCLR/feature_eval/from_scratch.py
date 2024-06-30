import torch
import sys
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
import argparse
from torchvision import models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch SimCLR Evaluation')
parser.add_argument('-traind', '--train-dataset', default='cifar100',
                    help='dataset name', choices=['stl10', 'cifar10', 'cifar100'])
parser.add_argument('-train_bs', '--train-batch-size', default=256, type=int,)
parser.add_argument('-testd', '--test-dataset', default='cifar100',
                    help='dataset name', choices=['stl10', 'cifar10', 'cifar100'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')

def get_transforms_for_cifar100():
    # 训练数据的转换
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 在32x32的图像周围填充4个像素然后随机裁剪，这样可以增加模型的健壮性
        transforms.RandomHorizontalFlip(),  # 随机水平翻转图像，可以增加数据的多样性
        transforms.ToTensor(),  # 将 PIL 图像或 NumPy ndarray 转换成 FloatTensor，并归一化到[0.0, 1.0]
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # 使用CIFAR-100的均值和标准差归一化
    ])

    # 测试数据的转换
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    return transform_train, transform_test

def get_stl10_data_loaders(download, shuffle=False, batch_size=256):
  train_dataset = datasets.STL10('/home/add_disk/zhangjinyu/dataset/', split='train', download=download,
                                  transform=transforms.ToTensor())

  train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
  
  test_dataset = datasets.STL10('/home/add_disk/zhangjinyu/dataset/', split='test', download=download,
                                  transform=transforms.ToTensor())

  test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  return train_loader, test_loader

def get_cifar10_data_loaders(download, shuffle=False, batch_size=256):
  train_dataset = datasets.CIFAR10('/home/add_disk/zhangjinyu/dataset/', train=True, download=download,
                                  transform=transforms.ToTensor())

  train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
  
  test_dataset = datasets.CIFAR10('/home/add_disk/zhangjinyu/dataset/', train=False, download=download,
                                  transform=transforms.ToTensor())

  test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  return train_loader, test_loader

# def get_cifar100_data_loaders(download, shuffle=False, batch_size=256):
#     train_dataset = datasets.CIFAR100('/home/add_disk/zhangjinyu/dataset/', train=True, download=download,
#                                   transform=transforms.ToTensor())

#     train_loader = DataLoader(train_dataset, batch_size=batch_size,
#                             num_workers=0, drop_last=False, shuffle=shuffle)
    
#     test_dataset = datasets.CIFAR100('/home/add_disk/zhangjinyu/dataset/', train=False, download=download,
#                                   transform=transforms.ToTensor())

#     test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
#                             num_workers=10, drop_last=False, shuffle=shuffle)
  
#     return train_loader, test_loader

def get_cifar100_data_loaders(download, shuffle=False, batch_size=256, dataset_directory='/home/add_disk/zhangjinyu/dataset/'):
    transform_train, transform_test = get_transforms_for_cifar100()
    train_dataset = datasets.CIFAR100(dataset_directory, train=True, download=download,
                                     transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, drop_last=False, shuffle=shuffle)

    test_dataset = datasets.CIFAR100(dataset_directory, train=False, download=download,
                                    transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=2*batch_size, num_workers=10, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    args = parser.parse_args()
    
    writer = SummaryWriter(f"/home/add_disk/zhangjinyu/work_dir/simclr/{args.arch}_{args.train_dataset}_{args.train_batch_size}s_eval-on_{args.test_dataset}_from_scratch/")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    
    if args.arch == 'resnet18':
            model = torchvision.models.resnet18(pretrained=False, num_classes=100).to(device)
    elif args.arch == 'resnet50':
            model = torchvision.models.resnet50(pretrained=False, num_classes=100).to(device)
        
    if args.test_dataset == 'cifar10':
        train_loader, test_loader = get_cifar10_data_loaders(download=True)
    elif args.test_dataset == 'stl10':
        train_loader, test_loader = get_stl10_data_loaders(download=True)
    elif args.test_dataset == 'cifar100':
        train_loader, test_loader = get_cifar100_data_loaders(download=True)
    print("Dataset:", args.test_dataset)
        
    # # freeze all layers but the last fc
    # for name, param in model.named_parameters():
    #     if name not in ['fc.weight', 'fc.bias']:
    #         param.requires_grad = False

    # parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    # assert len(parameters) == 2  # fc.weight, fc.bias 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    epochs = 1000
    n_iter = 0
    for epoch in range(epochs):
        top1_train_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if n_iter % 100 == 0:
                writer.add_scalar('loss', loss, global_step=n_iter)

            n_iter += 1

        top1_train_accuracy /= (counter + 1)
        top1_accuracy = 0
        top5_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
        
            top1, top5 = accuracy(logits, y_batch, topk=(1,5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]
        
        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)
        print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
        writer.add_scalar('acc/train_top1', top1_train_accuracy.item(), epoch)
        writer.add_scalar('acc/test top1', top1_accuracy.item(), epoch)
        writer.add_scalar('acc/test top5', top5_accuracy.item(), epoch)