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
parser.add_argument('-traind', '--train-dataset', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10', 'cifar100'])
parser.add_argument('-train_bs', '--train-batch-size', default=256, type=int,)
parser.add_argument('-testd', '--test-dataset', default='cifar100',
                    help='dataset name', choices=['stl10', 'cifar10', 'cifar100'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-e', '--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')


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

def get_cifar100_data_loaders(download, shuffle=False, batch_size=256):
    train_dataset = datasets.CIFAR100('/home/add_disk/zhangjinyu/dataset/', train=True, download=download,
                                  transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
    
    test_dataset = datasets.CIFAR100('/home/add_disk/zhangjinyu/dataset/', train=False, download=download,
                                  transform=transforms.ToTensor())

    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  
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
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    
    args = parser.parse_args()
    
    work_dir = f"/home/add_disk/zhangjinyu/work_dir/simclr/{args.arch}_{args.train_dataset}_{args.train_batch_size}/"
    writer = SummaryWriter(f"/home/add_disk/zhangjinyu/work_dir/simclr/{args.arch}_{args.train_dataset}_{args.epochs}ep_{args.train_batch_size}s_eval-on_{args.test_dataset}/")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    
    if args.arch == 'resnet18':
        if args.test_dataset == 'cifar10' or args.test_dataset == 'stl10':
            model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)
        elif args.test_dataset == 'cifar100':
            model = torchvision.models.resnet18(pretrained=False, num_classes=100).to(device)
    elif args.arch == 'resnet50':
        if args.test_dataset == 'cifar10' or args.test_dataset == 'stl10':
            model = torchvision.models.resnet50(pretrained=False, num_classes=10).to(device)
        elif args.test_dataset == 'cifar100':
            model = torchvision.models.resnet50(pretrained=False, num_classes=100).to(device)
    
    checkpoint = torch.load(os.path.join(work_dir, f'checkpoint_0{args.epochs}.pth.tar'), map_location=device)
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        if k.startswith('backbone.'):
            if k.startswith('backbone') and not k.startswith('backbone.fc'):
                # remove prefix
                state_dict[k[len("backbone."):]] = state_dict[k]
        del state_dict[k]
    
    log = model.load_state_dict(state_dict, strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']
    
    if args.test_dataset == 'cifar10':
        train_loader, test_loader = get_cifar10_data_loaders(download=True)
    elif args.test_dataset == 'stl10':
        train_loader, test_loader = get_stl10_data_loaders(download=True)
    elif args.test_dataset == 'cifar100':
        train_loader, test_loader = get_cifar100_data_loaders(download=True)
    print("Dataset:", args.test_dataset)
        
    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=0.0008)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    epochs = 500
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