import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from model import SimCLR
from utils import accuracy

import os
import torchlars

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

def train_linear_classifier(model, classifier, train_loader, criterion, optimizer, writer, epochs=300):
    for epoch in range(epochs):
        model.eval()
        classifier.train()
        running_loss = 0.0
        top1_correct = 0
        top5_correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                features, _ = model(images)
            optimizer.zero_grad()
            outputs = classifier(features)
            # import ipdb; ipdb.set_trace()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            _, top1_pred = outputs.topk(1, 1, True, True)
            _, top5_pred = outputs.topk(5, 1, True, True)

            top1_correct += top1_pred.eq(labels.view(-1, 1).expand_as(top1_pred)).sum().item()
            top5_correct += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        top1_acc = 100 * top1_correct / total
        top5_acc = 100 * top5_correct / total

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Top-1 Accuracy: {top1_acc:.2f}%, Top-5 Accuracy: {top5_acc:.2f}%')

        # Write to TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/top1_train', top1_acc, epoch)
        writer.add_scalar('Accuracy/top5_train', top5_acc, epoch)

def test(model, classifier, test_loader):
    model.eval()
    classifier.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            features, _ = model(images)
            outputs = classifier(features)
            _, top1_pred = outputs.topk(1, 1, True, True)
            _, top5_pred = outputs.topk(5, 1, True, True)
            
            top1_correct += top1_pred.eq(labels.view(-1, 1).expand_as(top1_pred)).sum().item()
            top5_correct += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
            total += labels.size(0)
    
    top1_acc = 100 * top1_correct / total
    top5_acc = 100 * top5_correct / total
    
    print(f'Top-1 Accuracy: {top1_acc}%')
    print(f'Top-5 Accuracy: {top5_acc}%')

if __name__ == '__main__':
    ckpt_epoch = 200
    batch_size = 256
    lr =  3e-2 # 0.3 # 0.1 # 3e-2 # 3e-6 # 3e-4 # 3e-3
    weight_decay=0.0008
    model_name = "resnet18" # "resnet50"
    dataset_name = "STL10" # "CIFAR100" # "STL10"
    
    # Load model and weights
    if model_name == "resnet18":
        resnet = models.resnet18(pretrained=False)
        # resnet.fc = nn.Identity()
        model = SimCLR(resnet)
        model = model.cuda()
        model.load_state_dict(torch.load(f'./ckpt/resnet18_simclr_stl10_epoch{ckpt_epoch}.pth'))
    elif model_name == "resnet50":
        resnet = models.resnet50(pretrained=False)
        # resnet.fc = nn.Identity()
        model = SimCLR(resnet)
        model = model.cuda()
        model.load_state_dict(torch.load(f'./ckpt/resnet50_simclr_stl10_epoch{ckpt_epoch}.pth'))
    
    # Load CIFAR-100 data
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    if dataset_name == "CIFAR100":
        cifar100_train = datasets.CIFAR100(root='/home/add_disk/zhangjinyu/dataset/cifar100', train=True, download=True, transform=transform)
        cifar100_test = datasets.CIFAR100(root='/home/add_disk/zhangjinyu/dataset/cifar100', train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(cifar100_train, batch_size=batch_size, num_workers=8, shuffle=True)
        test_loader = torch.utils.data.DataLoader(cifar100_test, batch_size=256, num_workers=8, shuffle=False)
    elif dataset_name == "STL10":
        stl10_train = datasets.STL10(root='/home/add_disk/zhangjinyu/dataset/stl10', split='train', download=True, transform=transform)
        stl10_test = datasets.STL10(root='/home/add_disk/zhangjinyu/dataset/stl10', split='test', download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(stl10_train, batch_size=batch_size, num_workers=8, shuffle=True)
        test_loader = torch.utils.data.DataLoader(stl10_test, batch_size=256, num_workers=8, shuffle=False)
    
    # Freeze all layers of ResNet-18, only train a linear classifier
    for param in model.parameters():
        param.requires_grad = False

    # Wrap the model with DataParallel for multi-GPU training
    # linear_classifier = nn.DataParallel(nn.Linear(512, 100)).cuda()
    linear_classifier = nn.Linear(512, 10).cuda()
    
    criterion = nn.CrossEntropyLoss().cuda()
    base_optimizer = optim.Adam(linear_classifier.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = optim.Lars

    # Initialize TensorBoard writer
    log_name = f'LinearEval_lr_{lr}_wd_{weight_decay}_bs_{batch_size}_epoch_{ckpt_epoch}'
    writer = SummaryWriter(log_dir=f'./runs/LinearEval/{log_name}')

    train_linear_classifier(model, linear_classifier, train_loader, criterion, optimizer, writer)
    test(model, linear_classifier, test_loader)

    writer.close()
