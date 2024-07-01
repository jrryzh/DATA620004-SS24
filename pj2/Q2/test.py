import argparse
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchtoolbox.transform import Cutout
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import yaml
import time
from torch.utils.tensorboard import SummaryWriter
from utils import CutMixCriterion, get_dataloader_cifar100
from models import get_model

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', 
                    type=str,
                    default='config.yaml')
    parser.add_argument('--lr',
                    type=float,
                    default=4e-3)
    parser.add_argument('--num_epochs',
                    type=int,
                    default=200)
    parser.add_argument('--model_save_path',
                    type=int,
                    default=200)
    args = parser.parse_args()
    
    if os.path.exists(args.config):
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
            for key, value in config.items():
                setattr(args, key, value)
    writer = SummaryWriter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.lr = float(args.lr)
    # breakpoint()
    train_dl , val_dl , test_dl = get_dataloader_cifar100(args)
    print(len(train_dl.dataset) , len(val_dl.dataset) , len(test_dl.dataset) )
    
    model = get_model(args)
    model.load_state_dict( torch.load(args.model_save_path) )
    model = model.to(device)
    model.eval()
    # 读 model
    total_correct = 0.0
    total_correctK = 0.0 
    for batch in test_dl:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        
        correct = (torch.max(outputs, dim=1)[1]  #get the indices
                .view(labels.size()) == labels).sum()
        total_correct = total_correct + correct.item()
        
        yresize = labels.view(-1,1)
        _, pred = outputs.topk(args.topK, 1, True, True)
        correctk = torch.eq(pred, yresize).sum()
        total_correctK += correctk.item()
    test_acc = total_correct / len(test_dl.dataset)
    test_acc_TopK = total_correctK / len(test_dl.dataset)
    print(test_acc, test_acc_TopK)
    
    