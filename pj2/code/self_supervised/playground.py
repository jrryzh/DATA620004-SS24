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

if __name__ == '__main__':
    ckpt_epoch = 200
    batch_size = 256
    lr =  3e-4 # 0.3 # 0.1 # 3e-2 # 3e-6 # 3e-4 # 3e-3
    weight_decay=0.0008
    
    
    # Load model and weights
    resnet = models.resnet18(pretrained=False)
    resnet.fc = nn.Identity()
    model = SimCLR(resnet)
    
    test_input = torch.randn(10, 3, 224, 224)
    model = nn.DataParallel(model).cuda()
    feature, output = model(test_input.cuda())
    import ipdb; ipdb.set_trace()
    print(feature.shape)
    print(output.shape)
    model.load_state_dict(torch.load(f'./ckpt/resnet18_simclr_epoch{ckpt_epoch}.pth'))
    feature, output = model(test_input.cuda())
    print(feature.shape)
    print(output.shape)