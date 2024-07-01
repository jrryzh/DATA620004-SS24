import numpy as np

def get_number_of_parameters(model):
    parameters_n = 0
    for parameter in model.parameters():
        parameters_n += np.prod(parameter.shape).item()
    return parameters_n


# * ---------   DATA Loader & DATA Set ------------- *
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, random_split, sampler
from torchvision import transforms
from torchtoolbox.transform import Cutout


def cutmix(batch,alpha=1):
    inputs,labels = batch
    indices = torch.randperm(inputs.size(0))
    shuffled_inputs = inputs[indices]
    shuffled_labels = labels[indices]

    weight,height= inputs.shape[2:]
    lamb = np.random.beta(alpha, alpha)

    rx=np.random.uniform(0, weight)
    ry=np.random.uniform(0, height)

    rw=weight*np.sqrt(1-lamb)
    rh=height*np.sqrt(1-lamb)

    x0 = int(np.round(max(rx - rw / 2, 0)))
    x1 = int(np.round(min(rx + rw / 2, weight)))
    y0 = int(np.round(max(ry - rh / 2, 0)))
    y1 = int(np.round(min(ry + rh / 2, height)))

    inputs[:, :, y0:y1, x0:x1] = shuffled_inputs[:, :, y0:y1, x0:x1]
    labels = (labels, shuffled_labels, lamb)

    return inputs,labels

class CutMixCollator:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = cutmix(batch, self.alpha)
        return batch

class CutMixCriterion:
    def __init__(self, reduction):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def __call__(self, preds, targets):
        targets1, targets2, lam = targets
        return lam * self.criterion(
            preds, targets1) + (1 - lam) * self.criterion(preds, targets2)



def get_dataloader_cifar100(args):
    CIFAR_PATH = args.data_path
    data_augement_type = args.aug_type # cutout , cutmix , random , None
    train_bz = args.train_batch_size
    test_bz = args.test_batch_size
    
    normalize = transforms.Normalize(mean=[125.3/255., 123./255., 113.9/255.],
                                     std=[63./255., 62.1/255., 66.7/255.])    

    data_transforms = transforms.Compose(
        [transforms.ToTensor(),
        normalize])
    
    collator = torch.utils.data.dataloader.default_collate
    if data_augement_type == 'cutout':
       train_transforms = transforms.Compose(
        [Cutout(p=.5, scale=(.1, .3), ratio=(.8, 1/.8), value=(0, 255)),
         transforms.RandomOrder(
             [transforms.RandomResizedCrop(size=32, scale=(.7, 1), ratio=(4/5, 5/4)),
              transforms.RandomHorizontalFlip()
              ]
            ),
         transforms.ToTensor(),
        normalize])
    elif data_augement_type == 'cutmix':
        cutmix_alpha = args.cutmix_alpha 
        train_transforms = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,])
        collator = CutMixCollator(cutmix_alpha)
    
    elif data_augement_type == 'random':
        train_transforms = transforms.Compose(
            [
            transforms.RandomOrder(
                [transforms.RandomResizedCrop(size=32, scale=(.7, 1), ratio=(4/5, 5/4)),
                transforms.RandomHorizontalFlip()
                ]
                ),
            transforms.ToTensor(),
            normalize])
    else:
        train_transforms = data_transforms
    
    
    
    cifar100_train_data = datasets.CIFAR100(root=CIFAR_PATH, train=True, download=True, transform=train_transforms)
    cifar100_valid_data = datasets.CIFAR100(root=CIFAR_PATH, train=True, download=True, transform=data_transforms)
    cifar100_test_data = datasets.CIFAR100(root=CIFAR_PATH, train=False, download=True, transform=data_transforms)
    
    size = len(cifar100_train_data)
    # 二八开  
    _, val_dataset = random_split(cifar100_train_data, [size -  size//10, size//10], generator=torch.Generator().manual_seed(42))
    train_dataset, _ = random_split(cifar100_valid_data, [size -  size//10, size//10], generator=torch.Generator().manual_seed(42))
    
    train_dl = DataLoader(train_dataset, batch_size=train_bz, shuffle=True, num_workers=4,collate_fn=collator)
    val_dl = DataLoader(val_dataset, batch_size=test_bz, shuffle=False, num_workers=4)
    test_dl = DataLoader(cifar100_test_data, batch_size=test_bz, shuffle=False, num_workers=4)
    return train_dl, val_dl ,test_dl 
