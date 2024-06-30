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
from utils import CutMixCriterion, get_dataloader_cifar100, get_number_of_parameters
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

    def test_LayerNorm():
        import torch
        import torch.nn as nn
        
        a = torch.randn(2,5,3) # batch , seq, channel
        layer_norm = nn.LayerNorm(3)
        batch_norm = nn.BatchNorm1d()
        
        
    
    model = get_model(args)
    print(model)
    print(get_number_of_parameters(model=model))
    # assert False
    model = model.to(device)
    
    loss_func_mix = CutMixCriterion(reduction='sum') 
    loss_func = nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs , 0)

    os.makedirs('./model_save' , exist_ok=True )

    start_time=time.time()
    best_acc = None
    ACC_EARLY_STOP_NUM = 0 # 超过 K=20 次下降，则停掉
    for epoch in range(args.num_epochs):
        model.train()
        total_loss=0.0
        # train
        for batch in train_dl:
            optimizer.zero_grad() 
            inputs, labels = batch
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            if args.aug_type == 'cutmix':
                labels, shuffled_labels, lamb=labels
                labels=(labels.to(device), shuffled_labels.to(device), lamb)
                loss = loss_func_mix(outputs, labels)
            else:
                labels = labels.to(device) 
                loss = loss_func(outputs, labels)

            loss.backward()  
            optimizer.step()
            total_loss = total_loss + loss.item()
        train_loss = total_loss / len( train_dl.dataset )
        print(f"epoch : {epoch} , train loss: {train_loss}")
        writer.add_scalar(f"Loss/train_{args.aug_type}", train_loss, epoch)
        # tensorboard xxx

        scheduler.step()        
        # eval on `validation set`
        model.eval()
        total_loss = 0
        total_correct = 0.0
        for batch in val_dl:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)  
            outputs = model(inputs)
            

            loss = loss_func(outputs, labels)
            total_loss += loss.item()
            
            correct = (torch.max(outputs, dim=1)[1]  #get the indices
                    .view(labels.size()) == labels).sum()
            total_correct = total_correct + correct.item()    
        
        valid_loss = total_loss / len(val_dl.dataset)
        valid_acc = total_correct / len(val_dl.dataset)
        writer.add_scalar(f"Loss/valid_{args.aug_type}", valid_loss, epoch)
        writer.add_scalar(f"Acc/valid_{args.aug_type}", valid_acc, epoch)
        print(f"epoch : {epoch} , valid loss: {valid_loss} , valid acc: {valid_acc}")
        print('* ------------------------------ * ')
        
        if best_acc is None:
            best_acc = valid_acc
        elif best_acc < valid_acc:
            best_acc = valid_acc
            ACC_EARLY_STOP_NUM = 0
            # save model             
            if epoch >= 36: # 30 
                model_save_path = f'./model_save/model_{args.model_name}_{args.aug_type}_epoch_{epoch}.pth'
                torch.save( model.state_dict(), model_save_path)
                
        else: # early stop in case of over-fitting
            ACC_EARLY_STOP_NUM += 1
            if ACC_EARLY_STOP_NUM == args.early_stop_num: break # 连续K轮不升，则停
            
    # test_dl
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
    
    
    model = get_model(args)
    model.load_state_dict( torch.load(model_save_path) )
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
    
    with open( f'model_{args.model_name}_{args.aug_type}_test.txt' , 'a+' , encoding='utf-8') as f:
        f.write('acc: {}'.format(test_acc) )
        f.write('\n')
        f.write('TopK acc: {}'.format(test_acc_TopK) )
    


    

        
    
    
    
    
    
    
    
    