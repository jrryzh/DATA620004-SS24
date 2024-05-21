import os
import argparse
import logging
import torch
import torchvision.models as models
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

from cubdataset import CUBDataset

# 创建解析器
parser = argparse.ArgumentParser(description='Train a ResNet model from pretrained weights on CUB dataset.')

# 添加参数
parser.add_argument('--fc_learning_rate', type=float, default=1e-6, help='learning rate for fc layers')
parser.add_argument('--pretrained_learning_rate', type=float, default=1e-5, help='learning rate for pretrained layers')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD optimizer')
parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs to train')
# 解析参数
args = parser.parse_args()

# 设置随机种子
torch.manual_seed(0)

# 设置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

# 设置GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 配置日志记录的格式和级别
logging.basicConfig(filename=f'/share/home/zjy/code_repo/DATA620004-SS24/pj1/code/resnet18/logs/train_from_pretrained_fc_lr_{args.fc_learning_rate}_pretrained_lr_{args.pretrained_learning_rate}_momentum_{args.momentum}_num_epochs_{args.num_epochs}.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 设置TensorBoard日志目录
writer = SummaryWriter(f'./runs/resnet18_from_pretrained_fc_lr_{args.fc_learning_rate}_pretrained_lr_{args.pretrained_learning_rate}_momentum_{args.momentum}_num_epochs_{args.num_epochs}')

# 定义数据集路径和类别数量
dataset_dir = '/share/home/zjy/data/CUB_200_2011'
num_classes = 200

# 定义预训练的CNN模型
pretrained_model = models.resnet18(pretrained=True)

# 修改输出层
num_ftrs = pretrained_model.fc.in_features
pretrained_model.fc = torch.nn.Linear(num_ftrs, num_classes)

# 移动到gpu
pretrained_model.to(device)

# 定义数据加载器
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1620, 0.1665, 0.1439], std=[0.2654, 0.2697, 0.2550])
])

# 定义数据集
train_dataset = CUBDataset(dataset_dir, split='train', transform=transform)
test_dataset = CUBDataset(dataset_dir, split='test', transform=transform)

# 分割训练集为训练集和验证集
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=8, shuffle=False)

# 定义优化器和损失函数
param_optimizer = list(pretrained_model.named_parameters())
fc_params = [p for n, p in param_optimizer if 'fc' in n]
pretrained_params = [p for n, p in param_optimizer if 'fc' not in n]
optimizer = torch.optim.SGD([
    {'params': fc_params, 'lr': args.fc_learning_rate},
    {'params': pretrained_params, 'lr': args.pretrained_learning_rate}
], momentum=args.momentum)
criterion = torch.nn.CrossEntropyLoss()

# 训练循环
num_epochs = args.num_epochs
best_val_acc = 0.0
patience = 0
for epoch in range(num_epochs):
    train_loss = 0.0
    for step, (images, labels) in enumerate(train_loader):
        # 移动到GPU
        images = images.to(device)  
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = pretrained_model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        logging.info('Epoch: {}, Step: {}, Loss: {:.4f}'.format(epoch+1, step+1, loss.item()))
    
    # 记录训练loss
    train_loss /= len(train_loader)
    writer.add_scalar('train_loss', train_loss, epoch)
    
    # 验证
    val_loss = 0.0
    correct = 0
    total = 0
    for images, labels in val_loader:
        # 移动到GPU
        images = images.to(device)
        labels = labels.to(device)
        # 前向传播
        outputs = pretrained_model(images)
        # outputs = outputs.to('cpu')
        val_loss += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = correct / total
    
    # 保存最优模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience = 0
        torch.save(pretrained_model.state_dict(), f'/share/home/zjy/code_repo/DATA620004-SS24/pj1/code/resnet18/ckpts/train_from_pretrained_fc_lr_{args.fc_learning_rate}_pretrained_lr_{args.pretrained_learning_rate}_momentum_{args.momentum}_num_epochs_{args.num_epochs}_best_val_acc_{val_acc:.4f}.pth')
    else:
        patience += 1
        if patience == 10:
            logging.info('Early stopping at epoch: {}'.format(epoch+1))
            break
        
    # 在TensorBoard中记录验证loss和accuracy
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    
    # 打印日志
    logging.info('Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, Acc: {:.4f}'.format(epoch+1, train_loss, val_loss, val_acc))
writer.close()