import os
import argparse
import logging
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from cubdataset import CUBDataset


# 设置GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
GPU_NUM = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

# 创建解析器
parser = argparse.ArgumentParser(description='Train a ResNet model from pretrained weights on CUB dataset.')

# 添加参数
parser.add_argument('--model', type=str, default='resnet18', help='name of the model to train')
parser.add_argument('--pretrained', action='store_true', help='whether to use pretrained weights')
parser.add_argument('--data_dir', type=str, default='/share/home/zjy/data/CUB_200_2011', help='directory of CUB dataset')
parser.add_argument('--batch_size', type=int, default=8*GPU_NUM, help='batch size for training')
parser.add_argument('--learning_rate', type=float, default=1e-3*GPU_NUM, help='learning rate for training from scratch')
parser.add_argument('--fc_learning_rate', type=float, default=1e-2*GPU_NUM, help='learning rate for fc layers')
parser.add_argument('--pretrained_learning_rate', type=float, default=1e-4*GPU_NUM, help='learning rate for pretrained layers')
parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer to use for training')
parser.add_argument('--scheduler', type=str, default='StepLR', help='scheduler to use for training')
parser.add_argument('--step_size', type=int, default=10, help='step size for StepLR scheduler')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD optimizer')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay for SGD optimizer')
parser.add_argument('--num_epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--augment', action='store_true', help='whether to use data augmentation')
parser.add_argument('--dropout_rate', type=float, default=0, help='Dropout rate')
# 解析参数
args = parser.parse_args()

# 设置随机种子
torch.manual_seed(42)

# 设置GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置filename
filename = f'model_{args.model}+pretrained_{args.pretrained}+fc_lr_{args.fc_learning_rate}+pretrained_lr_{args.pretrained_learning_rate}+momentum_{args.momentum}+augment_{args.augment}+weight_decay_{args.weight_decay}+dropout_{args.dropout_rate}+scheduler_{args.scheduler}+step_size_{args.step_size}+num_epochs_{args.num_epochs}'

# 配置日志记录的格式和级别
logging.basicConfig(filename=f'./logs/' + filename + '.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 设置TensorBoard日志目录
writer = SummaryWriter(f'./runs/' + filename)

# 定义数据集路径和类别数量
dataset_dir = '/home/add_disk_e/dataset/CUB_200_2011'
num_classes = 200

# 定义预训练的CNN模型
if args.model =='resnet18':
    pretrained_model = models.resnet18(pretrained=args.pretrained)
elif args.model =='resnet50':
    pretrained_model = models.resnet50(pretrained=args.pretrained)
elif args.model =='resnet152':
    pretrained_model = models.resnet152(pretrained=args.pretrained)

# 修改输出层
num_ftrs = pretrained_model.fc.in_features
# pretrained_model.fc = torch.nn.Linear(num_ftrs, num_classes)
if args.dropout_rate == 0:
    pretrained_model.fc = nn.Linear(num_ftrs, num_classes)  # 直接使用线性层
else:
    pretrained_model.fc = nn.Sequential(
        nn.Dropout(args.dropout_rate),  # 添加dropout层
        nn.Linear(num_ftrs, num_classes)  # 继续使用线性层
    )

# 移动到gpu
pretrained_model = torch.nn.DataParallel(pretrained_model)
pretrained_model.to(device)

# 定义数据加载器
regular_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1620, 0.1665, 0.1439], std=[0.2654, 0.2697, 0.2550])
])

aug_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    # transforms.RandomVerticalFlip(),  # 随机垂直翻转
    transforms.RandomRotation(10),  # 随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1620, 0.1665, 0.1439], std=[0.2654, 0.2697, 0.2550])
])

if args.augment:
    transform = aug_transform
else:
    transform = regular_transform

# 定义数据集
train_dataset = CUBDataset(dataset_dir, split='train', transform=transform)
test_dataset = CUBDataset(dataset_dir, split='test', transform=transform)

# 分割训练集为训练集和验证集
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False)

# 定义优化器和损失函数
param_optimizer = list(pretrained_model.named_parameters())
if args.pretrained:
    fc_params = [p for n, p in param_optimizer if 'fc' in n]
    pretrained_params = [p for n, p in param_optimizer if 'fc' not in n]
    optimizer = torch.optim.SGD([
        {'params': fc_params, 'lr': args.fc_learning_rate},
        {'params': pretrained_params, 'lr': args.pretrained_learning_rate}
    ], momentum=args.momentum, weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.SGD(pretrained_model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

# 定义学习率衰减策略
scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.1)

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
        print('Epoch: {}, Step: {}, Loss: {:.4f}'.format(epoch+1, step+1, loss.item()))
    
    # 记录训练loss
    train_loss /= len(train_loader)
    writer.add_scalar('Loss/train', train_loss, epoch)
    
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
        torch.save(pretrained_model.state_dict(), f'./ckpts/'+filename+'.pth')
        patience += 1
        if patience == 10:
            logging.info('Early stopping at epoch: {}'.format(epoch+1))
            print('Early stopping at epoch: {}'.format(epoch+1))
            break
        
    # 在TensorBoard中记录验证loss和accuracy
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    
    # 打印日志
    logging.info('Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, Acc: {:.4f}'.format(epoch+1, train_loss, val_loss, val_acc))
    print('Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, Acc: {:.4f}'.format(epoch+1, train_loss, val_loss, val_acc))
    
    # 更新学习率
    scheduler.step()
writer.close()
