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
parser = argparse.ArgumentParser(description='Train a ResNet model from scratch on CUB dataset.')

# 添加参数
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (L2 penalty)')
parser.add_argument('--nesterov', default=True, type=bool, help='use Nesterov momentum')
parser.add_argument('--step_size', default=30, type=int, help='step size for learning rate decay')
parser.add_argument('--gamma', default=0.1, type=float, help='learning rate decay factor')
parser.add_argument('--num_epochs', default=300, type=int, help='number of epochs to train')

# 解析参数
args = parser.parse_args()

# 设置随机种子
torch.manual_seed(0)

# 设置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 配置日志记录的格式和级别
logging.basicConfig(filename=f'/share/home/zjy/code_repo/DATA620004-SS24/pj1/code/resnet18/logs/train_from_scratch_{args.num_epochs}epochs_lr{args.lr}_step{args.step_size}_gamma{args.gamma}.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 设置TensorBoard日志目录
writer = SummaryWriter(f'./runs/resnet18_from_scratch_{args.num_epochs}epochs_lr{args.lr}_step{args.step_size}_gamma{args.gamma}')

# 定义数据集路径和类别数量
dataset_dir = '/share/home/zjy/data/CUB_200_2011'
num_classes = 200

# 定义预训练的CNN模型
scratch_model = models.resnet18(pretrained=False)

# 修改输出层
num_ftrs = scratch_model.fc.in_features
scratch_model.fc = torch.nn.Linear(num_ftrs, num_classes)

# 移动到gpu
scratch_model.to(device)

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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

# 定义优化器
optimizer = torch.optim.SGD(scratch_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
# 定义学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
# 定义损失函数
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
        outputs = scratch_model(images)
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        logging.info('Epoch: {}, Step: {}, Loss: {:.4f}'.format(epoch+1, step+1, loss.item()))
    # 学习率更新
    scheduler.step()
    
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
        outputs = scratch_model(images)
        # outputs = outputs.to('cpu')
        val_loss += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # 记录验证loss和accuracy
    val_loss /= len(val_loader)
    val_acc = correct / total
    
    # 保存最优模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience = 0
        torch.save(scratch_model.state_dict(), f'/share/home/zjy/code_repo/DATA620004-SS24/pj1/code/resnet18/ckpts/lr{args.lr}_step{args.step_size}_gamma{args.gamma}_epoch{epoch+1}_valacc{val_acc:.4f}.pth')
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