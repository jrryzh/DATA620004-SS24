import os
import logging
import torch
import torchvision.models as models
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

from cubdataset import CUBDataset

# 设置GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 配置日志记录的格式和级别
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 设置TensorBoard日志目录
writer = SummaryWriter('./runs/experiment')

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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

# 定义优化器和损失函数
param_optimizer = list(pretrained_model.named_parameters())
fc_params = [p for n, p in param_optimizer if 'fc' in n]
pretrained_params = [p for n, p in param_optimizer if 'fc' not in n]
fc_learning_rate = 1e-2
pretrained_learning_rate = 1e-3
optimizer = torch.optim.SGD([
    {'params': fc_params, 'lr': fc_learning_rate},
    {'params': pretrained_params, 'lr': pretrained_learning_rate}
], momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

# 训练循环
num_epochs = 50
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
        print('Epoch: {}, Step: {}, Loss: {:.4f}'.format(epoch+1, step+1, loss.item()))
    
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
        outputs = outputs.to('cpu')
        val_loss += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = correct / total
    
    # 在TensorBoard中记录验证loss和accuracy
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    
    print('Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, Acc: {:.4f}'.format(epoch+1, train_loss, val_loss, val_acc))
writer.close()

# 测试
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

test_loss = 0.0
correct = 0
total = 0
for images, labels in test_loader:
    # 移动到GPU
    images = images.to(device)
    labels = labels.to(device)
    # 前向传播
    outputs = pretrained_model(images)
    test_loss += criterion(outputs, labels).item()
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

test_loss /= len(test_loader)
acc = correct / total
print('Test Loss: {:.4f}, Acc: {:.4f}'.format(test_loss, acc))