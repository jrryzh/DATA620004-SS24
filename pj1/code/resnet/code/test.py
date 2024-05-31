import os
import argparse
import logging
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from cubdataset import CUBDataset

# 设置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
GPU_NUM = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

# 创建解析器
parser = argparse.ArgumentParser(description='Test a ResNet model on CUB dataset.')

# 添加参数
parser.add_argument('--model', type=str, default='resnet18', help='name of the model to test')
parser.add_argument('--data_dir', type=str, default='/home/add_disk_e/dataset/CUB_200_2011', help='directory of CUB dataset')
parser.add_argument('--batch_size', type=int, default=8, help='batch size for testing')
parser.add_argument('--dropout_rate', type=float, default=0, help='Dropout rate')
parser.add_argument('--checkpoint_path', type=str, required=True, help='path to the model checkpoint')
# 解析参数
args = parser.parse_args()

# 设置随机种子
torch.manual_seed(42)

# 设置GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义数据集路径和类别数量
dataset_dir = args.data_dir
num_classes = 200

# 定义预训练的CNN模型
if args.model == 'resnet18':
    model = models.resnet18(pretrained=False)
elif args.model == 'resnet50':
    model = models.resnet50(pretrained=False)
elif args.model == 'resnet152':
    model = models.resnet152(pretrained=False)

# 修改输出层
num_ftrs = model.fc.in_features
if args.dropout_rate == 0:
    model.fc = nn.Linear(num_ftrs, num_classes)  # 直接使用线性层
else:
    model.fc = nn.Sequential(
        nn.Dropout(args.dropout_rate),  # 添加dropout层
        nn.Linear(num_ftrs, num_classes)  # 继续使用线性层
    )

# 移动到GPU
model = torch.nn.DataParallel(model)
model.to(device)

# 加载模型权重
model.load_state_dict(torch.load(args.checkpoint_path))
model.eval()

# 定义数据加载器
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1620, 0.1665, 0.1439], std=[0.2654, 0.2697, 0.2550])
])

# 定义数据集
test_dataset = CUBDataset(dataset_dir, split='test', transform=transform)

# 创建数据加载器
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False)

# 定义损失函数
criterion = torch.nn.CrossEntropyLoss()

# 记录测试结果
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        # 移动到GPU
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 计算平均损失和准确率
test_loss /= len(test_loader)
test_acc = correct / total

# 打印测试结果
print('Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(test_loss, test_acc))
