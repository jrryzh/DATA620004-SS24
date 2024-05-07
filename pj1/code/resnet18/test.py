import os
import logging
import torch
import torchvision.models as models
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

from cubdataset import CUBDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# 设置GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# 加载模型权重
pretrained_model.load_state_dict(torch.load('/share/home/zjy/code_repo/DATA620004-SS24/pj1/code/resnet18/ckpt/best_model.pth'))

# 移动到gpu
pretrained_model.to(device)

# 定义数据加载器
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义数据集和数据加载器
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