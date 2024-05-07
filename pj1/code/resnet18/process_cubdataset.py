import torch
from torchvision import datasets, transforms
from cubdataset import CUBDataset

# 定义转换，将图像转换为Tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.0,), (1.0,))  # 这里使用一个简单的归一化，只是为了将图像转换为张量
])

# 加载数据集
train_dataset = CUBDataset(root_dir='/share/home/zjy/data/CUB_200_2011', split='train', transform=transform)
test_dataset = CUBDataset(root_dir='/share/home/zjy/data/CUB_200_2011', split='test', transform=transform)

# 初始化用于累加的变量
channels_sum = torch.zeros(3)
channels_sqr_sum = torch.zeros(3)

# 用于计数的变量
num_samples = 0

# 遍历数据集中的所有图像
for data, _ in train_dataset:
    # 累加像素值和平方和
    channels_sum += torch.mean(data, [1, 2])
    channels_sqr_sum += torch.mean(data ** 2, [1, 2])
    num_samples += data.size(0)  # 增加样本数量
for data, _ in test_dataset:
    # 累加像素值和平方和
    channels_sum += torch.mean(data, [1, 2])
    channels_sqr_sum += torch.mean(data ** 2, [1, 2])
    num_samples += data.size(0)  # 增加样本数量

# 计算均值和标准差
mean = channels_sum / num_samples
std = (channels_sqr_sum / num_samples - mean ** 2) ** 0.5

print(f'Mean: {mean}')
print(f'Std: {std}')