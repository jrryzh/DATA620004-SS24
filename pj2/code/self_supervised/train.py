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

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

    
# 数据预处理
class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

# SimCLR训练
def train_simclr(model, dataset_name, dataloader, optimizer, scheduler, epochs=500):
    total_iter = 0
    for epoch in range(epochs):
        model.train()
        epoch_iter, epoch_loss = 0, 0
        for (images_1, images_2), _ in dataloader:
            # import ipdb; ipdb.set_trace()
            images_1 = images_1.cuda()
            images_2 = images_2.cuda()

            optimizer.zero_grad()
            _, z1 = model(images_1)
            _, z2 = model(images_2)
            z = torch.cat([z1, z2], dim=0)
            loss, logits, labels = compute_contrastive_loss(z)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            total_iter += 1
            epoch_iter += 1
            
            if total_iter % 100 == 0:
                writer.add_scalar('Loss/train', epoch_loss / epoch_iter, total_iter)  # 记录训练损失
                writer.add_scalar('Lr', scheduler.get_lr()[0], total_iter)  # 记录学习率
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                print(f'Epoch [{epoch+1}/{epochs}], Step [{total_iter}], Lr: {scheduler.get_lr()[0]}, Loss: {epoch_loss / len(dataloader)}, acc/top1: {top1[0]}, acc/top5: {top5[0]}')
        
        # 调整学习率
        if epoch >= 10:
            scheduler.step()
            
        # 每隔save_every个epoch保存一次模型
        save_every = 20
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            torch.save(model.state_dict(), f'./ckpt/resnet50_simclr_{dataset_name}_epoch{epoch+1}.pth')
            print(f"Model saved at epoch {epoch+1}")
        
        # 每个epoch结束时，记录当前epoch的平均损失到TensorBoard
        writer.add_scalar('Epoch_Loss/train', epoch_loss / len(dataloader), epoch)
        
    writer.close()

# 定义自监督学习的NT-Xent损失函数
def compute_contrastive_loss(z, temperature=0.07):
    batch_size = z.shape[0] // 2  # 获取批量大小的一半，即正样本和负样本各一半
    
    # 对嵌入进行归一化
    z = F.normalize(z, dim=1)
    
    # 计算相似度矩阵
    similarity_matrix = torch.matmul(z, z.T) / temperature
    
    # 创建标签
    labels = torch.arange(batch_size).cuda()  # 创建范围从0到batch_size-1的标签
    labels = torch.cat([labels, labels], dim=0)  # 标签扩展到两倍大小
    
    # 创建掩码以排除自相似度
    mask = torch.eye(batch_size * 2, dtype=torch.bool).cuda()  # 创建对角为True的掩码
    
    # 提取正样本对
    positive_pairs = similarity_matrix[mask].view(batch_size * 2, -1)  # 提取对角线上的正样本对
    
    # 重塑相似度矩阵用于交叉熵损失
    similarity_matrix = similarity_matrix.masked_select(~mask).view(batch_size * 2, batch_size * 2 - 1)  # 排除对角线，重塑为适合交叉熵损失的形状
    
    # 创建logits: 包含正样本和所有负样本对的相似度分数
    logits = torch.cat([positive_pairs, similarity_matrix], dim=1)
    
    # 为交叉熵损失创建标签
    labels = torch.zeros(batch_size * 2, dtype=torch.long).cuda()  # 创建全零标签，表示正样本对
    
    # 计算交叉熵损失
    loss = nn.CrossEntropyLoss()(logits, labels)
    
    return loss, logits, labels  # 返回损失值

if __name__ == '__main__':
    # 选择数据集
    dataset_name = "stl10"
    
    # 初始化TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir='./runs/SimCLR_experiment_resnet50')

    # 定义数据预处理
    if dataset_name == 'cifar10':
        s, size = 1, 32
    elif dataset_name =='stl10':
        s, size = 1, 96
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            # GaussianBlur(kernel_size=int(0.1 * size)),
                                            transforms.GaussianBlur(kernel_size=int(0.1 * size), sigma=(0.1, 2.0)),
                                            transforms.ToTensor()])

    # 加载CIFAR-10数据集
    # dataset = datasets.CIFAR10(root='/home/add_disk/zhangjinyu/dataset/cifar10', train=True, download=True, transform=ContrastiveLearningViewGenerator(data_transforms, 2))
    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR100(root='/home/add_disk/zhangjinyu/dataset/', train=True, download=True, transform=ContrastiveLearningViewGenerator(data_transforms, 2))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=12, shuffle=True, pin_memory=True, drop_last=True)
    elif dataset_name == 'stl10':
        dataset = datasets.STL10(root='/home/add_disk/zhangjinyu/dataset/', split='unlabeled', download=True, transform=ContrastiveLearningViewGenerator(data_transforms, 2))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=12, shuffle=True, pin_memory=True, drop_last=True)

    # 加载ResNet-18并移除分类层
    # resnet = models.resnet18(pretrained=False)
    resnet = models.resnet50(pretrained=False)
    model = SimCLR(resnet)

    # 加载到cuda
    model = model.cuda()

    # 定义损失函数和优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader), eta_min=0, last_epoch=-1)
    
    # 训练模型
    train_simclr(model, dataset_name, dataloader, optimizer, scheduler)