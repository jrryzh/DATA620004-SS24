import torch
from PIL import Image
import os
from torchvision.transforms import transforms


class CUBDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        with open(os.path.join(root_dir, 'images.txt'), 'r') as f:
            image_list = f.readlines()
        self.image_list = [x.strip().split(' ')[-1] for x in image_list]
        
        with open(os.path.join(root_dir, 'image_class_labels.txt'), 'r') as f:
            labels = f.readlines()
        self.labels = [int(x.strip().split(' ')[-1]) for x in labels]
        
        train_test_split_file = os.path.join(root_dir, 'train_test_split.txt')
        with open(train_test_split_file, 'r') as f:
            split_list = f.readlines()
        split_list = [int(x.strip().split(' ')[-1]) for x in split_list]

        if split == 'train':
            self.image_list = [x for i, x in enumerate(self.image_list) if split_list[i] == 1]
            self.labels = [x for i, x in enumerate(self.labels) if split_list[i] == 1]
        elif split == 'test':
            self.image_list = [x for i, x in enumerate(self.image_list) if split_list[i] == 0]
            self.labels = [x for i, x in enumerate(self.labels) if split_list[i] == 0]
        else:
            raise ValueError('Invalid split value. Valid values are "train" and "test".')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        label = self.labels[idx]
        img = Image.open(os.path.join(self.root_dir, "images", img_path)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label-1
    
if __name__ == '__main__':
    dataset_dir = '/Users/jrryzh/Documents/lectures/神经网络/pj1/data/CUB_200_2011'
    # 定义数据加载器
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1620, 0.1665, 0.1439], std=[0.2654, 0.2697, 0.2550])
    ])
    dataset = CUBDataset(dataset_dir, split="test", transform=transform)
    print(len(dataset))
    print(dataset[0])