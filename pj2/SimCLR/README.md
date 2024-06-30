# PyTorch SimCLR: A Simple Framework for Contrastive Learning of Visual Representations

## 介绍

SimCLR 是一种简单而有效的对比学习方法，它可以训练一个模型来学习到图像的共现特征。SimCLR 由两个网络组成，一个是编码器网络，它将输入图像编码为一个低维的特征向量，另一个是解码器网络，它将编码后的特征向量解码为原始图像。然后，两个网络的输出通过一个线性层进行对比，以计算损失函数。

SimCLR 的主要优点是：

1. 简单：SimCLR 仅使用两个网络，不需要复杂的架构，因此可以快速训练。
2. 有效：SimCLR 使用了两个网络，因此可以捕捉到图像的共现特征。
3. 无监督：SimCLR 无需标签信息，因此可以训练无监督的模型。

这里的代码主要实现了SimCLR的训练和评估，同时作为对比添加了从零训练的ResNet18代码和基于imagenet预训练的ResNet18的线性评估代码。

## 安装环境

```
$ conda env create --name simclr --file env.yml
$ conda activate simclr
$ python run.py
```

## 训练模型


```python

$ python run.py -dataset-name cifar100 -b 512

```

## 评估模型

```python

$ python feature_eval/eval.py -a resnet18 -traind cifar100 -testd cifar100 -train_bs 256 -e 500

```

## 从零训练的ResNet18

```python

$ python feature_eval/from_scratch.py

```

## 基于imagenet预训练的ResNet18

```python

$ python feature_eval/eval_imagenet.py

```