import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt

# 直接加载预训练的ResNet-18
supervised_model = models.resnet18(pretrained=True)
supervised_model.fc = nn.Linear(512, 100)
supervised_model = supervised_model.cuda()

optimizer = optim.Adam(supervised_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train_supervised(model, train_loader, criterion, optimizer, epochs=100):
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

train_supervised(supervised_model, train_loader, criterion, optimizer)
test(supervised_model, linear_classifier, test_loader)
