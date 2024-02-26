# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        # 第一个卷积层：输入通道数为3（彩色图像），输出通道数为6，卷积核大小为5
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # 最大池化层：核大小为2，步长为2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二个卷积层：输入通道数为6，输出通道数为16，卷积核大小为5
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # 全连接层：根据卷积层和池化层的输出调整输入维度
        self.fc1 = nn.Linear(16 * 5 * 5, 256)  
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        shape_dict = {}
        # 通过第一个卷积层、ReLU激活层和池化层
        x = self.pool(self.relu(self.conv1(x)))
        shape_dict['conv1'] = x.shape

        # 通过第二个卷积层、ReLU激活层和池化层
        x = self.pool(self.relu(self.conv2(x)))
        shape_dict['conv2'] = x.shape

        # 展平操作，为全连接层做准备
        x = x.view(-1, 16 * 5 * 5) 
        shape_dict['flatten'] = x.shape

        # 通过全连接层
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)
        return out, shape_dict


def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    # model_params = 0.0
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model_params


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
