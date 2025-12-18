# **Experiment7--ResNet_SVHN** 
(实验7-ResNet实现SVHN-街道实景门牌数据集分类)
##### 本实验比较了三种残差块设计（恒等映射块、投影块和全映射块）在 SVHN 数据集上的准确率、计算资源耗时。
##

## 1.实验目的
###### 1.掌握ResNet架构卷积神经网络的基本原理。
###### 2.利用ResNet架构对SVHN-街道实景门牌数据集进行分类训练和测试。
###### 3.构建不同残差块结构（恒等映射块、投影块、全映射块）并对比分析他们的功能和实验效果。

##

## 2.实验内容
###### 为对比不同残差块的性能差别，搭建了包含 8 个残差块的轻量化 ResNet 网络，用 SVHN 数据集完成训练和测试；采用 32 维通道数，模拟简单场景下的残差网络；尝试选取不同规模大小的训练数据，对比恒等映射块、投影块、全映射块这三种残差块在参数量、计算时间开销上的差异，以及各自的分类准确率和损失变化。
##
#### 2.1.SVHN 数据集加载及处理
##
##### （1）导入Pytorch和相关的工具库
```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import Subset
from torchsummary import summary
import numpy as np
```
##
##### （2）设置训练批次大小和运行设备，加载 SVHN 数据集并进行图像进行归一化、尺寸调整处理；设置随机种子随机选取数据集子集作为训练集和测试集，保证实验可复现性。
```
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 适配轻量化ResNet输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])  # SVHN数据集归一化
])
# 加载SVHN完整训练集和测试集
train_full = datasets.SVHN('data', split='train', download=True, transform=transform)
test_full = datasets.SVHN('data', split='test', download=True, transform=transform)

# 控制数据量：选取全量数据的一定比例
n = 1  # n=1使用全量数据，可调整为更大值减少数据量
rng = np.random.default_rng(42)  # 固定随机种子
train_idx = rng.choice(len(train_full), len(train_full)//n, replace=False)
test_idx = rng.choice(len(test_full), len(test_full)//n, replace=False)

# 构建数据加载器
train_loader = torch.utils.data.DataLoader(Subset(train_full, train_idx), 
                                           batch_size=batch_size, shuffle=True)
```

##
#### 2.2.构建ResNet网络并设置不同残差模块
##### 定义三种不同的残差块（恒等映射块、投影块、全映射块），适配 SVHN 10 分类任务。
##
##### 1.恒等映射块
```
class IdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IdentityBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.channel_pad = out_channels - in_channels

    def forward(self, x):
        residual = x
        # 通道不匹配时零填充
        if self.channel_pad > 0:
            residual = torch.cat([residual, torch.zeros_like(residual)[:, :self.channel_pad, :, :].to(device)], dim=1)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        return out
```
##
##### 2.投影块，1×1卷积
```
class ProjectionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ProjectionBlock, self).__init__()
        # 主分支
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 捷径分支（1×1卷积投影）
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        return out
```
##
##### 3.全映射块，3×3卷积+激活
```
class FullMappingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FullMappingBlock, self).__init__()
        # 主分支
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 捷径分支（3×3卷积+激活）
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        return out
```
##
#### 2.3.Model Training and Evaluation
##### Configure the number of training epochs and demonstrate the procedure, including the forward pass, loss computation, and backpropagation.
###### 设置训练轮数并且展示数据训练过程的前向传播，损失计算，反向传播等流程。
###### Training Process
```
epochs=10
accs,losses=[],[]
for epoch in range(epochs):
    for batch_idx,(x,y) in enumerate(trainloader):
        x,y=x.to(device),y.to(device)
        optimizer.zero_grad()
        out = model(x)
        out = outputs.logits  # 主分类输出
        loss = F.cross_entropy(out,y)
        loss.backward()
        optimizer.step()
```
###### Testing Process
```
with torch.no_grad():
        for batch_idx,(x,y) in enumerate(testloader):
            x,y = x.to(device),y.to(device)
            out = model(x)
            total_loss +=F.cross_entropy(out,y).item()
            correct +=(out.argmax(1)==y).sum().item()
```
##
#### 2.4.Introduction to the Network Architecture(网络架构介绍)
##
##### (1)Results of the dataset
下载好的数据集的格式
```
train_full
```
##
```
Dataset FashionMNIST
    Number of datapoints: 60000
    Root location: data
    Split: Train
    StandardTransform
Transform: Compose(
               Resize(size=299, interpolation=bilinear, max_size=None, antialias=True)
               Grayscale(num_output_channels=3)
               ToTensor()
           )
```
##
随机选取完的训练集格式
```
train_loader
```
##
```
<torch.utils.data.dataloader.DataLoader at 0x15963dcea40>
```
##
##### (2)Introduction to the Output of the InceptionV3 Model(InceptionV3的model输出介绍)
总体结构图：
```
输入图片（299x299x3）
↓ 浅层卷积+池化（Conv2d_1a,maxpool2）
↓ InceptionA（Mixed_5b/5c/5d）[此处bcd是指3个InceptionA模块]
↓ InceptionB（Mixed_6a）
↓ InceptionC（Mixed_6b/6c/6d/6e）[此处bcde是指4个InceptionC模块]
↓ InceptionD（Mixed_7a）
↓ InceptionE（Mixed_7b/7c）[此处bc是指2个InceptionE模块]
↓ 池化+全连接 → 输出10分类结果
```
##
分别介绍InceptionA，B，C，D，E的结构和功能：
###### InceptionA:
```
1个 InceptionA 分为 4 个分支（基础多分支模块），用来特征提取，不会缩放图像尺寸。
输入（192通道）
├─ 分支1：1x1卷积 → 64通道（减小通道数，减少算力使用）
├─ 分支2：1x1卷积（48通道）→ 5x5卷积（64通道）（先减小通道数再获取中尺度特征）
├─ 分支3：1x1卷积（64通道）→ 3x3卷积（96通道）→ 3x3卷积（96通道）（先减小通道数再获取大尺度特征）
└─ 分支4：3x3池化 → 1x1卷积（32通道）（池化后再减小通道数，保留全局信息）
→ 4个分支结果拼接（64+64+96+32=256通道）→ 输出
```
##
###### InceptionB:
```
1个 InceptionB 分为 3(2个核心) 个分支（缩图模块）。
输入（288通道）
├─ 分支1：3x3卷积（ stride=2 ）→ 384通道（缩小图像尺寸，获取大尺度特征）
└─ 分支2：1x1卷积（64通道）→ 3x3卷积（96通道）→ 3x3卷积（ stride=2 ）→ 96通道（先提特征再缩小图像尺寸，保细节）
→ 2个分支结果拼接 + 没有列出来的池化层通道数→ 输出（768）
```
##
###### InceptionC:
```
1个 InceptionC 分为 3 个分支（长条特征模块），特征提取细化，不会缩放图像尺寸。
输入（768通道）
├─ 分支1：1x1卷积 → 192通道（减小通道数）
├─ 分支2：1x1卷积（128通道）→ 1x7卷积 → 7x1卷积 → 192通道（把7x7拆成1x7+7x1，获取长条特征）
└─ 分支3：1x1卷积（128通道）→ 7x1卷积 → 1x7卷积 → 7x1卷积 → 1x7卷积 → 192通道（多轮拆卷积，获取更细的长条）
→ 所有分支+池化分支拼接 → 输出（保持768通道）
```
##
###### InceptionD
```
1个 InceptionD 3(2个核心) 个分支（二次缩图模块）。
输入（768通道）
├─ 分支1：1x1卷积（192通道）→ 3x3卷积（ stride=2 ）→ 320通道（先减小通道数再缩小图像尺寸）
└─ 分支2：1x1卷积（192通道）→ 1x7卷积 → 7x1卷积 → 3x3卷积（ stride=2 ）→ 192通道（先获取长条特征再缩图）
→ 2个分支拼接 + 其他池化层 → 输出（1280通道）
```
##
######  InceptionE
```
1个 InceptionE 4(3个核心) 个分支（细化模块），最后特征提取，不会缩放图像尺寸，获取最细粒度特征）
输入（1280→2048通道）
├─ 分支1：1x1卷积 → 320通道（减小通道数）
├─ 分支2：1x1卷积（384通道）→ 拆成两个子分支：1x3卷积 + 3x1卷积 → 拼接（384+384=768通道）
└─ 分支3：1x1卷积（448通道）→ 3x3卷积 → 拆成两个子分支：1x3卷积 + 3x1卷积 → 拼接（384+384=768通道）
→ 所有分支+池化分支拼接 → 输出（2048通道）
```
##
输入model输出实际结构内容
```
model
```
##
```
Inception3(
  (Conv2d_1a_3x3): BasicConv2d(
    (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
  )

  ......
  ......

  (Mixed_5c): InceptionA(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
   ......
   ......
  )
  (Mixed_6a): InceptionB(
    (branch3x3): BasicConv2d(
      (conv): Conv2d(288, 384, kernel_size=(3, 3), stride=(2, 2), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    ......
    ......
  )
  (Mixed_6b): InceptionC(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    ......
    ......
  )
  (AuxLogits): InceptionAux(
    (conv0): BasicConv2d(
      (conv): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
   ......
   ......
  )
  (Mixed_7b): InceptionE(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(1280, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    )
    ......
    ......
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (dropout): Dropout(p=0.5, inplace=False)
  (fc): Linear(in_features=2048, out_features=10, bias=True)
```
##
## 3.Experimental Results and Analysis
#### Training Log(epoch,loss,accuracy)
下列结果是n=100,训练集600张图片的结果
```
epoch0:loss=0.5322,acc=0.7800
epoch1:loss=0.7331,acc=0.8000
epoch2:loss=0.7871,acc=0.7500
epoch3:loss=0.5310,acc=0.8400
epoch4:loss=0.7671,acc=0.7800
epoch5:loss=0.8424,acc=0.7800
epoch6:loss=1.1080,acc=0.7500
epoch7:loss=0.5214,acc=0.8600
epoch8:loss=0.6264,acc=0.8100
epoch9:loss=0.6391,acc=0.7600
```
##
下列结果是n=10,训练集6000张图片的结果
```
epoch0:loss=0.3711,acc=0.8680
epoch1:loss=0.3311,acc=0.8960
epoch2:loss=0.2609,acc=0.9160
epoch3:loss=0.2851,acc=0.9030
epoch4:loss=0.2565,acc=0.9200
epoch5:loss=0.3223,acc=0.9060
epoch6:loss=0.2958,acc=0.9200
epoch7:loss=0.3218,acc=0.9130
epoch8:loss=0.3321,acc=0.9160
epoch9:loss=0.4530,acc=0.8990
```
##
##### 实验结果发现，由于预训练模型很大，所以要数据较多才可以训练效果提升，最后在不同数量的数据集下，损失最小为0.2565，同时准确率在92%。
##
## 4.Experimental Summary
#### 4.1.Overall Reflection on the Experiment
###### 1.)本次实验使用了Fashion-MNIST数据集，与MNIST数据集都是基础分类数据集，输出为10分类任务。
###### 2.)本次实验虽然没有自己搭建InceptionV3架构，但是通过调用预训练模型，知道了网络图像输入尺寸参数，并且又一次复习了全连接分类层的通道数修改和又一次复现了网络训练和测试的流程。
###### 3.)通过上网搜索资料，查阅了InceptionV3的基本结构组成，了解了InceptionA，B，C，D，E不同模块的结构和功能。对于Inception系列网络模块基于感受野大小不变的原则有了更深刻的体会。
##
#### 4.2.Problems in the Experiment
###### 1.)实验中通过对参数的运行，例如model，一开始没有真正理解清楚模块参数变化带来的特征图尺寸或者是通道数的变化情况，通过查阅资料现已经理解清楚。
###### 2.)实验中还遇到训练效果不好的问题，上网查阅资料发现Inception预训练模型网络结构很深，所以少量数据及可能不足以支撑实验效果，通过改进实验数据量大小，改善了损失和准确率的大小。
##
