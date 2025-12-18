# **Experiment7--ResNet_SVHN** 
(实验7-ResNet实现SVHN分类)
##### This experiment compares the performance, generalization ability and resource consumption of three residual block designs (identity block, projection block, and full-mapping block) on the SVHN dataset.
###### 本实验比较了三种残差块设计（恒等映射块、投影块和全映射块）在 SVHN 数据集上的性能、泛化能力和资源消耗。
##

## 1.Experimental Purpose
##### 1.Master the basic principles of the convolutional neural network based on the InceptionV3 architecture.
##### 2.Conduct classification training and testing on the Fashion-MNIST dataset using the InceptionV3 architecture.
##### 3.Learn to load pre-trained weights and analyze the network architecture.

###### 1.掌握InceptionV3架构卷积神经网络基本原理。
###### 2.利用InceptionV3架构对Fashion-MNIST数据集进行分类训练和测试。
###### 3.学习调用预训练权重，并分析网络架构。

##

## 2.Experimental Content
##### Due to computational resource constraints, the data volume of the Fashion-MNIST dataset is limited, and the image size is converted to adapt to the InceptionV3 network architecture (3×299×299). Load the pre-trained weights of InceptionV3, and adjust the final fully connected layer to perform a classification task with 10 outputs. Finally, conduct a detailed analysis of the InceptionV3 network architecture model.
###### 由于计算资源的限制，对数据集Fashion-MNIST数据量进行限制，对图像的尺寸进行转换为适应InceptionV3网络架构（3x299x299）。调用InceptionV3预训练权重，并调整最后一层全连接层做输出为10的分类任务。最后对InceptionV3网络架构的模型具体分析。
##
#### 2.1.Fashion-MNIST Dataset Loading and Processing
##
##### （1）Import Pytorch and related tool libraries
###### 导入 PyTorch 及相关工具库
```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms,models
from torch.utils.data import Subset
from torchsummary import summary
import numpy as np
```
##
##### （2）Set the training batch size and runtime device, load the dataset, and perform image size conversion and tensor format transformation（3x299x299）on the data images. Set a scaling factor n to control the data volume, and after setting a random seed, randomly select a certain number of datasets to serve as the training set and test set.
###### 设置训练批次大小和运行设备，并加载数据集，对数据图片进行尺寸大小的转换（3x299x299）以及张量格式的变化。设置比例系数n控制数据量，并设置随机种子后，随机抽选一定数量的数据集作为训练集和测试集。
```
batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    # 将图像缩放到299×299尺寸（InceptionV3要求的输入尺寸）
    transforms.Resize(299),
    # 将FashionMNIST的单通道灰度图转换为3通道灰度图（InceptionV3要求3通道输入，三通道值相同）
    transforms.Grayscale(num_output_channels=3),
    # 将PIL图像/NumPy数组转换为PyTorch张量，同时将像素值从[0,255]归一化到[0.0,1.0]
    transforms.ToTensor()
])
# 加载FashionMNIST完整训练集，自动下载至data目录
train_full = datasets.FashionMNIST('data',train=True,download=True,transform=transform)
test_full = datasets.FashionMNIST('data',train=False,download=True,transform=transform)
```
##
###### 选取不同比例数据,创建一个固定随机种子的numpy随机数生成器,随机抽取子集索引
```
n = 10 
rng = np.random.default_rng(42)

# 从训练集全量数据的索引中随机抽取子集索引
# replace=False：不重复抽样（保证每个索引只选一次，避免同一个样本被多次选中）
train_idx = rng.choice(len(train_full), len(train_full)//n, replace=False)    # train_idx是一个一维数组，存放被选中的训练集样本索引
test_idx = rng.choice(len(test_full), len(test_full)//n, replace=False)
```
##
###### PyTorch核心数据加载器：将数据集封装为可迭代的批次迭代器
```
train_loader = torch.utils.data.DataLoader(Subset(train_full,train_idx),# 从完整数据集中截取指定子集
                                           batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(Subset(test_full,test_idx),
                                           batch_size=batch_size,shuffle=True)
```
##
#### 2.2.Construct and adjust the network architecture
##### Load the pre-trained weights of InceptionV3, and adjust the final fully connected layer to perform a classification task with 10 outputs.
###### 调用InceptionV3预训练权重，并调整最后一层全连接层做输出为10的分类任务。
```
model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features,10)
model = model.to(device)
optimizer = optim.Adam(model.parameters(),lr=1e-4)
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
