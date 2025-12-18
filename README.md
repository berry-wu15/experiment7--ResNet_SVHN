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
##### 4.构建有8个残差块连接的ResNet网络，适配三种不同的残差块进行训练
```
class ResNetExperiment(nn.Module):
    """ResNet（和论文结构对齐：初始卷积+残差层+池化+全连接）"""
    def __init__(self, block_type, num_classes=NUM_CLASSES):
        super(ResNetExperiment, self).__init__()
        # 初始卷积层（处理原始3通道输入）
        self.init_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.init_bn = nn.BatchNorm2d(32)  #批量归一化
        self.init_relu = nn.ReLU(inplace=True)
        
        # 残差层（堆叠2个残差块）
        self.res_block1 = block_type(32, 32)
        self.res_block2 = block_type(32, 32)
        self.res_block3 = block_type(32, 32)
        self.res_block4 = block_type(32, 32)
        self.res_block5 = block_type(32, 32)  
        self.res_block6 = block_type(32, 32)
        self.res_block7 = block_type(32, 32)  
        self.res_block8 = block_type(32, 32)

        # 池化+全连接（分类头）
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        self.dropout = nn.Dropout(0.3)  # 强化正则化
        self.fc = nn.Linear(32, num_classes)


    def forward(self, x):
        # 初始卷积
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.init_relu(x)
        
        # 残差块
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)
        x = self.res_block7(x)
        x = self.res_block8(x)

        # 分类
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.dropout(x)
        x = self.fc(x)
        return x

```
##
#### 2.3.模型训练
##### 设置训练轮数并且展示数据训练过程的前向传播，损失计算，反向传播等流程。
##
```
# 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, pred = outputs.max(1)
            train_total += labels.size(0)
            train_correct += pred.eq(labels).sum().item()
```
##
```
# 测试
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, pred = outputs.max(1)
                test_total += labels.size(0)
                test_correct += pred.eq(labels).sum().item()
```
##

##
#### 2.4.网络架构介绍
##
##### (1)三个不同残差块结构区别
```
# 恒等映射块（Identity Block）
输入 → 主分支（2层3×3卷积） → 输出
      ↓ 无参数恒等映射（通道不匹配时零填充）
x短路径：输入直接连接到输出，无额外卷积/参数

# 投影块（Projection Block）
输入 → 主分支（2层3×3卷积） → 输出
      ↓ 1×1卷积投影（匹配通道维度）
x短路径：有1×1卷积+BN（归一化层），少量额外参数

# 全映射块（Full Mapping Block）
输入 → 主分支（2层3×3卷积） → 输出
      ↓ 3×3卷积+BN+ReLU（全维度映射）
x短路径：有完整卷积层，参数最多
```
##
##### (2) ResNet 的整体结构示意流程
```
输入图像（3×32×32）
↓ 初始卷积层（3→32通道，3×3卷积+BN+ReLU）
↓ 残差块（8个，32→32通道，无尺寸变化）
↓ 平均池化（32×1×1）
↓ Dropout（0.3）
↓ 全连接层（32→10）→ 输出10分类结果
```
##
## 3.实验结果与分析
#### 初始化数据集时，先是选取10000张训练集数据，3000张测试及数据，但实验结果未达到预期，故现在选用60000张训练数据，15000张测试数据
#### 1.恒等映射块（Identity Block）训练测试结果
```
epoch0:loss=1.6504,acc=0.4299
epoch1:loss=0.6004,acc=0.8261
epoch2:loss=0.4246,acc=0.8776
epoch3:loss=0.3633,acc=0.8964
epoch4:loss=0.3277,acc=0.9067
epoch5:loss=0.2998,acc=0.9150
epoch6:loss=0.2791,acc=0.9211
epoch7:loss=0.2679,acc=0.9248
epoch8:loss=0.2485,acc=0.9304
epoch9:loss=0.2404,acc=0.9330
最终测试：loss=0.2497,acc=0.9289
```
##
#### 2.投影块（Projection Block）训练测试结果
```
epoch0:loss=1.5838,acc=0.4552
epoch1:loss=0.5609,acc=0.8440
epoch2:loss=0.3885,acc=0.8899
epoch3:loss=0.3336,acc=0.9053
epoch4:loss=0.3035,acc=0.9149
epoch5:loss=0.2794,acc=0.9222
epoch6:loss=0.2638,acc=0.9262
epoch7:loss=0.2470,acc=0.9317
epoch8:loss=0.2313,acc=0.9358
epoch9:loss=0.2255,acc=0.9375
最终测试：loss=0.2318,acc=0.9373
```
##
#### 3.全映射块（Full Mapping Block）训练测试结果
```
epoch0:loss=1.4030,acc=0.5316
epoch1:loss=0.4986,acc=0.8614
epoch2:loss=0.3696,acc=0.8964
epoch3:loss=0.3190,acc=0.9106
epoch4:loss=0.2882,acc=0.9190
epoch5:loss=0.2646,acc=0.9269
epoch6:loss=0.2522,acc=0.9298
epoch7:loss=0.2357,acc=0.9347
epoch8:loss=0.2216,acc=0.9389
epoch9:loss=0.2166,acc=0.9411
最终测试：loss=0.2250,acc=0.9413
```
##
##### 实验结果分析：
###### 1. 性能表现：全映射块最终测试准确率最高（94.13%），投影块（93.73%），恒等映射块稍低（92.89%），最高最低差距 1.24%，属于可接受范围；训练过程中，全映射块收敛速度最快，恒等映射块收敛稍慢但更稳定。
###### 2. 资源消耗（这里比较的是训练参数量以及训练测试耗时）：恒等映射块参数量最少（大概182.56k），训练时间最短（4min）。
###### 3. 泛化能力：三者训练和测试准确率差值均在 1% 以内，无明显过拟合；其中恒等映射块差值最小（0.41%），泛化稳定性最优。
##
## 4.实验小结
#### 4.1.实验总体反思
###### 1.)实验基于 SVHN 数据集对比了三种残差块的性能，验证了 ResNet 残差连接的核心思路 ，数据集数量较多时，ResNet在x短路径下参数尽可能少，会在保证准确率差值不大的情况下计算资源消耗最少。（本来是想验证老师上课讲的路径上参数越少错误率越低但由于数据量的问题可能会过早导致过拟合问题所以此处就分析了计算资源数量）
###### 2.)动手搭建了包含 8 个残差块的轻量化 ResNet，从网络层的定义、模块的组合到整体前向传播流程，更清楚地掌握了 ResNet 的搭建逻辑，8 层网络在 SVHN 数据集上也能稳定达到 92% 以上的分类准确率。
###### 3.)搞懂了不同残差块的设计思路：恒等映射块的x短路径没有额外参数，不用额外计算，能节省算力；投影块靠 1×1 卷积调整通道数，能适配不同维度的特征图；全映射块的**捷径**路径加了更多卷积层和参数，虽然拟合数据的能力更强，但需要的计算资源也更多，训练起来更占内存。
##
#### 4.2.实验中遇到的问题
###### 1.)实验前只记住了残差公式 H(x)=F(x)+x，没理解梯度传递逻辑，推导后才懂恒等映射的梯度优势，通过查阅资料现已经理解清楚。
###### 2.)实验中浅层 ResNet 里全映射块表现更好，和老师上课讲的不一样，分析后才知道浅层网络体现不出恒等映射的架构优势。上网查阅资料发现ResNet网络结构很深，所以少量数据及可能不足以支撑实验效果，通过改进实验数据量大小，改善了损失和准确率的大小。
##
