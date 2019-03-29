# 迁移学习（Transfer Learning Tutorial）

可以在以下链接中初步了解迁移学习：
[CS231n Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/transfer-learning/)

以下是两种主要的迁移学习场景：

- 微调卷积神经网络 我们使用已经训练好的网络参数初始化网络，而不是随机初始化。
- ConvNet作为固定特征提取器 冻结除去最后的完全连接层之外所有网络的权重 最后一个完全连接的层被替换为具有随机权重的新层，并且仅训练该层 。


```
# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # interactive mode
```

## 微调神经网络

### 加载数据

从[此处](https://download.pytorch.org/tutorial/hymenoptera_data.zip)下载数据 并将其解压缩到当前目录 


```
# 对训练集(trian)的扩充和归一化(normalization)
# 只对验证集(valid)归一化

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        #随机裁减图像尺寸为224 
        transforms.RandomHorizontalFlip(),
        #以一定的概率水平翻转（默认值0.5）
        transforms.ToTensor(),
        #转化一个PIL Image 或 numpy.ndarray 成为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #用平均值mean和标准偏差std归一化张量图像 
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        #图片大小调整为给定大小
        transforms.CenterCrop(224),
        #以中心裁减
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
```

```
data_dir = 'data/hymenoptera_data'
```

data_dir : 所有文件的主目录，包含`train`和`vaild`两个子文件夹 

```
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
# 用image_datasets，转换成 torch 能识别的 Dataset 
# 指出了数据从哪个文件夹里取出，以及进行数据增强 

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                            batch_size=4,
                            shuffle=True,
                            num_workers=4)
              for x in ['train', 'val']}
# 拿出转化好的image_datasets里的数据 
# 设置每次加载的数据需要的批次大小、是否洗牌、子进程数量 
              
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# 所有数据的个数

class_names = image_datasets['train'].classes
# 按顺序存储对应标签

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 定义设备
```
代码解析：

> ```
> {x:datasets.ImageFolder(os.path.join(data_dir, x),
> data_transforms[x]) 
> for x in ['train', 'val']} 
> ```
> 如果 x 是 'train'，替换掉：
> 
> ```
> {'train':datasets.ImageFolder(os.path>.join(data_dir, train),
> data_transforms['train'])}
> ```
>
> `os.path.join` 是合并两者路径,以下为例 
>
> ```
> a = os.path.join(data_dir,'train')
> print(a)
> ```
> 
> 输出为
> 
> ```
> data/hymenoptera_data/train
> ```

>代码简化为
>
> ```
{'train':datasets.ImageFolder("data/hymenoptera_data/train"
,data_transforms['train'])}
```
>指出了数据从哪个文件夹里取出，以及进行数据增强 


### 可视化一些图像

可视化一些训练图像，以便了解数据增强 

```
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # 把张量转化为numpy,然后进行转置，原本数组顺序是（0，1，2）现在变为（1，2，0）
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    # 把数值的大小限制在0和1之间（小于0的等于0，大于1 的等于1）,以上应该是归一化 
    
    plt.imshow(inp)
    # 绘图
    
    if title is not None:
        plt.title(title)
    # 输出图片名
    
    plt.pause(0.001)  
    # 稍作停顿，以便更新绘图

inputs, classes = next(iter(dataloaders['train']))
# 从dataloaders文件夹中取出训练数据，和标签（文件名） 
# 记住，dataloaders是小批次4张图，batch_size=4

out = torchvision.utils.make_grid(inputs)
# 传入inputs，make_grid的作用是将若干幅图像拼成一幅图像 四张图像并排 

imshow(out, title=[class_names[x] for x in classes])
# 将数据传入函数
```

## 设置如何训练模型

`torch.optim.lr_scheduler`提供了几种基于迭代数调整学习率的方法  



```
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    # 此函数需要传入模型、损失度量、优化器、动态优化器学习率方法，迭代次数 
    
    since = time.time()
    # 记录开始的时间
    
    best_model_wts = copy.deepcopy(model.state_dict())
    # state_dict()返回包含模块整个状态的字典（参数和缓存） 
    # deepcopy 拷贝对象，深拷贝（拷贝对象及其子对象）

    best_acc = 0.0
    # 定义一个准确率 

    for epoch in range(num_epochs):
    # 根据num_epochs迭代次数开始迭代：
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # 格式化函数，丰富字串符格式化功能 
        # 打印出来第一个应该是这样 epoch 0/24
        print('-' * 10)
        # ----------
        
        for phase in ['train', 'val']:
        # 两个阶段‘train’和‘val’
            if phase == 'train':
                scheduler.step()
                # 打开动态优化器学习率
                model.train()  
                # 将模型设置为训练模式 
            else:
                model.eval()   
                # 设置模型为评估模式 
            
            running_loss = 0.0 
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
             # 遍历小批次数据
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 把数据和标签放到设备上

                optimizer.zero_grad()
                # 梯度清零

                with torch.set_grad_enabled(phase == 'train'):
                # 只在训练集上跟踪操作历史
                    outputs = model(inputs)
                    # 向前传播
                    
                    _, preds = torch.max(outputs, 1)
                    # 预测结果
                    
                    loss = criterion(outputs, labels)
                    # 损失函数

                    if phase == 'train':
                        loss.backward()
                        # 只对训练集计算梯度 
                        
                        optimizer.step()
                        # 只对训练集使用优化器反向传播 

                running_loss += loss.item() * inputs.size(0)
                # 返回的损失是批次中所有示例的平均值，所以乘以批次尺寸，计算小批次所有损失
                
                running_corrects += torch.sum(preds == labels.data)
                # 预测正确的个数

            epoch_loss = running_loss / dataset_sizes[phase]
            # 计算出所有loss，除以数量 
            # 考虑到最后一个批次数量可能不是5，才这样计算 
            
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            # 计算正确率
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # 输出此时的阶段、一次迭代损失和准确率
            
             
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # 当前的准确率替换 best_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # 保存准确率最高的模型的参数（深度复制）
        print()

    time_elapsed = time.time() - since
    # 现在时间减去开始时间 
    
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))    
    # 打印出时间和准确率

    model.load_state_dict(best_model_wts)
    return model
    # 加载刚刚保存的模型返回模型
```

>参考资料：
>[state_dict()](https://pytorch.org/docs/stable/nn.html#torch.nn.Module.state_dict)
>[deepcopy](https://docs.python.org/3/library/copy.html)
>[Python format 格式化函数](http://www.runoob.com/python/att-string-format.html)
>[torch.optim - PyTorch主文档](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau)


### 微调卷积神经网络

```
model_ft = models.resnet18(pretrained=True)
# 加载模型，打开预训练

num_ftrs = model_ft.fc.in_features
# 线性图层的输入数量，也就是最后一层的输入数量，就是fc层上一层的输出数量

model_ft.fc = nn.Linear(num_ftrs, 2)
# 将最后一层线性层替换掉

model_ft = model_ft.to(device)
# 模型放入设备

criterion = nn.CrossEntropyLoss()
# 损失评价标准，损失函数

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# 观察所有参数是否存在，用SGD优化模型所有参数 

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
# 根据迭代数调整学习率的方法，每7个周期，LR衰减0.1倍

```
> 参考资料
> 
> [torch.optim.lr_scheduler.MultiStepL](https://pytorch.org/docs/stable/optim.html?highlight=lr_scheduler%20steplr#torch.optim.lr_scheduler.MultiStepLR)
> [pytorch/linear.py at master · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py)

### 训练和评估
```
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)
```

### 可视化模型预测

用于显示一些图像预测的泛型函数（generic function） 

```
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    # 一些模型使用具有不同训练和评估行为的模块，如批处理规范化 
    # model.train(mode=was_training)将要传入的参数 
    # model.eval() 将模块设置为评估模式 就不会使用类似Batch Normalization  和  Dropout 方法模式
    
    images_so_far = 0
    fig = plt.figure()
    # 如果未提供，将创建新图形，图形编号将递增 

    with torch.no_grad():
    # 不计算梯度
        for i, (inputs, labels) in enumerate(dataloaders['val']):
        # 每一次迭代验证集释放一小批数据
        # 列出数据下标和数据 
 
        
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 放入设备

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            # 预测结果
            
            for j in range(inputs.size()[0]):
            # 从小批量个数开始遍历 1-5
                
                images_so_far += 1
                
                ax = plt.subplot(num_images//2, 2, images_so_far)
                # 在当前图中添加子图 默认传入（3，2，？）递增，第一个是行数，第二个列数，第三个索引
                
                ax.axis('off')
                # 关闭轴线和标签 
                
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                # title标题等于预测的标签
                
                imshow(inputs.cpu().data[j])
                # 显示出inputs的图片
                
                if images_so_far == num_images:
                # 如果到达第六次之后 
                    model.train(mode=was_training)
                # 模型改为训练模式 
                    return
                    
        model.train(mode=was_training)
        # 模型改为训练模式 
```

> 参考资料
> [ def train](https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.train) 
> [matplotlib.pyplot.subplot - Matplotlib 3.0.3文档](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html#matplotlib.pyplot.subplot)

可视化：

```
visualize_model(model_ft)
```


## 将ConvNet作为固定特征提取器

在这里，我们需要冻结除最后一层之外的所有网络。我们需要设置冻结参数，以便不计算渐变。` requires_grad == Falsebackward()`

您可以在[此处](https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward)的文档中阅读更多相关信息 。

```
model_conv = torchvision.models.resnet18(pretrained=True)
# 设置模型为resnet18，打开预训练

for param in model_conv.parameters():
    param.requires_grad = False
# 迭代出模型的参数，并且关闭梯度

num_ftrs = model_conv.fc.in_features
# fc层的input特征数量

model_conv.fc = nn.Linear(num_ftrs, 2)
# 把fc层换个线性函数

model_conv = model_conv.to(device)
# 模型放入设备

criterion = nn.CrossEntropyLoss()
# 损失度量，损失函数

optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
# 对比之前，只针对最后的fc层进行了优化。

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
# 每7个周期，LR衰减0.1倍
```

### 训练和评估

```
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)
```

### 可视化

```
visualize_model(model_conv)

plt.ioff()
# 关闭图片

plt.show() 
# 再执行
# 之前已经执行plt.show()之后，程序会暂停到那儿，并不会继续执行下去。如果需要继续执行程序，就要关闭图片。


```


本文来源：[Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#finetuning-the-convnet)

原文作者：[Sasank Chilamkurthy](https://chsasank.github.io/)


