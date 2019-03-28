# 迁移学习（Transfer Learning Tutorial）

可以在以下链接中阅读更多关于迁移学习：
[CS231n Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/transfer-learning/)

以下是两种主要的迁移学习场景：

- 微调卷积神经网络。我们使用已经训练好的网络参数初始化网络，而不是随机初始化。
- ConvNet作为固定特征提取器。冻结除去最后的完全连接层之外所有网络的权重。最后一个完全连接的层被替换为具有随机权重的新层，并且仅训练该层。


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

## 加载数据

从[此处](https://download.pytorch.org/tutorial/hymenoptera_data.zip)下载数据 并将其解压缩到当前目录。


```
# 对训练集(trian)的扩充和归一化(normalization)
# 只对验证集(valid)归一化

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        #随机裁减图像尺寸为224。
        transforms.RandomHorizontalFlip(),
        #以一定的概率水平翻转（默认值0.5）
        transforms.ToTensor(),
        #转化一个PIL Image 或 numpy.ndarray 成为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #用平均值mean和标准偏差std归一化张量图像。
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

data_dir : 所有文件的主目录，包含`train`和`vaild`两个子文件夹。

```
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
# 使用数据增强

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                            batch_size=4,
                            shuffle=True,
                            num_workers=4)
              for x in ['train', 'val']}
#设置每次加载的数据需要的批次、是否洗牌、子进程数量
              
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
#所有数据的个数

class_names = image_datasets['train'].classes
#定义标签名字

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#定义设备
```
代码解析：

> ```
>{x:datasets.ImageFolder(os.path.join(data_dir, x),
>data_transforms[x]) 
>for x in ['train', 'val']} 
>```
> 如果 x 是 'train'，替换掉：
> 
> ```
{'train':datasets.ImageFolder(os.path.join(data_dir, train),
data_transforms['train'])}
```
>
> `os.path.join` 是合并两者路径,以下为例。

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
>指出了数据从哪个文件夹里取出，以及进行数据增强。


## 可视化一些图像

可视化一些训练图像，以便了解数据增强。

```
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # 把张量转化为numpy,然后进行转置，原本数组顺序是（0，1，2）现在变为（1，2，0）
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    # 把数值的大小限制在0和1之间（小于0的等于0，大于1 的等于1）
    
    plt.imshow(inp)
    # 绘图
    if title is not None:
        plt.title(title)
    # 输出图片名
    plt.pause(0.001)  
    # 稍作停顿，以便更新绘图

inputs, classes = next(iter(dataloaders['train']))
# 从dataloaders文件夹中取出训练数据，和标签（文件名）。
# 记住，dataloaders是小批次4张图，batch_size=4

out = torchvision.utils.make_grid(inputs)
# 传入inputs，make_grid的作用是将若干幅图像拼成一幅图像。四张图像并排。

imshow(out, title=[class_names[x] for x in classes])
# 将数据传入函数
```

## 训练模型

`torch.optim.lr_scheduler`提供了几种基于迭代数调整学习率的方法。

在下面，参数scheduler是来自`torch.optim.lr_scheduler`的LR调度器对象。 



```
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    # 记录开始的时间
    
    best_model_wts = copy.deepcopy(model.state_dict())
    #返回包含模块整个状态的字典（参数和缓存）。
    #拷贝对象，深拷贝（拷贝对象及其子对象）

    best_acc = 0.0
    #定义个比较好的准确率

    for epoch in range(num_epochs):
    # 根据num_epochs迭代次数开始迭代：
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # 格式化函数，丰富字串符格式化功能。
        # 打印出来第一个应该是这样 epoch 0/24
        print('-' * 10)
        # ----------
        
        for phase in ['train', 'val']:
        # 两个阶段‘train’和‘val’
            if phase == 'train':
                scheduler.step()
                # 打开优化器
                model.train()  
                # 将模型设置为训练模式 
            else:
                model.eval()   
                # 设置模型为评估模式 
            
            running_loss = 0.0 
            running_corrects = 0

           
            for inputs, labels in dataloaders[phase]:
             # 遍历所有数据
                inputs = inputs.to(device)
                labels = labels.to(device)
                #把数据和标签放到设备上

                optimizer.zero_grad()
                #梯度设置为0.

                # 向前传播
                with torch.set_grad_enabled(phase == 'train'):
                # 只在训练集上跟踪操作历史
                    outputs = model(inputs)
                    
                    _, preds = torch.max(outputs, 1)
                    # 预测结果
                    
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    # 只对训练集反向传播 + 优化器  

                # 统计数据
                running_loss += loss.item() * inputs.size(0)
                #返回的损失是批次中所有示例的平均值，所以乘以批次尺寸
                running_corrects += torch.sum(preds == labels.data)
                # 预测的正确个数

            epoch_loss = running_loss / dataset_sizes[phase]
            #计算出所有loss，除以数量，主要最后一个批次数量可能不是5，才这样计算
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            #算出正确率
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            #输出阶段，一次迭代，损失和准确率
            
             
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # 当前的准确率替换 best_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # 保存准确率最高的模型的参数（深度复制）
        print()

    time_elapsed = time.time() - since
    # 现在时间减去开始时间。
    
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


