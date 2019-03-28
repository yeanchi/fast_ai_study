# 迁移学习（Transfer Learning Tutorial）

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
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    # 把数值的大小限制在0和1之间（小于0的等于0，大于1 的等于1）
    
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  
    # 稍作停顿，以便更新绘图

# 得到一个小批量的数据集
inputs, classes = next(iter(dataloaders['train']))

# 从小批量里生成网格
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
```


