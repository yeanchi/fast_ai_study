[toc]

# 微调 torchvision 模型

在本教程中，我们将深入研究如何对 [torchvision](https://pytorch.org/docs/stable/torchvision/models.html) 模型进行[微调](https://pytorch.org/docs/stable/torchvision/models.html)和特征提取，所有这些模型都已在1000级Imagenet数据集上进行了预训练。本教程将深入了解如何使用几个现代CNN架构，并将建立一个对任何 PyTorch 模型进行微调的直觉。由于每个模型架构都是不同的，所以不存在适用于所有场景的样板微调代码。相反，研究人员必须查看现有的体系结构，并为每个模型进行自定义调整。

在本文中，我们将执行两种类型的迁移学习: finetuning（微调） 和feature extraction（特征提取） 。在finetuning中，我们从一个预训练的模型开始，为我们的新任务更新模型的**所有参数**，本质上是对整个模型进行再训练。在特征提取中，我们从一个预先训练的模型开始，仅更新从中导出预测的最终图层权重。之所以称之为特征提取，是因为我们使用预先训练好的CNN作为固定的特征提取器，只改变输出层。

一般来说，这两种迁移学习方法都遵循相同的几个步骤:

- 初始化预训练模型
- 重新构造最后一层，使其输出的数量与新数据集中类的数量相同
- 为优化算法定义要在训练期间更新的参数
- 运行训练步骤


```
from __future__ import print_function
# 就算在python2中print也要括号
from __future__ import division
# 导入精准除法
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
```

## 输入

以下是为运行更改的所有参数。我们将使用 hymenoptera_data 数据集，可在此处[下载](https://download.pytorch.org/tutorial/hymenoptera_data.zip) 。该数据集包含两个类，蜜蜂和蚂蚁，其结构使得我们可以使用 [ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.ImageFolder) 数据集，而不是编写我们自己的自定义数据集。下载数据并将 `data_dir` 输入设置为数据集的根目录。该 `model_name` 输入你想使用，必须从这个列表中选择模型的名称：

```
[resnet, alexnet, vgg, squeezenet, densenet, inception]
```

其他输入如下：`num_classes` 是数据集中的类数，`batch_size` 是用于训练的批量大小，可以根据您的机器的能力进行调整，`num_epochs` 是我们想要运行的训练周期的数量，并且 `feature_extract` 是布尔值这定义了我们是微调还是特征提取。

- 如果`feature_extract = False`，模型被微调并且所有模型参数都被更新。
- 如果`feature_extract = True`，仅更新最后一层参数，则其他参数保持不变。

```
data_dir = "./data/hymenoptera_data"
# 数据目录。这里，我们假设目录的格式符合ImageFolder结构


model_name = "squeezenet"
# 从 [resnet, alexnet, vgg, squeezenet, densenet, inception] 中选择模型


num_classes = 2
# 数据集的分类数量

batch_size = 8
# 选择训练的批次大小（根据你的内存大小而变化）

num_epochs = 15
# 训练迭代多少次

feature_extract = True
# 用于特征提取。为假，微调整个模型参数。为真，只更新被重塑的图形参数。

```

## 辅助函数

在编写调整模型的代码之前，我们先定义几个辅助函数。

### 处理给定模型训练和验证

`train_model` 函数处理给定模型的训练和验证。作为输入，它接受一个PyTorch模型、一个dataloader字典、一个loss函数、一个优化器、要训练和验证的特定数量的epoch，以及一个布尔标志(用于何时模型是inception模型)。

is_inception 标志用于容纳 Inception v3 模型，因为该体系结构使用了一个辅助输出，而整个模型的损失同时考虑了辅助输出和最终输出，[如这所述](https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958)。

函数为指定数量的epoch进行训练，每个epoch之后运行完整的验证步骤。它还跟踪最佳执行模型(就验证精度而言)，并在培训结束时返回最佳执行模型。在每个迭代之后，输出训练和验证的准确性。


```
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
# 需要传入的参数有：模型、处理好的数据、损失度量、优化器、迭代数、是微调还是只调整最后一层。
    
    since = time.time()
    # 记录时间

    val_acc_history = []
    # 验证集准确率

    best_model_wts = copy.deepcopy(model.state_dict())
    # 深度复制模型的整个状态（参数和缓存）
    
    best_acc = 0.0
    # 最佳准确率
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # 打印迭代周期

        for phase in ['train', 'val']:
        # 分为训练集合验证集两个阶段
        
            if phase == 'train':
                model.train()  
                # 如果为训练集，打开训练模式
            
            else:
                model.eval()  
                # 设置为评估模式
                
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
            # 取出本次迭代里的小批量数据
            
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 分别将数据和标签放到设备。

                optimizer.zero_grad()
                # 梯度设置为0

                
                with torch.set_grad_enabled(phase == 'train'):
                # 如果为训练集，则打开梯度
                
                    if is_inception and phase == 'train':
                # 如果为训练集且is_inception = True（微调）
                    
                        outputs, aux_outputs = model(inputs)
                        # 模型的特殊情况，多出一个辅助输出。
                        
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                        # 验证集中我们通过最终输出和辅助输出之和来计算损失,再配上一定的比例。
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        # 但在验证集中，我们只考虑最终输出。
                    _, preds = torch.max(outputs, 1)
                    # 预测的结果

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    # 只在训练阶段，计算反向传播，并用优化器更新梯度。

                running_loss += loss.item() * inputs.size(0)
                # 统计小批量损失
                
                running_corrects += torch.sum(preds == labels.data)
                # 统计小批量正确率

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # 统计每次迭代的损失
            
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            # 统计每次迭代的准确率

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

           
            if phase == 'val' and epoch_acc > best_acc:
            # 如果本次阶段是验证集，且本次准确率高于最佳准确率
                
                best_acc = epoch_acc
                # 最佳准确率被替换
                
                best_model_wts = copy.deepcopy(model.state_dict())
                # 本次模型的参数和缓存被保存
                
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                # 如果阶段是验证集那么把准确率的历史添加到val_acc_history
        print()

    time_elapsed = time.time() - since
    # 每次迭代花费时间
    
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    # 加载效果最好的模型
    
    return model, val_acc_history
```

### 设置模型参数的 .requires_grad 属性

当我们提取特征时，这个辅助函数将模型中参数的`.requires_grad` 属性设置为 `False`。

```
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
```

默认情况下，当我们加载一个预训练的模型时，所有的参数都有`.requires_grad=True`，如果我们从零开始训练或者使用**微调**，这是可以的。但是，如果我们正在提取特性，并且只想为新初始化的层计算梯度，那么我们希望所有其他参数都不需要梯度。这在后面会更有意义。

## 初始化化和重塑网络

现在到了最有趣的部分。这里是我们处理每个网络的重塑的地方。注意，这不是一个自动过程，并且对每个模型都是惟一的。回想一下，CNN模型的最后一层(通常是FC层)的节点数与数据集中输出类的数量相同。由于所有的模型都是在Imagenet上预先训练的，所以它们都有大小为1000的输出层，每个类有一个节点。

这里的目标是重新构造最后一层，使其具有与以前相同的输入数量，并具有与数据集中的类数量相同的输出数量。在下面的部分中，我们将讨论如何分别更改每个模型的体系结构。但是首先，有一个关于微调和特征提取之间区别的重要细节。

在进行特征提取时，我们只需要更新最后一层的参数，或者换句话说，我们只需要更新正在重构的层的参数。因此，我们不需要计算没有更改的参数的梯度，因此为了提高效率，我们将.requires_grad属性设置为False。这一点很重要，因为默认情况下，该属性被设置为True。然后，当我们初始化新层时，默认情况下新参数为.requires_grad=True，因此只更新新层的参数。当我们设置finetuning时，我们可以将.required_grad的所有值设置为True。

**最后，请注意，inception_v3要求输入大小为(299,299)，而所有其他模型都要求(224,224)。**

### Resnet

在论文 [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)中介绍了。
有几种不同大小的变体，包括Resnet18、Resnet34、Resnet50、Resnet101和Resnet152，所有这些都可以从 torchvision 模型中获得。这里我们使用 Resnet18 ，因为我们的数据集很小，只有两个类。打印模型时，最后一层为全连通层，如下图所示:

```
(fc): Linear(in_features=512, out_features=1000, bias=True)

```
因此，我们必须重新初始化 model.fc 为一个线性层，具有512个输入特性和2个输出特性，其中:

```
model.fc = nn.Linear(512, num_classes)
```

### Alexnet

Alexnet是在论文 [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) 中引入的，是第一个非常成功的CNN在ImageNet数据集上。当我们打印模型架构时，我们看到模型输出分类器来自的第6层.

```
(classifier): Sequential(
    ...
    (6): Linear(in_features=4096, out_features=1000, bias=True)
 )
```

要使用数据集的模型，我们将该层重新初始化为

```
model.classifier[6] = nn.Linear(4096,num_classes)
```

### VGG

VGG 引用来自论文 [Very Deep Convolutional Networks for Large-Scale Image Recognition.Torchvision](https://arxiv.org/pdf/1409.1556.pdf)，Torchvision 提供了8个不同长度的VGG版本，其中一些具有批处理正常化层。这里我们使用带批处理规范化的VGG-11。输出层类似于Alexnet，即


```
(classifier): Sequential(
    ...
    (6): Linear(in_features=4096, out_features=1000, bias=True)
 )
```

因此，我们使用相同的技术来修改输出层

```
model.classifier[6] = nn.Linear(4096,num_classes)
```

### Squeezenet

[SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360) 描述了Squeeznet的体系结构并使用不同于这里显示的任何其他模型的输出结构。Torchvision有两个版本的Squeezenet，我们使用的是1.0版。输出来自分类器的第1层1x1卷积层:


```(classifier): Sequential(
    (0): Dropout(p=0.5)
    (1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
    (2): ReLU(inplace)
    (3): AvgPool2d(kernel_size=13, stride=1, padding=0)
 )
```
为了修改网络，我们重新初始化 Conv2d 层，使其具有深度（通道）2的输出特征图为:

```
model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
```

### Densenet

Densenet是在论文 [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) 中介绍的. Torchvision 有四个变种的 Densenet，但在这里我们只使用Densenet-121。输出层为线性层，具有1024个输入特征:

```
(classifier): Linear(in_features=1024, out_features=1000, bias=True)
```

为了重新构造网络，我们将分类器的线性层初始化为：


```
model.classifier = nn.Linear(1024, num_classes)
```

### Inception v3
最后，Inception v3是在 [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567v1.pdf).这个网络是独特的，因为它在训练时有两个输出层。第二个输出称为辅助输出，包含在网络的 AuxLogits 部分中。主输出是网络末端的线性层。注意，在测试时，我们只考虑主要输出。加载模型的辅助输出和主输出打印如下:


```
(AuxLogits): InceptionAux(
    ...
    (fc): Linear(in_features=768, out_features=1000, bias=True)
 )
 ...
(fc): Linear(in_features=2048, out_features=1000, bias=True)
```

为了完善这个模型，我们必须重塑这两个层次。这是通过以下步骤完成的

```
model.AuxLogits.fc = nn.Linear(768, num_classes)
model.fc = nn.Linear(2048, num_classes)

```

注意，许多模型都具有类似的输出结构，但是每个模型的处理方式必须略有不同。此外，检查重新构造的网络的打印模型体系结构，并确保输出特性的数量与数据集中类的数量相同。

```
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # 初始化这些变量，这些变量将在这个if语句中设置。每个变量都是特定于模型的。
    
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        # 打开预训练
        set_parameter_requires_grad(model_ft, feature_extract)
        # 是否关闭梯度
        num_ftrs = model_ft.fc.in_features
        # 获取这个层输入特征
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        # 替换该层
        input_size = 224
        # 设置图片输入大小为224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # 处理辅助网络
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        # 处理主要网络
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
# 运行初始化模型

print(model_ft)
# 打印刚刚实例化的模型
```

输出：

```
SqueezeNet(
  (features): Sequential(
    (0): Conv2d(3, 96, kernel_size=(7, 7), stride=(2, 2))
    (1): ReLU(inplace)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    (3): Fire(
      (squeeze): Conv2d(96, 16, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace)
      (expand1x1): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace)
      (expand3x3): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace)
    )
    (4): Fire(
      (squeeze): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace)
      (expand1x1): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace)
      (expand3x3): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace)
    )
    (5): Fire(
      (squeeze): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace)
      (expand1x1): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace)
      (expand3x3): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace)
    )
    (6): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    (7): Fire(
      (squeeze): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace)
      (expand1x1): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace)
      (expand3x3): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace)
    )
    (8): Fire(
      (squeeze): Conv2d(256, 48, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace)
      (expand1x1): Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace)
      (expand3x3): Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace)
    )
    (9): Fire(
      (squeeze): Conv2d(384, 48, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace)
      (expand1x1): Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace)
      (expand3x3): Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace)
    )
    (10): Fire(
      (squeeze): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace)
      (expand1x1): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace)
      (expand3x3): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace)
    )
    (11): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    (12): Fire(
      (squeeze): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace)
      (expand1x1): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace)
      (expand3x3): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace)
    )
  )
  (classifier): Sequential(
    (0): Dropout(p=0.5)
    (1): Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
    (2): ReLU(inplace)
    (3): AdaptiveAvgPool2d(output_size=(1, 1))
  )
)
```

## 加载数据

现在我们知道了输入大小必须是多少，我们可以初始化数据转换、图像数据集和 dataloader。注意，模型使用 hard-coded normalization 进行了预训练，[如下所述](https://pytorch.org/docs/master/torchvision/models.html)。


```
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
# 训练数据的扩充和归一化
# 验证数据只归一化

print("Initializing Datasets and Dataloaders...")


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# 创建训练和验证集，把数据和标签放在一起，转换成 torch 能识别的 Dataset,以及数据增强

dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
# 创建训练和验证集，随机的批次大小，子进程

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 检测是否有GPU可用
```

## 创建优化器

既然模型结构是正确的，那么微调和特性提取的最后一步就是创建一个优化器，它只更新所需的参数。回想一下，在加载预训练模型之后，但在重新构造之前，如果feature_extract=True，我们手动将所有参数的.requires_grad属性设置为False。然后，重新初始化的层的参数默认为.requires_grad=True。现在我们知道，所有具有.requires_grad=True的参数都应该优化。接下来，我们将这些参数列表并将该列表输入到SGD算法构造函数中。

要验证这一点，请查看打印参数以了解。当微调时，这个列表应该很长，并且包含所有的模型参数。但是，当特征提取时，这个列表应该很短，并且只包含被重构层的权重和偏差。

```
model_ft = model_ft.to(device)
# 将模型发到cpu

params_to_update = model_ft.parameters()
# 模型参数

print("Params to learn:")
if feature_extract:
# 如果梯度是关闭的

    params_to_update = []
    # 参数为空
    for name,param in model_ft.named_parameters():
    # 取出模型中的名字和参数
        if param.requires_grad == True:
        # 我们只会更新刚刚初始化的参数里面带有requires_grad=True的参数
            params_to_update.append(param)
            # 将更新的参数放入params_to_update
            
            print("\t",name)
else:
# 如果梯度没有关闭，直接继承之前的参数
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
# 观察所有参数都在优化
```

输出：


```
Params to learn:
         classifier.1.weight
         classifier.1.bias
```


## 运行培训和验证步骤

最后，最后一步是为模型设置损失，然后针对设定的迭代数运行训练和验证功能。请注意，根据迭代的数量，此步骤可能需要一段时间才能在CPU上执行。此外，默认学习速率对于所有模型都不是最佳的，因此为了获得最大准确率，有必要分别调整每个模型。


```
criterion = nn.CrossEntropyLoss()
# 设置损失函数

model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
# 训练和评估

```

## 与从零开始训练的模型比较

只是为了好玩，让我们看看如果我们不使用转移学习，模型将如何学习。微调与特征提取的性能在很大程度上取决于数据集，但一般而言，两种转移学习方法相对于从头开始训练的模型，在训练时间和总体准确性方面产生有利结果。


```
# 初始化用于此运行的模型的非预训练版本
scratch_model,_ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)

scratch_model = scratch_model.to(device)

scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)

scratch_criterion = nn.CrossEntropyLoss()

_,scratch_hist = train_model(scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer, num_epochs=num_epochs, is_inception=(model_name=="inception"))

#绘制验证准确度的训练曲线和训练周期的数量#(转移学习方法vs从头开始训练的模型)

ohist = []
shist = []

ohist = [h.cpu().numpy() for h in hist]
shist = [h.cpu().numpy() for h in scratch_hist]
# 数据numpy()化

plt.title("Validation Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1,num_epochs+1),ohist,label="Pretrained")
plt.plot(range(1,num_epochs+1),shist,label="Scratch")
# 画出转移学习方法vs从头开始训练的模型的准确率

plt.ylim((0,1.))

plt.xticks(np.arange(1, num_epochs+1, 1.0)

plt.legend()
plt.show()
```

## 最后的想法和下一步的去处

尝试运行其他一些模型，看看准确度有多好。另外，请注意特征提取花费的时间较少，因为在向后传递中我们不必计算大部分梯度。这里有很多地方可以去。你可以：

- 使用更难的数据集运行此代码，并查看转移学习的更多好处
- 使用此处描述的方法，使用转移学习更新不同的模型，可能在新域（即NLP，音频等）中
- 一旦您对模型感到满意，您可以将其导出为ONNX模型，或使用混合前端跟踪它以获得更快的速度和优化机会。


