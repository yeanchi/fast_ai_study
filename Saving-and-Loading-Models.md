[toc]

# 保存和加载模型

本文档提供了有关PyTorch模型保存和加载的各种用例的解决方案。

涉及到在保存和加载模型时，有三个核心功能需要熟悉：

1. [torch.save](https://pytorch.org/docs/stable/torch.html?highlight=save#torch.save)：将序列化对象保存到磁盘。该函数使用Python的 pickle 实用程序进行序列化。使用此功能可以保存各种对象的模型，张量和字典。


2. [torch.load](https://pytorch.org/docs/stable/torch.html?highlight=torch%20load#torch.load)：使用 [pickle](https://docs.python.org/3/library/pickle.html) 的unpickling工具将pickle对象文件反序列化到内存中。此功能还有助于数据加载到设备。（请参阅最后 [跨设备保存和加载模型]）。


3. [torch.nn.Module.load_state_dict](https://pytorch.org/docs/stable/nn.html?highlight=load_state_dict#torch.nn.Module.load_state_dict)：使用反序列化的`state_dict` 加载模型的参数字典 。

> 额外：
> 
> 要序列化对象层次结构，只需调用该dumps()函数即可。
> 同样，要对数据流进行反序列化，请调用该loads()函数。

## 什么是 stata_dict

在PyTorch中，`torch.nn.Module` 模型的可学习参数（权重和偏差）包含在模型的参数中。（ 通过访问 `model.parameters()` ）

`state_dict`只是一个Python dictionary对象，它将每个层映射到它的参数张量。

注意，只有具有可学习参数的层 (卷积层、线性层等) 和被寄存的缓存区 ( batchnorm 的 running_mean ) 在模型的 `state_dict` 中有条目。

> running_mean和running_var不是可以学习的变量，是统计量，只是训练过程对很多batch的数据统计。


优化器对象(torch.optim)也有一个 `state_dict` ，其中包含有关优化器状态以及使用的超参数的信息。

因为state_dict对象是Python字典，所以可以很容易地保存、更新、修改和恢复它们，从而为PyTorch模型和优化器增加了大量的模块化。

例：

让我们看一下 [Training a classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py) 教程中使用的简单模型中的`state_dict`。

```
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = TheModelClass()
# 初始化模型


optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# 初始化优化器

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
# 打印出模型的 state_dict

print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
# 打印出优化器的 state_dict
    
```



输出：

```
Model's state_dict:
conv1.weight     torch.Size([6, 3, 5, 5])
conv1.bias   torch.Size([6])
conv2.weight     torch.Size([16, 6, 5, 5])
conv2.bias   torch.Size([16])
fc1.weight   torch.Size([120, 400])
fc1.bias     torch.Size([120])
fc2.weight   torch.Size([84, 120])
fc2.bias     torch.Size([84])
fc3.weight   torch.Size([10, 84])
fc3.bias     torch.Size([10])

Optimizer's state_dict:
state    {}
param_groups     [{'lr': 0.001, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [4675713712, 4675713784, 4675714000, 4675714072, 4675714216, 4675714288, 4675714432, 4675714504, 4675714648, 4675714720]}]
```

## 保存和加载推理模型

保存/加载 state_dict

保存：

```
torch.save(model.state_dict(), PATH)
```

加载：

```
model = TheModelClass(*args, **kwargs)
# 初始化模型

model.load_state_dict(torch.load(PATH))
# load_state_dict 必须加载字典对象

model.eval()
# 模型改为评估模式
```

使用torch.save()函数保存模型的 state_dict 将为您以后恢复模型提供最大的灵活性。

一个常见的PyTorch约定是使用.pt或.pth文件扩展名保存模型。

请记住，在运行推理之前，必须调用model.eval()将dropout和批处理规范化层设置为评估模式。如果不这样做，将产生不一致的推理结果。

> 注意：
> 请注意，该load_state_dict()函数采用字典对象，而不是保存对象的路径。
> 这意味着在将保存的state_dict传递给load_state_dict()函数之前 ，必须对其进行反序列化 torch.load(PATH)。
> 例如，您无法加载使用 model.load_state_dict(PATH)。

## 保存和加载整个模型

保存：

```
torch.save(model, PATH)
# 直接读取路径

```


加载：

```
# 模型类必须在某处定义
model = torch.load(PATH)
# 直接读取路径

model.eval()

```
此保存/加载过程使用最直观的语法并涉及最少量的代码。以这种方式保存模型将使用Python的pickle模块保存整个模块。

这种方法的缺点是序列化数据绑定到特定类以及保存模型时使用的确切目录结构。**这是因为pickle不保存模型类本身。**相反，它会保存包含类的文件的路径，该文件在加载时使用。**因此，当您在其他项目中或在重构之后使用时，您的代码可能会以各种方式中断。**

一个常见的PyTorch约定是使用.pt或.pth文件扩展名保存模型。

> 请记住，model.eval()在运行推理之前，必须先调用dropout和batch normalization图层到评估模式。如果不这样做，将导致不一致的推理结果。


##  保存和加载一个通用检查点
用于推理且或恢复训练

保存：

```
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)

```

加载：

```
model = TheModelClass(*args, **kwargs)
# 初始化模型

optimizer = TheOptimizerClass(*args, **kwargs)
# 初始化优化器

checkpoint = torch.load(PATH)
# 加载路径

model.load_state_dict(checkpoint['model_state_dict'])
# 模型的 model_state_dict 加载

optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# 优化器 optimizer_state_dict 参数加载

epoch = checkpoint['epoch']
# 迭代数

loss = checkpoint['loss']
# 损失

model.eval()
# - or -
model.train()
# 可以选择加载训练模式或者评估模式

```

保存优化器的state_dict也很重要，因为它包含作为模型训练更新的缓冲区和参数。

您可能想要保存的其他项目是您停止使用的迭代，最新记录的训练损失，外部torch.nn.Embedding 图层等。

要保存多个组件，请将它们组织在字典中并用于 torch.save()序列化字典。常见的PyTorch约定是使用.tar文件扩展名保存这些检查点。

要加载项目，首先初始化模型和优化器，然后使用本地加载字典torch.load()。从这里，您可以通过简单地查询字典来轻松访问已保存的项目。

> 请记住，model.eval()在运行推理之前，必须先调用dropout和batch normalization图层到评估模式。如果不这样做，将导致不一致的推理结果。如果您希望恢复培训，请引用 model.train() 以确保这些图层处于培训模式。

## 在一个文件中保存多个模型

保存：

```
torch.save({
            'modelA_state_dict': modelA.state_dict(),
            'modelB_state_dict': modelB.state_dict(),
            'optimizerA_state_dict': optimizerA.state_dict(),
            'optimizerB_state_dict': optimizerB.state_dict(),
            ...
            }, PATH)
```

加载：

```
modelA = TheModelAClass(*args, **kwargs)
modelB = TheModelBClass(*args, **kwargs)
optimizerA = TheOptimizerAClass(*args, **kwargs)
optimizerB = TheOptimizerBClass(*args, **kwargs)

checkpoint = torch.load(PATH)
modelA.load_state_dict(checkpoint['modelA_state_dict'])
modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])

modelA.eval()
modelB.eval()
# - or -
modelA.train()
modelB.train()

```

保存由多个模型组成的模型torch.nn.Modules（例如GAN，seq2seq或模型集合）时，您将遵循与保存常规检查点时相同的方法。换句话说，保存每个模型的state_dict和相应的优化器的字典。如前所述，您可以通过简单地将它们附加到字典中来保存任何其他可能有助于您恢复培训的项目。

常见的PyTorch约定是使用.tar文件扩展名保存这些检查点 。

要加载模型，首先初始化模型和优化器，然后使用本地加载字典torch.load()。从这里，您可以通过简单地查询字典来轻松访问已保存的项目

> 请记住，model.eval()在运行推理之前，必须先调用dropout和batch normalization图层到评估模式。如果不这样做，将导致不一致的推理结果。如果您希望恢复培训，请引用 model.train() 以确保这些图层处于培训模式。


## 使用来自不同模型的参数对模型进行预热

保存：

```
torch.save(modelA.state_dict(), PATH)
```

加载： 

```
modelB = TheModelBClass(*args, **kwargs)
modelB.load_state_dict(torch.load(PATH), strict=False)
# 加载时忽略不匹配的键

```

在转移学习或训练新的复杂模型时，部分加载模型或加载部分模型是常见的情况。利用经过训练的参数，即使只有少数可用，也有助于热启动训练过程，并希望帮助您的模型比从头开始训练更快地收敛。

无论您是从缺少一些键的部分state_dict加载，还是加载一个包含比正在加载的模型更多键的state_dict，**您都可以在load_state_dict()函数中将strict参数设置为False，以忽略不匹配的键。**

如果要将参数从一个层加载到另一个层，**但某些键不匹配，只需更改正在加载的state_dict中的参数键的名称**， 以匹配要加载到的模型中的键。

## 跨设备保存和加载模型

### 保存用GPU，在CPU上加载

保存：

```
torch.save(model.state_dict(), PATH)
```

加载：

```
device = torch.device('cpu')

model = TheModelClass(*args, **kwargs)

model.load_state_dict(torch.load(PATH, map_location=device))
# 将张量底层的存储动态地重新映射到CPU设备。
```

当你在CPU上加载在GPU上训练的模型时，**请传递 torch.device('cpu')给map_location函数中的 torch.load()参数**。

在本例中，使用map_location参数将张量底层的存储动态地重新映射到CPU设备。

### 用CPU存储，在GPU上加载

保存：

```
torch.save(model.state_dict(), PATH)
```

加载：

```
device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0")) 
# 这会将模型加载到给定的GPU设备。

model.to(device)
# 将模型的参数张量转换为CUDA张量。


# 确保在您提供给模型的任何输入张量上调用 
# input = input.to(device) 手动覆盖张量

```

在已训练并保存在CPU上的GPU上加载模型时，请将`map_location`函数中的参数设置 `torch.load()` 为 `cuda：device_id`。这会将模型加载到给定的GPU设备。

接下来，请务必调用 `model.to(torch.device('cuda'))` 将模型的参数张量转换为CUDA张量。

最后，确保在所有模型上输入`.to(torch.device('cuda'))` 函数，来为CUDA优化模型准备数据。

请注意，调用 `my_tensor.to(device)` 会在GPU上返回 `my_tensor`的新副本。它不会覆盖` my_tensor`。

因此，请记住手动覆盖张量： 
`my_tensor = my_tensor.to(torch.device('cuda'))`



### 保存用GPU上，在GPU上加载

保存：

```
torch.save(model.state_dict(), PATH)
```

加载：

```
device = torch.device("cuda")
# 设置设备为GPU

model = TheModelClass(*args, **kwargs)

model.load_state_dict(torch.load(PATH))

model.to(device)

# 确保在您提供给模型的任何输入张量上调用 
# input = input.to(device) 手动覆盖张量


```


在GPU上训练并保存在GPU上的模型时，只需将初始化model模型转换为CUDA优化模型即可 `model.to(torch.device('cuda'))`。

**此外，请务必`.to(torch.device('cuda'))`在所有模型输入上使用该功能来准备模型的数据。**

请注意，调用 `my_tensor.to(device)` 会在GPU上返回 `my_tensor`的新副本。它不会覆盖` my_tensor`。

因此，请记住手动覆盖张量： 
`my_tensor = my_tensor.to(torch.device('cuda'))`

## 保存torch.nn.DataParallel模型

保存：

```
torch.save(model.module.state_dict(), PATH)
```

加载：

```
# Load to whatever device you want
```

`torch.nn.DataParallel`是一个模型包装器，可以实现并行GPU的使用。通常要保存`DataParallel`模型，请保存 `model.module.state_dict()`。这样，您就可以灵活地以任何方式加载模型到您想要的任何设备。

> 文章来源：[Saving and Loading Models — PyTorch Tutorials 1.0.0.dev20190327 documentation](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices)
> 原作者： [Matthew Inkawhich](https://github.com/MatthewInkawhich)

