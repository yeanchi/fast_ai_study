# 什么是 torch.nn ？

为了更好的理解，我们将在不使用这些模型的任何特征的情况下，在MNIST数据集上训练基本神经网络。

我们最初只会使用最基本的PyTorch张量功能。然后，我们将逐步从增加一个功能`torch.nn`，`torch.optim`，`Dataset`，或 `DataLoader` 。以精确地显示每个片段的功能，以及如何使代码更简洁或更灵活。

## MNIST数据设置

我们将使用经典 [MNIST](http://deeplearning.net/data/mnist/) 数据集，它由手绘数字(0到9)的黑白图像组成。

我们将使用 pathlib 处理路径(Python 3标准库的一部分)，并使用 [requests](http://docs.python-requests.org/en/master/) 下载数据集。我们只在使用模块时导入它们，这样您就可以确切地看到在每个点上使用的是什么。

```
from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
# 路径设置

PATH.mkdir(parents=True, exist_ok=True)
# 生成路径

URL = "http://deeplearning.net/data/mnist/"
# 文件来源链接

FILENAME = “mnist.pkl.gz"
# 文件名

if not (PATH / FILENAME).exists():
# 检查文件是否存在 os.path.exists 
# 如果不存在
        content = requests.get(URL + FILENAME).content
        # 下载文件
        (PATH / FILENAME).open("wb").write(content)
        # 传入标识符  'wb' 表示文本写入二进制文件

```

该数据集采用 numpy 数组格式，并使用 [pickle](https://docs.python.org/3/library/pickle.html) 存储，pickle 是用于序列化数据的特定于 python 的格式，我们需要 unpickle 。

```
import pickle
import gzip 

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
# as_posix(). 将Windows 路径分隔符'\' 改为Unix 样式'/'。 

        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
# 使用encoding='latin1' unpickle NumPy数组和由python2 pickle的datetime、date和time实例是必需的。              
```

每个图像大小为28x28，存储为长度为784 (=28x28)的扁平数组。我们先查看一张图，需要先重塑为2维。

```
from matplotlib import pyplot
import numpy as np

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
print(x_train.shape)
```

PyTorch使用torch.tensor而不是numpy数组，因此我们需要转换数据。

```
import torch

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())
```


## 从无到有的神经网络（没有 torch.nn）

我们随机生成张量，创建线性模型的权重和偏差。

注意：我们在这里用[Xavier](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)初始化初始化权重（乘以 1/sqrt(n)).）


```
import math

weights = torch.randn(784, 10) / math.sqrt(784)

weights.requires_grad_()
# 初始化之后打开梯度，以防初始化之后的计算算入梯度
# 在PyTorch中使用一个尾随的 `_` 表示操作是就地执行的。

bias = torch.zeros(10, requires_grad=True)

```

你可以使用PyTorch提供的激活函数和loss函数，也可以自己编写一个。

log_softmax（负对数似然）,在softmax的结果上再做多一次log运算。

$$ log\left(\frac{exp(x_i)}{\sum_jexp(x_i)}\right) $$

以下版本等价于等价log(softmax(x))，优点速度更快更稳定。
具体可以查看：[torch.nn — PyTorch master documentation](https://pytorch.org/docs/stable/nn.html#torch.nn.functional.log_softmax)


```
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)
    # @表示点积运算
```
我们将直接使用 model 预测结果，这是个前向传播，因为用了随机权重，结果不比乱猜测好多少。

```
bs = 64 
# 批次大小 

xb = x_train[0:bs] 
# 设置小批次

preds = model(xb)  
# 预测

print(preds[0], preds.shape)
# 打印preds第一行，和以及preds的形状。
# 你会发现preds不光包含预测的张量值，还有返回的梯度函数。
```

让我们实现 log-likelihood（最大似然估计） 用作损失函数（同样，我们可以使用标准Python）：

```
def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll
```

让我们用我们的随机模型检查我们的损失，这样我们就可以看到我们是否在一个backprop传递之后有所改进。

```
yb = y_train[0:bs]
print(loss_func(preds, yb))
# 拿出一个数据来看看效果
    
```

我们来实现一个函数来计算模型的准确度。对于每个预测，如果值最大的索引与目标值匹配，则预测是正确的。


```
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    # 返回最大值的索引
    return (preds == yb).float().mean()
```


```
print(accuracy(preds, yb))
# 让我们拿一个数据测试下随机权重模型效果
```

我们现在开始运行一个训练循环。对于每次迭代，我们将：
- 使用小批量数据（size sb）
- 用model进行预测
- 计算loss
- loss.backward()更新模型的梯度(权重和损失）



```
from IPython.core.debugger import set_trace
# Python调试器单步执行Pytorch代码，允许您在每一步检查各种变量值。

lr = 0.5  
# 学习速率
epochs = 2 
# 迭代两次

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        # set_trace()
        # 取消以上注释，可以每一步检查各种变量值。
        
        start_i = i * bs
        # 每次迭代数据开始索引
        
        end_i = start_i + bs
        # 每次迭代数据结尾索引
        
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        # 小批量：取出批量数据和标签
        
        pred = model(xb)
        loss = loss_func(pred, yb)
        # 模型的预测和损失

        loss.backward()
        # 计算梯度
        
        with torch.no_grad():
        # 更新权重时关掉梯度
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            
            weights.grad.zero_()
            bias.grad.zero_()
            # 权重更新好之后把梯度归零，以免下次迭代，继续叠加计算梯度。
```

检测下损失和准确率：

```
print(loss_func(model(xb), yb), accuracy(model(xb), yb))
```

## 使用 torch.nn.functional

重构代码，使其执行与之前相同的操作,利用PyTorch的nn类使其更加简洁和灵活。

首先，使用将手写的激活函数和loss函数替换为来自`torch.nn.functional`函数（通常按照约定导入的命名空间f），从而缩短代码。

这个模块包含了torch.nn 库中的所有函数（而库的其他部分包含类）。

这里还有一些创建神经网络的便捷函数，比如 pooling 函数。虽然也有函数用于卷积、线性层等，但是我们将看到，使用库的其他部分通常可以更好的处理这些函数。

如果你使用负对数似然损失函数（negative log likelihood loss ）记录softmax激活。

那么Pytorch提供了一个[F.cross_entropy](https://pytorch.org/docs/stable/nn.html#cross-entropy)将两者结合起来的功能。所以我们甚至可以从我们的模型中删除激活函数。


```
import torch.nn.functional as F

loss_func = F.cross_entropy
# 该标准将log_softmax和nll_loss组合在一个函数中。

def model(xb):
    return xb @ weights + bias

```

注意，我们不再在模型函数中调用`log_softmax`。我们确认一下我们的损失和准确性和之前一样:

```
print(loss_func(model(xb), yb), accuracy(model(xb), yb))
```

## 使用 nn.Module 重构

接下来，我们将使用`nn.Module`和`nn.Parameter`，以获得更清晰，更简洁的训练循环。

我们子类`nn.Module`（它本身是一个类，能够跟踪状态）。在这种情况下，我们想要创建一个包含前向步骤的权重，偏差和方法的类。 我们将使用`nn.Module`许多属性和方法（例如`.parameters()`和`.zero_grad()`）。

 
> 注意：nn.Module（大写字母M）是一个PyTorch特定的概念，是一个“类(class)”。nn.Module不要与（小写m）[model](https://docs.python.org/3/tutorial/modules.html) 的Python概念混淆，后者是可以导入的Python代码文件。   

```
from torch import nn

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        # 在子类中调用父类的某个已经被覆盖的方法。
        
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))
        # 当它们被指定为模块属性时，它们被自动添加到它的参数列表中,并且会出现在Parameters()迭代器中。
        
    def forward(self, xb):
        return xb @ self.weights + self.bias
```

参考:[8.7 调用父类方法 — python3-cookbook 3.0.0 文档](https://python3-cookbook.readthedocs.io/zh_CN/latest/c08/p07_calling_method_on_parent_class.html)


我们现在使用的是对象object而不是仅仅使用函数，因此我们首先必须实例化我们的模型：

```

model = Mnist_Logistic()

```

和函数一样，Pytorch会自动调用我们的forward方法。

```
print(loss_func(model(xb), yb))
```

更新参数值：


```
with torch.no_grad():
    for p in model.parameters(): p -= p.grad * lr
    # 取出权重值减去梯度
    model.zero_grad()
```

我们将把我们的小训练循环包装在一个fit函数中，以便稍后再运行它。


```
def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()

fit()
```

## 使用nn.Linear重构

我们继续重构我们的代码，而不是手动定义和初始化self.weights和self.bias计算。并计算 `xb @ self.weights + self.bias`

而是使用Pytorch类[nn.Linear](https://pytorch.org/docs/stable/nn.html#linear-layers)来处理一个线性层，它为我们完成了所有这些工作。

```
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)
```

实例化模型，输出损失

```
model = Mnist_Logistic()
print(loss_func(model(xb), yb))
```

训练模型：

```
fit()
print(loss_func(model(xb), yb))
```

## 重复使用 optim

Pytorch还有一个包含各种优化算法的包torch.optim。我们可以使用step优化器中的方法来执行前进步骤，而不是手动更新每个参数。

这将让我们取代之前的手动编码优化步骤：


```
with torch.no_grad():
    for p in model.parameters(): p -= p.grad * lr
    model.zero_grad()
```

而只需要：

```
opt.step()
opt.zero_grad()
```
我们将定义一个小函数来创建我们的模型和优化器。

注意：optim.zero_grad()将渐变重置为0，我们需要在计算下一个小批量的渐变之前调用它。

```
from torch import optim
```

```
def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)
    # 返回模型，和传递优化器参数

model, opt = get_model()
# 使用以上函数

print(loss_func(model(xb), yb))

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
    # n为数据个数，bs是批次大小
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        # 取出批次大小的数据
        
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        # step优化器
        opt.zero_grad()
        # 将渐变重置为0
        
print(loss_func(model(xb), yb))
```

## 使用Dataset重构

PyTorch有一个抽象的Dataset类。数据集可以是任何具有一个`__len__` 函数（由Python的标准len函数调用）。和一个作为索引方法的`__getitem__`函数的数据集。

本教程将介绍创建自定义 `FacialLandmarkDataset` 类作为 `Dataset` 子类的一个很好的示例。

PyTorch的TensorDataset是一个包装张量的数据集。通过定义长度和索引方法，这也为我们提供了一种沿着张量的第一维迭代、索引和切片的方法。这将使我们在训练时更容易地同时访问**自变量**和**因变量**。

```
from torch.utils.data import TensorDataset
```

x_train和y_train可以并在一个单一的组合TensorDataset，这将是容易遍历和切片。


```
train_ds = TensorDataset(x_train, y_train)
```

以前，我们必须分别迭代x和y值的小批量：

```
xb = x_train[start_i:end_i]
yb = y_train[start_i:end_i]
```

现在，我们可以一起完成这两个步骤：

```
xb,yb = train_ds[i*bs : i*bs+bs]
```

```
model, opt = get_model()

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        xb, yb = train_ds[i * bs: i * bs + bs]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))
```

## 使用DataLoader重构

Pytorch 的 DataLoader负责管理批次。您可以从任何Dataset创建一个DataLoader。DataLoader使遍历批变得更容易。DataLoader不需要使用`train_ds[i*bs: i*bs+bs]`，而是自动为我们提供每个小批处理。

```
from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
# 使用TensorDataset，放入训练和标签
train_dl = DataLoader(train_ds, batch_size=bs)
# 使用DataLoader 生成批次数据
```

（xb，yb）是从数据加载器自动加载的，不需要再用`for i in range((n-1)//bs + 1)`。

```
for xb,yb in train_dl:
    pred = model(xb)
```

```
model, opt = get_model()

for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))

```
由于Pytorch的nn.Module，nn.Parameter，Dataset，和DataLoader，我们的训练循环现显着更小，更容易理解。

现在让我们尝试添加在实践中创建有效模型所需的基本功能。

## 添加验证集

在第1部分中，我们只是尝试设置合理的训练循环以用于我们的训练数据。实际上，您总是应该有一个验证集，以确定您是否过度拟合。

改变训练数据 对于防止批次与过度拟合之间的相关性非常 重要。另一方面，无论我们是否对验证集进行洗牌，验证损失都是相同的。由于改组需要额外的时间，因此对验证数据进行洗牌是没有意义的。

我们将使用批量大小作为验证集，其大小是训练集的两倍。这是因为验证集不需要反向传播，因此占用的内存较少（不需要存储渐变）。我们利用这一点来使用更大的批量大小并更快地计算损失。


```
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)
```

我们将在每个迭代结束时计算并打印验证损失。

（请注意，我们总是在训练之前调用model.train()和在评估之前调用model.eval() ，因为这些是由例如nn.BatchNorm2d 和 nn.Dropout之类的层使用它们来确保这些不同阶段的适当行为。）


```
model, opt = get_model()

for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        # 更新梯度一次
        opt.zero_grad()

    model.eval()
    # 打开评估模式
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

    print(epoch, valid_loss / len(valid_dl))
```


## 创建 fit() 和 get_data()

现在我们自己来做一点重构。由于我们经历了两次计算训练集和验证集的损失类似的过程，所以让我们将其转换为自己的函数loss_batch，**它计算一个批的损失**。

我们为训练集传递一个优化器，并使用它来执行backprop。对于验证集，我们不传递优化器，因此该方法不执行backprop。


```
def loss_batch(model, loss_func, xb, yb, opt=None):
# 传入模型、损失度量函数、小批量的数据和标签、优化器 

    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)
```

`fit` 运行必要的操作来训练我们的模型并计算每个时期的训练和验证损失。

```
import numpy as np

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)
```

`get_data` 返回训练和验证集的数据加载器。

```
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )
```
现在，我们获取数据加载器和拟合模型的整个过程可以在3行代码中运行：


```
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
```

## 切换到CNN

现在我们要构建一个有三个卷积层的神经网络。由于上一节中的函数都没有假定任何关于模型形式的内容，所以我们将能够使用它们来训练CNN，而无需进行任何修改。

我们将使用Pytorch预定义的Conv2d类作为卷积层。我们定义了一个具有3个卷积层的CNN。每个卷积后面都有一个ReLU。最后，我们执行一个average pooling。（注意这view是PyTorch的numpy版本 reshape）


```
class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        # 重组成四维
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))
        # 展开为一列

lr = 0.1
```

[动量](https://cs231n.github.io/neural-networks-3/#sgd)是随机梯度下降的一种变化，它也考虑了以前的更新，通常会导致更快的训练。

```
model = Mnist_CNN()

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)
```

## nn.Sequential

`torch.nn` 还有一个方便的类，我们可以用它来简单地编写代码:[sequence](https://pytorch.org/docs/stable/nn.html#torch.nn.Sequential)。`Sequential`对象以**顺序的方式运行其中包含的每个模块**。这是一种更简单的方法来写我们的神经网络。

为了利用这一点，我们需要能够轻松地从给定函数定义**自定义层**。

例如，PyTorch没有 view layer，我们需要为我们的网络创建一个view layer。`Lambda`将创建一个层，然后我们可以使用它来定义一个具有`Sequential`的网络。


```
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def preprocess(x):
    return x.view(-1, 1, 28, 28)
```

创建的模型Sequential很简单：


```
model = nn.Sequential(
    Lambda(preprocess),
    # view layer 
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(4),
    Lambda(lambda x: x.view(x.size(0), -1)),
    # 返回x变成一行向量
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)
```

## 包装 DataLoader

我们的CNN相当简洁，但它只适用于MNIST，因为：

- 它假设输入是28 * 28长矢量
- 它假设最终CNN网格大小为4 * 4(因为这是我们使用的平均池内核大小)

让我们去掉这两个假设，我们的模型适用于任何二维单通道图像。首先，我们可以删除初始Lambda层，但将数据预处理移动到生成器中:

```
def preprocess(x, y):
    return x.view(-1, 1, 28, 28), y
    # 预加工、预处理


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
# 得到批量数据
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)
```

接下来，我们可以替换nn.AvgPool2d为nn.AdaptiveAvgPool2d，这允许我们定义我们想要的输出张量的大小，而不是我们所拥有的输入张量。因此，我们的模型将适用于任何大小的输入。


```
model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
```
我们来试试吧：


```
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
```

## 使用GPU

首先检查你的GPU是否在Pytorch中工作：
```
print(torch.cuda.is_available())
```
然后为它创建一个设备对象：

```
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
```

让我们更新preprocess以将批次移动到GPU：


```
def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)


train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)
```

您应该发现它现在运行得更快：


```
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
```

```
model.to(dev)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
```

# 结束思想

那么让我们总结一下我们所看到的：

- torch.nn
  - Module：创建一个可调用的（callable），其行为类似于函数，但也可以包含状态（例如神经网络层权重）。它知道Parameter里包含什么，并且可以将所有渐变归零，循环通过它们以进行权重更新等。
  - Parameter：张量的包装器，告诉 Module 它在backprop 期间它有需要更新的权重。仅更新具有requires_grad 属性集的张量。
  - functional：一个模块（通常F按惯例导入到命名空间中），它包含激活函数，损失函数等，以及层的非有状态版本，如卷积层和线性层。
- torch.optim：包含优化程序，例如SGD，更新Parameter 向后步骤中的权重。
- Dataset：一个对象的抽象接口，其中包含带有带有__len__和__getitem__对象，包括用Pytorch提供的类，如TensorDataset。
- DataLoader：获取任何Dataset并创建一个返回批量数据的迭代器。

