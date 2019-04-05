# 使用PYTORCH进行神经传递

## 介绍

本教程解释了如何实现 Leon A. Gatys，Alexander S. Ecker和Matthias Bethge开发的 [Neural-Style](https://arxiv.org/abs/1508.06576) 算法。神经风格或神经传递允许您拿一副图像并以新的艺术风格再现它。该算法采用input image，content-image 和 style-image 三种图像，并将输入改变为类似于内容图像的内容和样式图像的艺术风格。

## 基本原则

原理很简单：我们定义两个距离，一个用于内容（Dc）和一个风格（Ds）。 Dc 测量两个图像之间的内容有多么不同, Ds 测量两个图像之间的风格有多么不同。然后，我们随机生成第三个图像，即 input ，并对其进行变换，以最小化其与内容图像的内容距离及其与样式图像的样式距离。现在我们可以导入必要的包并开始神经传递。

## 导入包和选择设备

下面是实现神经传递所需的包的列表。

- torch，torch.nn，numpy（使用PyTorch进行神经网络的必备软件包）
- torch.optim （有效梯度下降）
- PIL，PIL.Image，matplotlib.pyplot（加载和显示图像）
- torchvision.transforms （将PIL图像转换为张量）
- torchvision.models （训练或加载预先训练的模型）
- copy （深层复制模型;系统包）


```
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

```

接下来，我们需要选择运行网络的设备并导入内容和样式图像。在大型图像上运行神经传递算法需要更长的时间，并且在GPU上运行时会更快。我们可以使用torch.cuda.is_available() 来检测是否有可用的GPU。接下来，我们将整个教程中的torch.device设置为使用。此外，.to(device)方法用于将张量或模块移动到期望的设备。

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## 加载图像

现在我们将导入 style 和 content 图像。**原始PIL图像的值介于0和255之间，但是当转换为torch张量时，它们的值将转换为介于0和1之间。**图像也需要调整大小以具有相同的尺寸。需要注意的一个重要细节是，torch 库中的神经网络训练的张量值范围从0到1.如果您尝试为网络提供0到255张量图像，则激活的要素图将无法感知预期的内容和风格。然而，来自Caffe库的预训练网络训练有0到255个张量图像。

> 注意
> 以下是下载运行教程所需图像的链接：[picasso.jpg](https://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg) 和 [dancing.jpg](https://pytorch.org/tutorials/_static/img/neural-style/dancing.jpg)。下载这两个图像并将它们添加到当前工作目录中具有名称 images 的目录中。


```
# 所需的输出的大小
imsize = 512 if torch.cuda.is_available() else 128  
# 如果不在GPU上则用小图片。

loader = transforms.Compose([
    transforms.Resize(imsize),  # 缩放导入的图片的尺寸
    transforms.ToTensor()])  # 转换成torch张量


def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    # 增加一个维度，适合网络的张量尺寸
    return image.to(device, torch.float)


style_img = image_loader("./data/images/neural-style/picasso.jpg")
content_img = image_loader("./data/images/neural-style/dancing.jpg")

assert style_img.size() == content_img.size(), \
    "我们需要导入相同大小的样式和内容图像"
# 断言，如果尺寸不同反馈错误
```

现在，让我们创建一个函数，通过将其副本重新转换为PIL格式并使用 plt.imshow 显示副本来显示图像。我们将尝试显示内容和样式图像，以确保它们被正确导入。

```
unloader = transforms.ToPILImage()  
# 重新转换为PIL图像

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # 我们克隆张量以不对其进行更改
    image = image.squeeze(0)      # 删除假批量维度
    image = unloader(image)       # 转化为pil图像
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # 暂停一下，以便更新图表


plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')
```

## 损失函数

### content loss

content loss 是表示单个层的 content 距离的加权版本的函数。该函数获取网络处理输入 X 中的 L 层的特征映射  $F_{XL}$，并返回加权 content距离 $W_{CL}$。

 $D_L^C(X，C)$ 在  X 图像 和 content 图像 C 之间。content 图像（$F_{CL}$）的特征图必须由函数知道，以便计算 content 距离。我们将此函数实现为具有构造函数的 torch 模块，该构造函数将$F_{CL}$ 作为输入。距离$‖F_{XL}−F_{CL}‖^2$是两组特征图之间的均方误差，可以使用nn.MSELoss。

我们将在用于计算 content 距离的卷积层之后直接添加此 content loss 模块。这样，每次向网络馈送输入图像时，将在期望的层计算content loss，并且由于自动梯度，将计算所有梯度。现在，为了使content loss层透明，我们必须定义一个计算content loss的 forward 方法，然后返回图层的输入。计算的损失将保存为模块的参数。


```
class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__() 
        self.target = target.detach()
        # 我们从用于动态计算梯度的树中“detach”目标内容：
        # 这是一个声明的值，而不是变量。否则标准的正向方法将引发错误
        # detach则在backward中计算梯度时不对target之前所在的计算图存在任何影响

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        # 计算损失
        return input
        # 返回输入
```

> 重要细节：尽管此模块已命名ContentLoss，但它不是真正的 PyTorch Loss 功能。如果要将内容 loss 定义为 PyTorch Loss 函数，则必须创建 PyTorch autograd 函数以在 backward 方法中手动重新计算/实现梯度。

### 风格损失 style loss

样式丢失模块与内容丢失模块类似地实现。它将充当网络中的透明层，用于计算该层的样式丢失。

为了计算样式损失，我们需要计算 gram 矩阵$G_{XL}$。gram矩阵是将给定矩阵乘以其转置矩阵的结果。

在该应用中，给定矩阵是层L的特征映射$F_{XL}$的重新形成的版本。将$F_{XL}$重新整形以形成$\hat{F}_{XL}$，即 KxN 矩阵，其中K是层L处的特征映射的数量，并且N是任何矢量化特征映射$F^k_{XL}$的长度。例如，$\hat{F}_{XL}$ 的第一行对应于第一矢量化特征映射$F^1_{XL}$。

最后，必须通过将每个元素除以矩阵中元素的总数来对 gram 矩阵进行归一化。该归一化是为了抵消具有大N维的$\hat{F}_{XL}$矩阵在Gram矩阵中产生较大值的事实。这些较大的值将导致第一层（在合并图层之前）在梯度下降期间产生较大的影响。样式特征往往位于网络的更深层，因此这种规范化步骤至关重要。

```
def gram_matrix(input):
    a, b, c, d = input.size()  
    # a = batch size(=1)
    # b = number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  
    # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  
    # compute the gram product

    return G.div(a * b * c * d)
    # we 'normalize' the values of the gram matrix
    # 通过除以每个 feature maps 中的元素数量。

```

现在，style loss 模块看起来几乎与 content loss 模块完全相同。还使用$G_{XL}$ 和 $G_{SL}$之间的均方误差来计算样式 style 距离。 


```
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
```

## 导入模型

现在我们需要导入一个预先训练好的神经网络。我们将使用19层VGG网络，就像本文中使用的那样。

PyTorch的VGG实现是一个分为两个子`Sequential` 模块的模块：`features`（包含卷积和池化层）和`classifier`（包含完全连接的层）。我们将使用`features`模块，因为我们需要各个卷积层的输出来测量内容和样式loss。某些层在训练期间具有与评估不同的行为，因此我们必须使用`.eval()`将网络设置为评估模式。


```
cnn = models.vgg19(pretrained=True).features.to(device).eval()
# 设置为评估模式，打开预训练，只需要features这个Sequential模块
```

另外，VGG网络在图像上训练，每个 channel 通过 mean= [0.485,0.456,0.406] 和std = [0.229,0.224,0.225] 归一化(normalized)。在将图像发送到网络之前，我们将使用它们对图像进行 normalize。


```
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# 创建一个模块来规范化输入图像，以便我们可以轻松地将其放入 nn.Sequential

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        # 使用 .view 重塑 the mean and std 使它们 [C x 1 x 1] 以便它们可以直接使用图像Tensor of shape [B x C x H x W].
        # B is batch size. 
        # C is number of channels.
        # H is height and W is width.

    def forward(self, img): 
        return (img - self.mean) / self.std
        # normalize img
```

Sequential 模块包含子模块的有序列表。例如，vgg19.features包含以正确的深度顺序对齐的序列（Conv2d，ReLU，MaxPool2d，Conv2d，ReLU ...）。我们需要在他们检测到的卷积层之后立即添加内容loss和样式loss层。为此，我们必须创建一个新的Sequential模块，它正确插入了内容损失和样式损失模块。

```
# 期望的深度层来计算样式/内容损失 :

content_layers_default = ['conv_4']
# 默认的内容损失需要计算的层

style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
# 默认的风格损失需要计算的层

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
   
    cnn = copy.deepcopy(cnn)
    # 拷贝cnn
   
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    # 归一化参数

    content_losses = []
    style_losses = []
    # 只是为了拥有可迭代的访问权限的内容/样式损失列表
    

    model = nn.Sequential(normalization)
    # 创建一个新的nn.Sequential来放入应该按顺序激活的模块

    i = 0  
    # 每当我们看到一个转换时增量
    
    for layer in cnn.children():
    # 从cnn中提取出层
        if isinstance(layer, nn.Conv2d):
        # 类似 type ,判断是否是 nn.Conv2d 类型
            i += 1
            name = 'conv_{}'.format(i)
            # 名字是 conv_i
        
        elif isinstance(layer, nn.ReLU):
        # 如果不是，而是和nn.ReLU一个类型
        
            name = 'relu_{}'.format(i)
            # 名字是 relu_i
            
            layer = nn.ReLU(inplace=False)
            # （in-place）版本与我们在下面插入的 ContentLoss 和 StyleLoss 不能很好地协作。所以我们在这里替换（out-of-place）。
           
        elif isinstance(layer, nn.MaxPool2d):
        # 如果不是，而是nn.MaxPool2d类型
            name = 'pool_{}'.format(i)
            
        elif isinstance(layer, nn.BatchNorm2d):
        # 如果不是，而是nn.BatchNorm2d类型
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        # 除此以外，输出错误的层：xxx

        model.add_module(name, layer)
        # 我们的模型添加模块

        if name in content_layers:
        # 如果 name 在 'conv_4'中：
        
            target = model(content_img).detach()
            # 在当前层，得到features
            
            content_loss = ContentLoss(target)
            # content_loss 实例
            # 定义.loss 属性，将用于计算与input损失。
            model.add_module("content_loss_{}".format(i), content_loss)
            # 模型加入模块
            
            content_losses.append(content_loss)
            # 可迭代的content损失 访问列表

        if name in style_layers:
        # 如果名字在 ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
            target_feature = model(style_img).detach()
            # 关闭梯度
            
            style_loss = StyleLoss(target_feature)
            # style_loss实例
            # 定义.loss 属性，将用于计算与input损失。           
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

   
    for i in range(len(model) - 1, -1, -1):
    # 递减
    
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
        # 从倒数开始，如果在model的第i层，如果只要出现contentloss和styleloss层，就打断，直接执行最后的代码。
        
            break
    model = model[:(i + 1)]
    # 裁减最后几个无相关的层
    return model, style_losses, content_losses
```

接下来，我们选择输入图像。您可以使用内容图像或白噪声的副本。


```
input_img = content_img.clone()
# 如果您想使用白噪声而取消注释以下行：
# input_img = torch.randn(content_img.data.size(), device=device)

# 将原始输入图像添加到图中：
plt.figure()
imshow(input_img, title='Input Image')
```

## 梯度下降

正如此算法的作者Leon Gatys所建议的[那样](https://discuss.pytorch.org/t/pytorch-tutorial-for-neural-transfert-of-artistic-style/336/20?u=alexis-jacq)，我们将使用L-BFGS算法来运行梯度下降。与 training 网络不同，我们希望训练输入图像以最小化内容/风格损失。我们将创建一个PyTorch L-BFGS优化器 optim.LBFGS 并将我们的图像传递给它作为要优化的张量。

```
def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    # 定义输入的图片的数据是需要优化的参数
    return optimizer
```

最后，我们必须定义一个执行神经传递的函数。对于网络的每次迭代，它被馈送更新的输入并计算新的损失。我们将运行每个loss模块的 backward 来动态计算它们的梯度。优化器需要一个“closure闭包”函数，它重新评估模块并返回损失。

我们还有一个最后的约束要解决。网络可以尝试使用超过图像的0到1张量范围的值来优化输入。我们可以通过在每次运行网络时将输入值更正为0到1来解决此问题。


```
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    # 得到模型，风格损失，内容损失    
        
        
    optimizer = get_input_optimizer(input_img)
    # 打开优化器 ，优化输入图片

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
    # 迭代
    
        def closure():
       # 优化器需要一个“closure闭包”函数，它重新评估模块并返回损失。
            # 更正更新的输入图像的值
            input_img.data.clamp_(0, 1)
            # 正则化

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
            # 取出实例
                style_score += sl.loss
                # 分批计算损失
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight
            # 乘以对应权重
            
            loss = style_score + content_score
            # 损失等于两者相加
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

   
    input_img.data.clamp_(0, 1)
    # 最后的修正...
    
    return input_img
```

最后我们运行这个算法：

```
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

plt.figure()
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()
```

