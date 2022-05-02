# Hasaki

## 简介
一个用于医学图像分割的工具包.
模块:
* data: 包含多个医学图像数据集的Dataset类封装.
* loss: 常用的医学图像分割的损失函数.
* score: 常用的医学图像分割的评价指标.
* transforms: 用于医学图像分割任务的图像变换, 基于torchvision.transforms并兼容.
* utils: 包含一些方便的小工具，如颜色归一化, 连通域分割等.
* visualization: 包含一些简单的可视化工具, 目前重要是标签图片的染色方法.
  
在data模块当中, 每一个数据集被封装为一个类, 可以选择使用不同的数据子集(训练集, 测试集等)以及选择下载.
data模块用到了许多其他库，很多库可能用不到，可以选择性安装根据需要使用内容(阅读源代码).

在transforms模块当中, 我们重写了一些带有随机性质的数据增强方法, 如随机裁剪, 因为这些变换需要图像与标签一起变换, 然而torchvision并不能优雅地做到这一点.
此外，我们通过TransformAll为其变换进行兼容，使用TransformPart为序列当中的一部分应用变换.

注意：
* utils模块的labeling_2d(连通域分割)默认使用[cc_torch](https://github.com/zsef123/Connected_components_PyTorch) 作为执行后端,但是此项目要求在CUDA上执行. 如果想要让这个函数执行在CUDA上请保证:您的计算机安装了CUDA工具(通过执行nvcc -V 命令检查),您的PyTorch使用相同版本的CUDA进行编译. 安装请阅读[Connected_components_PyTorch/readme.md]{https://github.com/peng-1998/Connected_components_PyTorch/blob/1b3da42b821e9e43e7cbcfc54821e642c188a0c0/readme.md} . 如果无法在CUDA上执行,labeling_2d将会使用scipy库当中的函数作为后端.

## 使用 
将Hasaki克隆到项目文件夹下就可以使用

## python 版本
主分支基于python3.10,一些语法与python3.10之前的版本不兼容,请克隆python38分支如果使用python3.10之前的版本.
```bash
git clone -b python38 https://github.com/peng-1998/Hasaki
```