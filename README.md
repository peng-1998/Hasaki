# Hasaki
一个用于医学图像分割的工具包。
模块：
* data:包含多个医学图像数据集的Dataset类封装.
* loss:常用的医学图像分割的损失函数.
* score:常用的医学图像分割的评价指标.
* transforms:用于医学图像分割任务的图像变换，基于torchvision.transforms并兼容.

在data模块当中,每一个数据集被封装为一个类,可以选择使用不同的数据子集(训练集、测试集等)以及选择下载。

在transforms模块当中，我们重写了一些带有随机性质的数据增强方法，如随机裁剪，因为这些变换需要图像与标签一起变换，然而torchvision并不能优雅地做到这一点。
此外，我们通过TransformAll为其变换进行兼容，使用TransformPart为序列当中的一部分应用变换。


