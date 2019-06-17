[本文来自于此链接](https://medium.com/@jonathan_hui/gan-a-comprehensive-review-into-the-gangsters-of-gans-part-1-95ff52455672)

在学习一个新知识之前，有必要先了解其中的应用，比如说GAN，以下将会介绍GAN目前的几个应用

##  早期GAN——图像产生

![1560759811695](C:\Users\fanzhihao\AppData\Roaming\Typora\typora-user-images\1560759811695.png)

2014年最早的一篇关于GAN的文章，关注的只是图像的产生

## cross-domain GAN

![1560759888226](C:\Users\fanzhihao\AppData\Roaming\Typora\typora-user-images\1560759888226.png)

这类GAN，研究的是从一domain到另一个domain，比如这里的通过文字生成图片，

![1560759943118](C:\Users\fanzhihao\AppData\Roaming\Typora\typora-user-images\1560759943118.png)

 通过encoder提取文字$y$的特征，接下来操作和以上一样

##   cross-domain transfer

![1560760029498](C:\Users\fanzhihao\AppData\Roaming\Typora\typora-user-images\1560760029498.png)

GAN在风格转换方面的应用，通过一张原始图，生成各种风格的图，比如：Monet style（莫泰风格，什么鬼？？）

![1560760107895](C:\Users\fanzhihao\AppData\Roaming\Typora\typora-user-images\1560760107895.png)

这篇论文很感兴趣，过会再看如何实现的。

## meta-data in generating images

![1560760163350](C:\Users\fanzhihao\AppData\Roaming\Typora\typora-user-images\1560760163350.png)

这里3D重建中的不同视角，通过图像的一个视角，产生另外视角。（有意思）

![1560760238438](C:\Users\fanzhihao\AppData\Roaming\Typora\typora-user-images\1560760238438.png)

## 超分辨率 

![1560760340703](C:\Users\fanzhihao\AppData\Roaming\Typora\typora-user-images\1560760340703.png)

![1560760347632](C:\Users\fanzhihao\AppData\Roaming\Typora\typora-user-images\1560760347632.png)

这个也可以看看，有幸读到这篇文章，太幸运了（2019/6/17）

## Problem

GAN模型的训练并不简单，以下列举常见的几种问题：

- Mode collapse：生成器产什么的图像不具有多样性
- Diminished gradient：辨别器训练的太好，导致梯度消失，生成器学不到什么
- Non-convergence：不收敛，参数变化很大
- 生成器和辨别器不平衡
- 对超参数敏感

## Measurement

GAN的损失函数的大小不能反映图片质量，和模型训练进度。

![1560760795325](C:\Users\fanzhihao\AppData\Roaming\Typora\typora-user-images\1560760795325.png)

目前，一般通过 Inception Score （IS）来评价图像的质量和多样性。