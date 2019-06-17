参考资料：

[LSTM通俗讲解1](https://zhuanlan.zhihu.com/p/32085405)

[LSTM参数计算](https://www.cnblogs.com/wushaogui/p/9176617.html)

[LSTM图像分类实例](http://www.ainoobtech.com/pytorch/pytorch-rnn.html)

[LSTM更新参数](https://ilewseu.github.io/2018/01/06/LSTM%E5%8F%82%E6%95%B0%E6%9B%B4%E6%96%B0%E6%8E%A8%E5%AF%BC/)

[很好的解释LSTM空间上](https://blog.csdn.net/weixin_41041772/article/details/88032093)

## LSTM简介

LSTM是为了解决长序列训练过程中梯度消失和梯度爆炸问题产生。

## RNN

![img](https://pic2.zhimg.com/80/v2-71652d6a1eee9def631c18ea5e3c7605_hd.jpg)

## LSTM

- RNN只有一个状态变量
- LSTM有两个状态变量

![img](https://pic4.zhimg.com/80/v2-e4f9851cad426dfe4ab1c76209546827_hd.jpg)

根据输入信号$X$ 和 隐藏层状态$h$ 产生四个状态。

- z ：相当于输入信号
- $z^i$ ：门控制信号
- $z^f$ ：忘记门控
- $z^o$ ：输出门控

![1560681567793](C:\Users\fanzhihao\AppData\Roaming\Typora\typora-user-images\1560681567793.png)

LSTM三个阶段：

- 忘记阶段： 针对上一阶段$c^{t-1}$ ，点乘忘记$z^f$ ，忘记不重要信息。
- 选择记忆阶段：输入信号$z$ 和 信号选择门控$z^i$ 点乘，记住重要信息
- 输出阶段：输出门控信号$z^o$ 

![img](https://pic2.zhimg.com/80/v2-556c74f0e025a47fea05dc0f76ea775d_hd.jpg)

## LSTM实现

LSTMpytorch实现： 

```
CLASS torch.nn.LSTM(*args, **kwargs)
```

以上图为例
输入：

- input_size:输入$x_t$ 的大小 （seq_len, batch inputsize）
- hidden_size: 隐藏层神经元个数
- num_layers: LSTM网络的层数 (空间上的层数)
- bias: 和以往一样
- batch_first: 默认
- dropout:
- bidirectional:

输出： 

- output：（seq_len, batch, num_directions*hidden_size）
- $h_n$ :(num_layers*num_directions, batch, hidden_size)
- $c_n$ :(num_layers*num_directions , batch, hidden_size)

我们以 用LSTM对手写字图像进行分类为例子

```python
# 设置超参数
sequence_length = 28 # 时间序列长度
input_size = 28 # 输入长度
hidden_size = 128 #隐藏层神经元个数
num_layers = 2 # 隐藏层数
num_classes =10 # 输出种类
batch_size = 100 # 批量大小
num_epochs = 2 # 迭代次数
learning_rate = 0.01 # 学习率
```

MNIST手写字图像大小为28x28，我们设置28个时间序列，每个序列输入为28，那么每个序列的输出大小为128（hidden_size大小），因此整个LSTM一个batch的输出为(28, 100，1*128)，对于输出，我们直接用一个线性分类器就可以区分。

## 一个LSTMcell中参数个数

1.1 忘记门参数

![忘记门层](https://shgwumarkdown.oss-cn-shenzhen.aliyuncs.com/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1528854541990.png)

由上可知，$h_{t-1} = 128, x_t = 28$ ,所以将两者拼接起来就为156，又因为输出为$hiddern_size = 128$ ,如果bias为true，该层神经元个数为：$(128+ 28)*28 + 128$

1. 2 cell状态

![确定该时刻细胞要更新的内容](https://shgwumarkdown.oss-cn-shenzhen.aliyuncs.com/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1528855984659.png)

神经元个数：[（128 +28） * 28 + 128]*2

一个是用来更新信息选择门$i_t$ ,一个用于产生输入信号$\tilde{C}_t$

![细胞状态更新](https://shgwumarkdown.oss-cn-shenzhen.aliyuncs.com/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1528856337080.png)

更新$C_{t-1}$ 这一步没有参数需要学习。

1.3 输出层

![输出门层](https://shgwumarkdown.oss-cn-shenzhen.aliyuncs.com/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1528856462439.png)

神经元个数为：（128+28）*28 + 128

**LSTM 时间序列是参数共享的，就是说虽然这里LSTM序列长为28，其实只有cell，每次再这个cell上输入不同，输出不同**

## LSTM更新参数

后向传播分为两部分：

- 时间序列上传播

- 空间上LSTM上传播（numble_layer） 

  **空间上，就是下一层的$h$当作上一层的输入**

  ![img](https://img-blog.csdnimg.cn/20190228155947692.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTA0MTc3Mg==,size_16,color_FFFFFF,t_70)

