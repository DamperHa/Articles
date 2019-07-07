> 日期：2019/6/12

[参考地址1](https://zhuanlan.zhihu.com/p/34395749)

argparse模块可以轻松编写用户命令行接口；

### 基本步骤

1. 构建ArgumentParser
2. 通过args.add_argument添加参数
3. 通过args.parse_args解析参数

在parser种有两种参数，

- 可选参数：可选参数通过"-"表示
- 位置参数

通过以上三步构造，我们可以通过``arg.参数名`` 对参数调用 。

在深度学习程序种，我们通过arg参数形式，对超参进行设定

```python
import sys
import argparse

def cmd():
    # description为程序的描述信息
    # epilog为程序描述信息
    # 其他的参数默认即可
    args = argparse.ArgumentParser(description = 'Personal Information ',epilog = 'Information end ')
    #必写属性,第一位
    # argparse中有两种参数，一种可选参数，一种位置参数，可选参数用“-”，位置参数不用参数
    args.add_argument("name",         type = str,                  help = "Your name")
    #必写属性,第二位
    args.add_argument("birth",        type = str,                  help = "birthday")
    #可选属性,默认为None
    #dest为参数的一个别名
    args.add_argument("-r",'--race',  type = str, dest = "race",   help = u"民族")
    #可选属性,默认为0,范围必须在0~150
    #choices设定参数值取值范围
    args.add_argument("-a", "--age",  type = int, dest = "age",    help = "Your age",         default = 0,      choices=range(150))
    #可选属性,默认为male
    args.add_argument('-s',"--sex",   type = str, dest = "sex",    help = 'Your sex',         default = 'male', choices=['male', 'female'])
    #可选属性,默认为None,-p后可接多个参数
    #nargs参数的个数“*”代表任意多参数
    args.add_argument("-p","--parent",type = str, dest = 'parent', help = "Your parent",      default = "None", nargs = '*')
    #可选属性,默认为None,-o后可接多个参数
    #required是否为必选参数
    args.add_argument("-o","--other", type = str, dest = 'other',  help = "other Information",required = False,nargs = '*')
    # action=store_true/ store_false有参数时，值为true，没有为false
    args.add_argument("--no-cuda", action="store_true", default=False,help="enables CUDA training")
    args = args.parse_args()#返回一个命名空间,如果想要使用变量,可用args.attr
    print("argparse.args=",args,type(args))
    print('name = %s'%args.name)
    d = args.__dict__
    for key,value in d.items():
        print('%s = %s'%(key,value))

if __name__=="__main__":
    cmd()


```

以下是一个深度学习的例子：

深度学习种设定的参数：

1. batch_size
2. epochs
3. no_cuda
4. seed
5. log_interval

```python
parser = argparse.ArgumentParser(description="VAE MNIST Example")

parser.add_argument("--batch_size", type=int, default=128, metavar="N", 
                    help="input batch size for training (default is 128)")

parser.add_argument("--epochs", type=int, default=10, metavar="N",
                    help="number of epochs to train (default is 10)")

parser.add_argument("--no_cuda", action="store_true", default=False,
                    help="enables CUDA training")

parser.add_argument("--seed", type=int, default=1, metavar="S",
                    help="random seed (default is 1)")

parser.add_argument("--log_interval", type=int, default=10, metavar="N",
                    help="how many batches to wait before logging training status")
#argparse.Namespace 对象
args = parser.parse_args()

#可以通过args访问以上设置的参数
args.no_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
```

通过argparse，可以很方便的对程序进行调试。通过上面的也能注意到：在add_argument中，我们只需要设定 ``关键词,type, default, metavar, help`` 这五项即可！！！

