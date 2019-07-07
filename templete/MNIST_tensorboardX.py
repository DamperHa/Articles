# 导入类
from __future__ import print_function
from argparse import ArgumentParser
import torch 
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F 
from torch.optim import SGD 
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

# 判断是否有此安装包
try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found")

from ignite.engine import Events, create_supervised_evaluator,create_supervised_trainer
from ignite.metrics import Accuracy, Loss

# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()  # 这个不熟
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # 这里应该全在self，用nn定义
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)  # [64, 320] [batch, 320]
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)  # 需要记录

# 获取Dataloader
def get_data_loaders(train_batch_size, val_batch_size):
    data_transform = Compose([ToTensor(), Normalize([0.1307], [0.3081])])

    train_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=True),
                                batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=False),
                                batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader

# 建立Summarywriter
def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(logdir= log_dir)
    # 这一句不是很明白
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph:{}".format(e))
    return writer 

# 定义run函数
def run(train_batch_size, val_batch_size, epochs, lr, momentum, log_interval, log_dir):
    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)  # Dataloader实例化
    model = Net()           # 网络模型
    writer = create_summary_writer(model, train_loader, log_dir)
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"
    
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)  # 优化器
    # 定义trainer，传入，model、optimizer、loss、device实例化
    trainer = create_supervised_trainer(model, optimizer, F.nll_loss, device=device)
    evaluator = create_supervised_evaluator(model, metrics={"accuracy": Accuracy(),
                                                            "nll":Loss(F.nll_loss)},
                                                            device=device)

    
    # 定义触发事件
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_train_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % log_interval == 0:
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
            .format(engine.state.epoch, iter, len(train_loader), engine.state.output))  # 这里的engine.state.output 不是很明白
            writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)  # iteration是总共的迭代次数

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics   # 
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        print("Training Results -Epoch:{} Avg accuracy: {:.2f} Avg loss: {:.2f}"
                .format(engine.state.epoch, avg_accuracy, avg_nll))
        
        writer.add_scalar("training/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_accuracy, avg_nll))
        writer.add_scalar("valdation/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("valdation/avg_accuracy", avg_accuracy, engine.state.epoch)

    trainer.run(train_loader, max_epochs=epochs)
    writer.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val_batch_size', type=int, default=1000,
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log_dir", type=str, default="tensorboard_logs",
                        help="log directory for Tensorboard log output")

    args = parser.parse_args()

    #run(args.batch_size, args.val_batch_size, args.epochs, args.lr, args.momentum,
        #args.log_interval, args.log_dir)
    a = Net()
    for i in a.named_parameters():
        print(i[0])
    print(a.named_parameters())


