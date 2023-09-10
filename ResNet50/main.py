import sys
from tqdm import tqdm
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.cuda

#from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.tensorboard import SummaryWriter
from ResNet_ import *
from clr import *


def train_net(model, epoch, criterion, optimizer, clr):
    losses = []
    acces = []
    eval_losses = []
    eval_acces = []
    e = math.ceil(_clr.step_size * 6)
    # clr = _clr
    tb_writer = SummaryWriter()

    device = torch.device("cuda:3")
    criterion = criterion.to(device)
    net = model.to(device)

    for epochCounter in range(epoch):
        train_loss = 0
        train_acc = 0
        net.train()

        data_loder = tqdm(train_data, file=sys.stdout)
        for im, label in data_loder:
            im = im.to(device)
            label = label.to(device)
            # 前向传播
            out = net(im)
            loss = criterion(out, label)

            # 周期学习率
            if epochCounter < e:
                lr = clr.clr()
                optimizer.param_groups[0]['lr'] = lr
            else:
                if i < math.floor(_clr.step_size * 7):
                    lr = optimizer.param_groups[0]['lr'] / 10
                else:
                    lr = optimizer.param_groups[0]['lr'] / 100

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 记录误差
            train_loss += loss.item()
            # 计算分类的准确率
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / im.shape[0]
            train_acc += acc

        losses.append(train_loss / len(train_data))
        acces.append(train_acc / len(train_data))
        # 在测试集上检验效果
        eval_loss = 0
        eval_acc = 0
        net.eval()  # 将模型改为预测模式

        for im, label in test_data:
            im = im.to(device)
            label = label.to(device)
            out = net(im)
            loss = criterion(out, label)
            # 记录误差
            eval_loss += loss.item()
            # 记录准确率
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / im.shape[0]
            eval_acc += acc

        clr.on_batch_end(1)

        eval_losses.append(eval_loss / len(test_data))
        eval_acces.append(eval_acc / len(test_data))
        print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
              .format(epochCounter, train_loss / len(train_data), train_acc / len(train_data),
                      eval_loss / len(test_data), eval_acc / len(test_data)))
        print(', Current Lr: {}'.format(lr))

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], losses[epochCounter], epochCounter)
        tb_writer.add_scalar(tags[1], acces[epochCounter], epochCounter)
        tb_writer.add_scalar(tags[2], eval_losses[epochCounter], epochCounter)
        tb_writer.add_scalar(tags[3], eval_acces[epochCounter], epochCounter)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epochCounter)

    torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    data_path = './datasets'
    train_set = CIFAR10(data_path, train=True, transform=data_tf, download=False)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_set = CIFAR10(data_path, train=False, transform=data_tf, download=False)
    test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

    resnet50 = ResNet(block_num=[3, 4, 6, 3])
    model = resnet50
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    _clr = CyclicLR(base_lr=0.0001, max_lr=0.0003, step_size=100, mode='triangular2')

    epoch = 400
    e = math.ceil(_clr.step_size * 7)
    clr = _clr
    '''
    for i in range(epoch):
        if i < e:
            lr = clr.clr()
            clr.on_batch_end(1)
            print(lr)

        else:
            if i < math.floor(_clr.step_size * 7):
                lr = optimizer.param_groups[0]['lr'] / 10
                print(lr)
            else:
                lr = optimizer.param_groups[0]['lr'] / 100
                print(lr)
    '''

    train_net(model, epoch=epoch, criterion=criterion, optimizer=optimizer, clr=_clr)