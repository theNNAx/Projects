import os
import random
import sys

import torch
import torch.nn
from tqdm import tqdm
from apex import amp
import torch.distributed as dist

def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    _class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    _class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(_class))

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型

    # 遍历每个文件夹下的文件
    for cla in _class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
#    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    return train_images_path, train_images_label, val_images_path, val_images_label


def train_one_epoch(model, optimizer, data_loder, epoch):

    model.train()
    loss_fun = torch.nn.CrossEntropyLoss()

    accu_loss = torch.zeros(1).cuda(non_blocking=True)
    accu_num = torch.zeros(1).cuda(non_blocking=True)
   # accu_loss = torch.zeros(1).to(device)
   # accu_num = torch.zeros(1).to(device)

    sample_num = 0
    data_loder = tqdm(data_loder, file=sys.stdout)

    for step, data in enumerate(data_loder):
        images, labels = data
        sample_num += images.shape[0]

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
       # images = images.to(device)
       # labels = labels.to(device)

        # (64, 3 ,224, 224)
        pred = model(images)
        # (64,4)
        pred_classes = torch.max(pred, dim=1)[1]
        '''
        torch.max() 函数的返回值包含两个部分：
        第一个部分是最大值张量，其中每个元素都是 pred 张量对应行中的最大值。
        第二个部分是最大值索引张量，其中每个元素都是 pred 张量对应行中最大值的索引（类别标签）。这就是我们关心的部分，即预测的类别标签。
        '''
        accu_num += torch.eq(pred_classes, labels).sum()
        loss = loss_fun(pred, labels)
#        loss.backward()  # 反向传播
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        
        accu_loss += loss

        data_loder.desc = "[train epoch {}, loss:{:.3f}, acc:{:.3f}]".format(epoch,
                                                                             accu_loss.item() / (step + 1),
                                                                             accu_num.item() / sample_num)

        optimizer.step()
        optimizer.zero_grad()

    train_loss = accu_loss.item() / (step + 1)
    train_acc = accu_num.item() / sample_num
    return train_loss, train_acc


@torch.no_grad()
def evaluate(model, data_loder, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).cuda(non_blocking=True)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).cuda(non_blocking=True)  # 累计损失

    sample_num = 0
    data_loder = tqdm(data_loder, file=sys.stdout)
    for step, data in enumerate(data_loder):
        images, labels = data
        sample_num += images.shape[0]
        
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        pred = model(images)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()

        loss = loss_function(pred, labels)
        accu_loss += loss

        data_loder.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

