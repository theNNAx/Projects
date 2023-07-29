import torch
from torch.utils.data import Dataset
from PIL import Image

from torchvision.transforms.functional import to_tensor

class MyDataSet(Dataset):
    def __init__(self, images_path, images_label, transform=None):
        self.images_path = images_path
        self.images_label = images_label
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        '''
        通常情况下，我们在训练模型时会使用数据加载器（data loader）来批量地获取样本数据。
        数据加载器会自动调用数据集的__getitem__()方法，并将样本组织成小批量进行训练。
        这样可以更高效地处理大量数据并利用多线程加载数据，从而加快训练过程。
        '''
        img = Image.open(self.images_path[item])
        # PIL对象
        label = self.images_label[item]
        img = self.transform(img)
        return img, label

    @staticmethod
    def collate_fn(batch):
        #  对一个批次数据进行自定义处理的函数
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)  # 堆叠，升维
        labels = torch.as_tensor(labels)  # 通过 torch.as_tensor(labels) 操作，它将会将 labels 转换为一个 PyTorch 张量

        return images, labels
