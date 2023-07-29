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
        img = Image.open(self.images_path[item])
        # PIL对象
        label = self.images_label[item]
        img = self.transform(img)
        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)  # 堆叠，升维
        labels = torch.as_tensor(labels)

        return images, labels
