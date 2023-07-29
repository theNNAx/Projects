import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as dist

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import argparse
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel
from apex.parallel import convert_syncbn_model


from utils import read_split_data, train_one_epoch, evaluate
from dataset import MyDataSet
from VIT_model import vit_base_patch16_224 as vit_model


def main(args):
#    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    tb_writer = SummaryWriter()
    torch.backends.cudnn.benchmark = True

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)

    #  数据集分割-> 训练集 + 验证集
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    #  数据预处理管道
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
    #  实例化数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_label=train_images_label,
                              transform=data_transform["train"]
                              )
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_label=val_images_label,
                            transform=data_transform["val"]
                            )

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    #  数据加载 -> 迭代器
    train_loder = DataLoader(train_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             pin_memory=True,
                             collate_fn=train_dataset.collate_fn,
                             sampler=train_sampler
                             )
   

    val_loder = DataLoader(val_dataset,
                           batch_size=args.batch_size,
                           shuffle=False,
                           pin_memory=True,
                           collate_fn=val_dataset.collate_fn,
                           sampler=val_sampler
                           )





#    device = torch.device(args.device)
#    model = vit_model(num_classes=args.num_classes).to(device)
    model = vit_model(num_classes=args.num_classes)
#    model = torch.nn.DataParallel(model).cuda()
    
    init_img = torch.zeros((1, 3, 224, 224), device = 'cuda:0')
    tb_writer.add_graph(model, init_img)

    device = torch.device(f'cuda:{args.local_rank}')    
    
    
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-4)

    model = convert_syncbn_model(model).cuda()
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')



#    weights_dict = torch.load(args.weights, map_location=device)
#    model.load_state_dict(weights_dict)


    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model = DistributedDataParallel(model)
#    model = model.cuda()

    # lf = lambda x:((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=args.lr, max_lr=args.lr * 3,
                                      step_size_up=100, step_size_down=100, mode='triangular2')
    '''
    在深度学习中，动量（momentum）的值一般在0.8到0.99之间选择。通常较常见的取值为0.9，因为在实践中它通常表现得相当好。
    动量是优化算法中的一个重要超参数，它引入了一种惯性效应，帮助优化算法在参数更新时更好地跨越平坦的区域，加速收敛，
    并减少在优化过程中的震荡。通过使用动量，算法可以在多个迭代之间积累之前的梯度信息，从而在优化方向上产生更加平滑和一致的更新。
    尽管动量值通常设置为0.9，但在某些特定问题上，不同的值可能会产生更好的结果。
    因此，在实际应用中，可以尝试不同的动量值，然后根据模型在验证集上的性能来选择最佳的值。
    一般来说，如果训练过程中模型的收敛速度较慢或出现震荡，可以尝试增加动量值，例如使用0.95或更高。
    反之，如果收敛速度过快或过度震荡，可以尝试减小动量值，例如使用0.8或更低。
    
    weight_decay（权重衰减）是深度学习优化算法中的一个重要超参数，用于控制模型参数的正则化。
    通常，weight_decay的值在0.0001到0.001之间是常见的选择。然而，具体的取值应该根据特定的问题和数据集来进行调整，以获得更好的模型性能。
    权重衰减通过在优化过程中向损失函数添加正则化项，对模型参数进行约束，以防止过拟合。它对于许多深度学习任务都是一种有效的正则化方法。
    如果weight_decay设置得太小，正则化的效果可能不够明显，导致模型过拟合。相反，如果weight_decay设置得太大，可能会对模型的训练产生过度的影响，
    导致模型性能下降。因此，通常建议从较小的值开始，然后逐步增大，直到找到一个在验证集上表现最佳的值。
    '''

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loder=train_loder,
                                                epoch=epoch
                                                #device=device
                                                )
        scheduler.step()
        val_loss, val_acc = evaluate(model=model, data_loder=val_loder, epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        if epoch > 770:
            if args.local_rank == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, './weights/checkpoint_{}.pth'.format(epoch))
#            torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--num-classes', type=int, default=525)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--gpu', type=str, default='0,1,2,3')
#    parser.add_argument('--device', default='cuda:0')
    
    parser.add_argument('--weights', type=str, default="/home/np/Desktop/projects/VIT_self/weights/msi.pth")
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--data-path', type=str, default="/home/np/Desktop/projects/VIT_self/dataset/bird525/train")

    opt = parser.parse_args()
    main(opt)
