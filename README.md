# 实验室4卡训练Vision transformer(apex库加速) + tensorboard可视化
> 设备：4张3090， 128G显存， 1T内存
## 调用apex库，混合精度加速
```python
from apex import amp
from apex.parallel import DistributedDataParallel
```
Apex 是 NVIDIA 开源的用于混合精度训练和分布式训练库。Apex 对混合精度训练的过程进行了封装，大幅度降低显存占用，节约运算时间。此外，Apex 也提供了对分布式训练的封装，针对 NVIDIA 的 NCCL 通信库进行了优化。
```python
import torch.distributed as dist
dist.init_process_group(backend='nccl', init_method='env://')
```

现有的标准 Batch Normalization 因为使用数据并行（Data Parallel），是单卡的实现模式，只对单个卡上对样本进行归一化，相当于减小了批量大小（batch-size）（详见BN工作原理部分）。对于比较消耗显存的训练任务时，往往单卡上的相对批量过小，影响模型的收敛效果。之前在我们在图像语义分割的实验中，Jerry和我就发现使用大模型的效果反而变差，实际上就是BN在作怪。跨卡同步 Batch Normalization 可以使用全局的样本进行归一化，这样相当于‘增大‘了批量大小，这样训练效果不再受到使用 GPU 数量的影响。最近在图像分割、物体检测的论文中，使用跨卡BN也会显著地提高实验效果，
```python
from apex.parallel import convert_syncbn_model
model = convert_syncbn_model(model).cuda()
```

### 汇总
```python


    model = vit_model(num_classes=args.num_classes)
    model = convert_syncbn_model(model).cuda()

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-4)

    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    model = DistributedDataParallel(model)
```



在混合精度训练上，Apex 的封装十分优雅。直接使用 amp.initialize 包装模型和优化器，apex 就会自动帮助我们管理模型参数和优化器的精度了，根据精度需求不同可以传入其他配置参数。
>
```python
model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
'''
opt_level 为精度的优化设置，O0（第一个字母是大写字母O）
O0：纯FP32训练，可以作为accuracy的baseline
O1：混合精度训练（推荐使用），根据黑白名单自动决定使用FP16（GEMM, 卷积）还是FP32（Softmax）进行计算
O2：“几乎FP16”混合精度训练，不存在黑白名单，除了Batch norm，几乎都是用FP16计算
O3：纯FP16训练，很不稳定，但是可以作为speed的baseline
'''
```
## 反向传播代码更新
```python
# 反向传播时需要调用 amp.scale_loss，用于根据loss值自动对精度进行缩放
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
```



## 调用 torch.distributed.launch 启动器启动
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py
```

## 启动tensorBoard服务器
1. ```cd runs```
2. ```bash tensorboard.sh```
