# 实验室4卡训练Vision transformer + tensorboard可视化

# 调用apex库，混合精度加速
Apex 是 NVIDIA 开源的用于混合精度训练和分布式训练库。Apex 对混合精度训练的过程进行了封装，改两三行配置就可以进行混合精度的训练，从而大幅度降低显存占用，节约运算时间。此外，Apex 也提供了对分布式训练的封装，针对 NVIDIA 的 NCCL 通信库进行了优化。

在混合精度训练上，Apex 的封装十分优雅。直接使用 amp.initialize 包装模型和优化器，apex 就会自动帮助我们管理模型参数和优化器的精度了，根据精度需求不同可以传入其他配置参数。
> ```model, optimizer = amp.initialize(model, optimizer, opt_level='O1')```
> 其中 opt_level 为精度的优化设置，O0（第一个字母是大写字母O）
> O0：纯FP32训练，可以作为accuracy的baseline；
> O1：混合精度训练（推荐使用），根据黑白名单自动决定使用FP16（GEMM, 卷积）还是FP32（Softmax）进行计算。
> O2：“几乎FP16”混合精度训练，不存在黑白名单，除了Batch norm，几乎都是用FP16计算。
> O3：纯FP16训练，很不稳定，但是可以作为speed的baseline；
---------------------------------------
## 调用 torch.distributed.launch 启动器启动
```CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py```

## 启动tensorBoard服务器
1. ```cd runs```
2. ```bash tensorboard.sh```
