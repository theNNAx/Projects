### 分布式启动
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py

### 启动tensorBoard服务器
1. cd runs
2. bash tensorboard.sh
