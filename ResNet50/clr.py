import math


class CyclicLR:
    # 实现循环学习率策略的回调函数类
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular'):
        super().__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.clr_iterations = 0.
        self.triangular_iterations = 0.

    def clr(self):
        cycle = math.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.mode == 'triangular':
            return self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))
        elif self.mode == 'triangular2':
            return self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) / float(2 ** (cycle - 1))
        else:
            return self.base_lr

    def on_train_begin(self, logs={}):
        logs = logs or {}
        self.clr_iterations = 0.

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.clr_iterations += 1
        self.triangular_iterations += 1
        lr = self.clr()
        logs['lr'] = lr