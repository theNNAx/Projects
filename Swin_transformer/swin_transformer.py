import torch
import torch.nn as nn


class SwinTransformer(nn.Module):

    def __init__(self, in_chans, patch_size, window_size, embed_dim, depths, num_heads, num_classes):
        super().__init__()



def swin_tiny_patch4_window7_224(num_classes: int = 1000):
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                            )
    return model
