import torch
import torch.nn as nn


def patch_partition(x, window_size: int):
    pass


def patch_reverse(window, window_szie, H: int, W: int):
    pass


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_c, embed_dim, norm_layer):
        super().__init__()

    def forward(self, x):
        pass


class PatchMerging(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class WindowAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class SwinTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, ):
        super().__init__()

    def forward(self, x):
        pass


class SwinTransformer(nn.Module):
    def __init__(self, in_chans, patch_size, window_size, embed_dim, depths, num_heads, num_classes,
                 drop_rate, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_layers = len(depths)



        self.patch_embed = PatchEmbed(patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                      norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layers = BasicLayer()



            self.layers.append(layers)


    def forward(self, x):
        pass


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
