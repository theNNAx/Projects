import torch
import torch.nn as nn
from droppath import DropPath
from typing import Optional


def window_partition(x, window_size: int):
    pass


def window_reverse(window, window_szie, H: int, W: int):
    pass


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_c, embed_dim, norm_layer):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_c = in_c
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        pass


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        pass


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.ac1 = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        pass


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5




    def forward(self, x, mask: Optional[torch.Tensor] = None):
        pass


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size, qkv_bias,
                 mlp_ratio, drop_path, drop, attn_drop, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shit_size = shift_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim=dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)

        self.mlp_ratio = mlp_ratio
        mlp_hidden_dim = int(mlp_ratio * dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        pass


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, drop_path, qkv_bias, drop, attn_drop,
                 mlp_ratio, downsample=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = window_size // 2

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim,
                                 num_heads=num_heads,
                                 window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else self.shift_size,
                                 drop_path=drop_path,
                                 drop=drop,
                                 attn_drop=attn_drop,
                                 qkv_bias=qkv_bias,
                                 mlp_ratio=mlp_ratio,
                                 norm_layer=norm_layer
                                 )
            for i in range(depth)
        ])

        #  PatchMerging
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        pass


    def forward(self, x):
        pass


class SwinTransformer(nn.Module):
    def __init__(self, in_chans=3, patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24), num_classes=1000, mlp_ratio=4., qkv_bias=True, attn_drop_rate=0.,
                 drop_rate=0., drop_path_rate=0.2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_c=in_chans,
                                      embed_dim=embed_dim, norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, len(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layers = BasicLayer(dim=embed_dim,
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                attn_drop=attn_drop_rate,
                                drop=drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None
                                )

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
