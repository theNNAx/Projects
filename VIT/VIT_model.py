import torch
import torch.nn as nn
from functools import partial
import droppath


def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

   # elif isinstance(m, nn.LayerNorm):


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_c, embed_dim):
        super(PatchEmbed, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size  # 224,224
        self.patch_size = patch_size  # 16,16

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 14,14
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.pe = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (64, 3, 224, 224) -> (64, 768, 14, 14)
        # (64, 768, 14, 14) -> (64, 768, 196)
        # (64, 768, 196) -> (64, 196, 768)
        x = self.pe(x).flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, msa_drop_ratio, proj_drop_ratio):
        super(Attention, self).__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.msa_drop = nn.Dropout(msa_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # (64, 197, 768 )
        # (64, 197, 768 ) -> (64, 197, 768 * 3) -> (64, 197, 3, 8, 96) -> (3, 64, 8, 197, 96)
        qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        msa = (q @ k.transpose(-2, -1)) * self.scale
        # (64, 8, 197, 197) -> (64, 8, 197, 96) -> (64, 197, 768)
        msa = msa.softmax(dim=-1)
        msa = self.msa_drop(msa)

        x = (msa @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, dim):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.ac = nn.GELU()
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        # (64, 197 ,768)
        x = self.drop(self.fc2(self.drop(self.ac(self.fc1(x)))))
        return x


class Block(nn.Module):
    def __init__(self, dim, norm_layer, drop_path_ratio, embed_dim, num_heads, msa_drop_ratio, proj_drop_ratio):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)

        self.msa = Attention(dim=embed_dim, num_heads=num_heads, msa_drop_ratio=msa_drop_ratio,
                             proj_drop_ratio=proj_drop_ratio)

        self.drop_path = droppath.DropPath(drop_path_ratio)
        self.norm2 = norm_layer(dim)

        self.mlp = Mlp(dim=embed_dim)

    def forward(self, x):
        # (64, 197, 168)
        x = x + self.drop_path(self.msa(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, depth, num_heads, num_classes, in_c=3, embed_dim=768,
                 embed_layer=PatchEmbed, drop_path_ratio=0.2, msa_drop_ratio=0.2, proj_drop_ratio=0.2):
        super(VisionTransformer, self).__init__()
        self.depth = depth
        # self.num_heads = num_heads
        self.num_features = embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.ones(1, num_patches + 1, embed_dim))
        self.drop = nn.Dropout(p=drop_path_ratio)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, norm_layer=norm_layer, drop_path_ratio=drop_path_ratio, embed_dim=embed_dim,
                  num_heads=num_heads,
                  msa_drop_ratio=msa_drop_ratio, proj_drop_ratio=proj_drop_ratio)
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(self.num_features, num_classes)
        self.apply(_init_vit_weights)

    def forward(self, x):
        x = self.patch_embed(x)  # (64, 196, 768)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # (64, 197, 768)
        x = x + self.pos_embed
        x = self.drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 0]  # (64, 1, 768)
        x = self.head(x)
        return x

    # img_size = 224
    # img (3,224,224)


def vit_base_patch16_224(num_classes: int = 1000):
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              depth=8,
                              num_heads=8,
                              num_classes=525)
    return model
