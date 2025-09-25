"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial#从标准库 functools 里导入 partial，用来冻结函数的一部分参数，返回一个新函数
from collections import OrderedDict#典型用途：给 nn.Sequential 显式命名各层，保持顺序与可读性

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from utils.Token_performer import Token_performer#把代码里会反复用到的通用模块（比如注意力实现、卷积模块、数据处理函数）集中放进 utils/,这通常是项目作者自己建的一个目录（文件夹），专门用来放一些“utility（工具函数/工具模块）”
from utils.PRM import PRM
# cls和pos位置编码还没加入


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=33, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        self.img_size = img_size
        self.patch_size = (patch_size, patch_size)
        self.num_patches = 255

        # self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)# 要修改为一维卷积
        # 使用1D卷积，将1024个通道降至32个，卷积核大小为1
        # self.proj = nn.Conv1d(in_channels=in_c, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Conv1d(in_channels=1, out_channels=768, kernel_size=patch_size, stride=4, padding=patch_size//2)
        self.conv1d = nn.Conv1d(257, 255, kernel_size=3, padding=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):


        B, C, H = x.shape # x [64, 1, 1024]
        # assert H == self.img_size[0] , \
        #     f"Input image size ({H}*{'1'}) doesn't match model ({self.img_size[0]}*{{'1'}})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        # x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.proj(x).transpose(1, 2)
        x = self.conv1d(x)
        x = self.norm(x)    # [32, 255, 768]
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 修改这里，使用1维的自适应平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):                   # [Batch, C, W] [Batch, 256, 256]
        b, c, _ = x.size()          
        y = self.avg_pool(x).view(b, c)     # Batch C
        y = self.fc(y).view(b, c, 1) 
        result =  x * y.expand_as(x) 
        # # 打印结果
        # print("x 第一个 batch 的第一行:")
        # print(x[0, 0, :])

        # print("\ny 第一个 batch 的第一行:")
        # print(y[0, 0, :])

        # print("\nresult 第一个 batch 的第一行:")
        # print(result[0, 0, :])
        return result


class BlockRC1(nn.Module):
    def __init__(self, img_size=1024, in_chans=1, embed_dims=64, token_dims=64, downsample_ratios=4, kernel_size=7,
                 num_heads=1, dilations=[1,2,3,4],group=1,share_weights=False, op='cat'):
        super(BlockRC1, self).__init__()
        self.dilations = dilations
        self.downsample_ratios = downsample_ratios
        self.PRM = PRM(img_size=img_size, kernel_size=kernel_size, downsample_ratio=downsample_ratios, dilations=self.dilations,
                in_chans=in_chans, embed_dim=embed_dims, share_weights=share_weights, op=op)
        in_chans = self.PRM.out_chans                                   # 256 = embed_dims * len(self.dilations) = 64*4
        self.se_layer = SELayer(channel=in_chans)
        self.attn = Token_performer(dim=in_chans, in_dim=token_dims, head_cnt=num_heads, kernel_ratio=0.5)


    def forward(self, x):
                                                                        # x [32, 1, 1024]
        PRM_x = self.PRM(x)                                             # [Batch, 256, 256]                         
                                                                        # [Batch, (1024/4), embed_dims=64 * 4]                                                                                    
        # PRM_x = self.se_layer(PRM_x)
        x = self.attn.attn(self.attn.norm1(PRM_x))                      # [Batch, 256, 64]                                                                            
        x = x + self.attn.drop_path(self.attn.mlp(self.attn.norm2(x)))  # [Batch, 256, 64]                                           

        return x     
    

class BlockRC2(nn.Module):
    def __init__(self, img_size=1024, in_chans=64, embed_dims=64, token_dims=64, downsample_ratios=2, kernel_size=3,
                 num_heads=1, dilations=[1,2,3],group=1,share_weights=False, op='cat'):
        super(BlockRC2, self).__init__()
        self.dilations = dilations
        self.downsample_ratios = downsample_ratios
        self.PRM = PRM(img_size=img_size, kernel_size=kernel_size, downsample_ratio=downsample_ratios, dilations=self.dilations,
                in_chans=in_chans, embed_dim=embed_dims, share_weights=share_weights, op=op)
        in_chans = self.PRM.out_chans                                       # 192 = embed_dims * len(self.dilations) = 64*3
        self.attn = Token_performer(dim=in_chans, in_dim=token_dims, head_cnt=num_heads, kernel_ratio=0.5)


    def forward(self, x):
                                                                            # [Batch, 256, 64]  
        x = x.permute(0, 2, 1)                                              # [Batch, 64, 256]  
        PRM_x = self.PRM(x)                                                 # [Batch, 128, 192]                         
                                                                            # [Batch, (1024/8), embed_dims=64 * 3]                             
        x = self.attn.attn(self.attn.norm1(PRM_x))                          # [Batch, 128, 64]                                                                            
        x = x + self.attn.drop_path(self.attn.mlp(self.attn.norm2(x)))      # [Batch, 128, 64]                                           

        return x    
    

class BlockRC3(nn.Module):
    def __init__(self, img_size=1024, in_chans=64, embed_dims=160, token_dims=320, downsample_ratios=2, kernel_size=3,
                 num_heads=1, dilations=[1,2],share_weights=False, op='cat'):
        super(BlockRC3, self).__init__()
        self.dilations = dilations
        self.downsample_ratios = downsample_ratios
        self.PCM = None
        self.PRM = PRM(img_size=img_size, kernel_size=kernel_size, downsample_ratio=downsample_ratios, dilations=self.dilations,
                in_chans=in_chans, embed_dim=embed_dims, share_weights=share_weights, op=op)
        in_chans = self.PRM.out_chans                                   # 320 = embed_dims * len(self.dilations) = 160*2 
        self.attn = Token_performer(dim=in_chans, in_dim=token_dims, head_cnt=num_heads, kernel_ratio=0.5)



    def forward(self, x):
                                                                        # [Batch, 128, 64]   
        x = x.permute(0, 2, 1)                                          # [Batch, 64, 128]         
        PRM_x = self.PRM(x)                                             # [Batch, 64, 320]                         
                                                                        # [Batch, (1024/16), embed_dims=160 * 2]                        

        # 为什么能把PRM_x的C维度降低到64
        x = self.attn.attn(self.attn.norm1(PRM_x))                      # [Batch, 64, 320]                                                                            
        x = x + self.attn.drop_path(self.attn.mlp(self.attn.norm2(x)))  # [Batch, 64, 320]                                           
        return x   
    

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.dim = dim 
        self.norm1 = norm_layer(self.dim)
        self.attn = Attention(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(self.dim)
        mlp_hidden_dim = int(self.dim * mlp_ratio)
        self.mlp = Mlp(in_features=self.dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
        # self.PCM = nn.Sequential(
        #             nn.Conv2d(self.dim, mlp_hidden_dim, 3, 1, 1, 1, 64),
        #             nn.BatchNorm2d(mlp_hidden_dim),
        #             nn.SiLU(inplace=True),
        #             nn.Conv2d(mlp_hidden_dim, self.dim, 3, 1, 1, 1, 64),
        #             nn.BatchNorm2d(self.dim),
        #             nn.SiLU(inplace=True),
        #             nn.Conv2d(self.dim, self.dim, 3, 1, 1, 1, 64),
        #             nn.SiLU(inplace=True),
        #             )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module): # attention in_c=1
    def __init__(self, img_size=224, patch_size=16, in_c=1, num_classes=1000,
                 embed_dim=320, depth=12, num_heads=12, mlp_ratio=2, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*
        [BlockRC1()]+[BlockRC2()]+[BlockRC3()]
        +
        [
            Block(dim=embed_dim, num_heads=4, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(1,11)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=320,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model