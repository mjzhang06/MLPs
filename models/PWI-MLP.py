# 2024.05.07-Main script for HWcat_MLPcct model for tongue consititution image 

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
from pytorch_wavelets import DWTForward


# def _cfg(url='', **kwargs):
#     return {
#         'url': url,
#         'num_classes': 9, 'input_size': (3, 224, 224), 'pool_size': None,
#         'crop_pct': .96, 'interpolation': 'bicubic',
#         'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
#     }

# default_cfgs = {
#     'HWCAT_T': _cfg(crop_pct=0.9),
#     'HWCAT_S': _cfg(crop_pct=0.9),
#     'HWCAT_M': _cfg(crop_pct=0.9),
#     'HWCAT_B': _cfg(crop_pct=0.875),
# }


# ============== deep-wide con module 0721 =============
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x
# ============== deep-wide con module ==================


class Mlp(nn.Module):   # 一维卷积
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.dwconv = DWConv(hidden_features)   # deep-wide conv
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # 增加特征， 20240514
        x = self.fc1(x)
        x = x + self.dwconv(x)

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class HWCAT(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.h_conv = nn.Sequential(
                nn.Conv2d(dim, dim, 1, 1, bias=True),
                nn.BatchNorm2d(dim),
                nn.ReLU()
                )
        self.w_conv = nn.Sequential(
                nn.Conv2d(dim, dim, 1, 1, bias=True),
                nn.BatchNorm2d(dim),
                nn.ReLU()
                )
        self.fc_c = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
         
        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Conv2d(dim, dim, 1, 1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.fc_h = nn.Conv2d(2*dim, dim, (1,7), stride=1, padding=(0,7//2), groups=dim, bias=False)
        self.fc_w = nn.Conv2d(2*dim, dim, (7,1), stride=1, padding=(7//2,0), groups=dim, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape    # B为batch size, C为卷积后输出特征的维度dim, H为输出特征的高度，W为输出特征的宽度
        
        h = self.h_conv(x)      # Conv 1×1,BN,ReLU
        w = self.w_conv(x)      # Conv 1×1,BN,ReLU
        c = self.fc_c(x)

        # 修正，20240514， 不加权重
        hh = torch.cat([h*W, h], dim=1)
        ww = torch.cat([w*H, w], dim=1)

        h = self.fc_h(hh)
        w = self.fc_w(ww)
        # print(hh.shape)

        a = F.adaptive_avg_pool2d(h + w + c, output_size=1)    # 平均池化输出
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)

        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)        # 混合sum
        x = self.proj_drop(x)    

        return x
        

# ==== 交互特征模块
class PermuteMLPattention(nn.Module):
    def __init__(self, dim, segment_dim=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.segment_dim = segment_dim

        # self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        # self.mlp_h = nn.Linear(dim, dim, bias=qkv_bias)
        # self.mlp_w = nn.Linear(dim, dim, bias=qkv_bias)

        # 修改为1×1卷积
        self.mlp_c = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.mlp_h = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.mlp_w = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)

        self.reweight = Mlp(dim, dim // 4, dim *3)
        
        # self.proj = nn.Linear(dim, dim)
        self.proj = nn.Conv2d(dim, dim, 1, 1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # B, H, W, C = x.shape
        B, C, H, W = x.shape    # B为batch size, C为卷积后输出特征的维度dim, H为输出特征的高度，W为输出特征的宽度
        # print(x.shape)  # ([64, 256, 8, 8])/ ([64, 128, 28, 28])/ ([64, 64, 56, 56])

        S = C // self.segment_dim # 32 / 8
        # print(S)

        # ==== dim=64 ====
        h = x.reshape(B, H, W, self.segment_dim, S).permute(0, 3, 2, 1, 4).reshape(B, H*S // 7, self.segment_dim * 7, W)
        # [64,56,56,32,2]==B,H,W,C,S;; b,c,h,w,s;; b,c,8×32,w
        h = self.mlp_h(h).reshape(B, self.segment_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, C, H, W)  #这里改
        # print('h:',h.shape)
        w = x.reshape(B, H, W, self.segment_dim, S).permute(0, 1, 3, 2, 4).reshape(B, W*S // 7, H, self.segment_dim * 7)
        w = self.mlp_w(w).reshape(B, H, self.segment_dim, W, S).permute(0, 1, 3, 2, 4).reshape(B, C, H, W)  #这里改

        # ==== dim=256 ====
        # h = x.reshape(B, H, W, self.segment_dim, S).permute(0, 3, 2, 1, 4).reshape(B, H*S*4, self.segment_dim // 4, W)
        # # [64,8,8,32,8]==B,H,W,C,S;; b,c,h,w,s;; b,c,w,8×32
        # # [64,28,28,32,4]===
        # h = self.mlp_h(h).reshape(B, self.segment_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, C, H, W)  #这里改
        # print('h:',h.shape)
        # w = x.reshape(B, H, W, self.segment_dim, S).permute(0, 1, 3, 2, 4).reshape(B, W*S*4, H, self.segment_dim // 4)
        # w = self.mlp_w(w).reshape(B, H, self.segment_dim, W, S).permute(0, 1, 3, 2, 4).reshape(B, C, H, W)  #这里改

        c = self.mlp_c(x)
        
        # a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        # a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)
        a = F.adaptive_avg_pool2d(h + w + c, output_size=1)    # 平均池化输出
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)

        x = h * a[0] + w * a[1] + c * a[2]
        # print('置换x:',x.shape)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class PermutatorBlock(nn.Module):

    def __init__(self, dim, segment_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, skip_lam=1.0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PermuteMLPattention(dim, segment_dim=segment_dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x


class HWBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = HWCAT(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        B, C, H, W = x.shape    # B, C, H, W,0,1,2,3->0,2,3,1
        x = self.norm1(x)
        short = x
        x = short + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvTokenizer(nn.Module):
    def __init__(self, embedding_dim=64):
        super(ConvTokenizer, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, embedding_dim // 2, kernel_size=(3, 3), # 输入通道数，输出维度，kernel size
                      stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim // 2, embedding_dim // 2, kernel_size=(3, 3), 
                      stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim // 2, embedding_dim, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                         dilation=(1, 1))
        )

    def forward(self, x):
        return self.block(x)

class PatchEmbedOverlapping(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=768, 
                norm_layer=nn.BatchNorm2d, groups=1, use_norm=True):
        super().__init__()
        patch_size = (patch_size, patch_size)
        stride = (stride, stride)
        padding= (padding, padding)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding, groups=groups)      
        self.norm = norm_layer(embed_dim) if use_norm==True else nn.Identity()

    def forward(self, x):
        x = self.proj(x)    # B, C, H, W
        x = self.norm(x)

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x) # B, C, H, W
        return x


class Downsample(nn.Module):
    def __init__(self, in_embed_dim, out_embed_dim, patch_size, norm_layer=nn.BatchNorm2d,use_norm=True):
        super().__init__()

        assert patch_size == 2, patch_size
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.norm = norm_layer(out_embed_dim) if use_norm==True else nn.Identity()

    def forward(self, x):
        x = self.proj(x)    # B, C, H, W
        x = self.norm(x)

        return x

# DWT下采样
class Downsample_DWT(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Downsample_DWT, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(out_ch),   
                                    nn.ReLU(inplace=True),                                 
                                    ) 
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)        
        x = self.conv_bn_relu(x)

        return x

def basic_blocks(dim, index, layers, segment_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, attn_drop=0.,
                 drop_path_rate=0.,norm_layer=nn.BatchNorm2d, **kwargs):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        if index == 0:
            blocks.append(PermutatorBlock(dim, segment_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,\
            attn_drop=attn_drop, drop_path=block_dpr))
        else:
            blocks.append(HWBlock(dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      attn_drop=attn_drop, drop_path=block_dpr, norm_layer=norm_layer))
    blocks = nn.Sequential(*blocks)
    return blocks

class HWCATNet(nn.Module):  # segment_dim=32, 8
    def __init__(self, layers, img_size=224, patch_size=4, in_chans=3, num_classes=10,
        embed_dims=None, transitions=None, segment_dim=8, mlp_ratios=4.,
        qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_layer=nn.BatchNorm2d, fork_feat=False,ds_use_norm=True,args=None): 
        super().__init__()
        
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        # self.patch_embed = PatchEmbedOverlapping(patch_size=7, stride=4, padding=2, in_chans=3, embed_dim=embed_dims[0],norm_layer=norm_layer,use_norm=ds_use_norm)
        self.tokenizer = ConvTokenizer(embedding_dim=embed_dims[0])
        # self.patch_embed = PatchEmbed(img_size=img_size, patch_size=7, in_c=in_chans, embed_dim=embed_dims[0],norm_layer=norm_layer)
        # self.patch_embed = PatchEmbed(img_size = img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims[0])

        # 增加位置编码，20260305
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dims[0], 224//4, 224//4))

        network = []
        for i in range(len(layers)):    # 原mlp_ratio=mlp_ratios[i]
            stage = basic_blocks(embed_dims[i], i, layers, segment_dim, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
                                 qk_scale=qk_scale, attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate,
                                 norm_layer=norm_layer)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i+1]:
                patch_size = 2 if transitions[i] else 1
                # 卷积下采样
                # network.append(Downsample(embed_dims[i], embed_dims[i+1], patch_size, norm_layer=norm_layer, use_norm=ds_use_norm))
                # 小波下采样
                network.append(Downsample_DWT(embed_dims[i], embed_dims[i+1]))

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            self.norm = norm_layer(embed_dims[-1]) 
            self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        # x = self.patch_embed(x)
        x = self.tokenizer(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            return outs
        return x

    def forward(self, x):
        # x = self.forward_embeddings(x)
        x = self.forward_embeddings(x) + self.pos_embed
        B, C, H, W = x.shape
        # print(x.shape)
        x = self.forward_tokens(x)
        if self.fork_feat:
            return x
        # print(x.shape)
        x = self.norm(x)
        cls_out = self.head(F.adaptive_avg_pool2d(x,output_size=1).flatten(1))
        return cls_out

def MyNorm(dim):
    return nn.GroupNorm(1, dim)    


@register_model
def PWIMLP_Ttoken(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [2, 2, 4, 2]
    # mlp_ratios = [4, 4, 4, 4]
    mlp_ratios = 4
    embed_dims = [64, 128, 320, 512]
    # embed_dims = [64, 128, 256, 512]
    model = HWCATNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios, **kwargs)
    model.default_cfg = _cfg
    return model

@register_model
def PWIMLP_Stoken(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [2, 3, 10, 3]
    mlp_ratios = [4, 4, 4, 4]
    # embed_dims = [64, 128, 320, 512]
    embed_dims = [64, 128, 256, 512]
    model = HWCATNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios,norm_layer=MyNorm, **kwargs)
    model.default_cfg = _cfg
    return model

@register_model
def PWIMLP_Mtoken(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [3, 4, 18, 3]
    mlp_ratios = [8, 8, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = HWCATNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios,norm_layer=MyNorm,ds_use_norm=False, **kwargs)
    model.default_cfg = _cfg
    return model

@register_model
def PWIMLP_Btoken(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [2, 2, 18, 2]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [96, 192, 384, 768]
    model = HWCATNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios,norm_layer=MyNorm,ds_use_norm=False, **kwargs)
    model.default_cfg = _cfg
    return model

