# -*- encoding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from timm.models.layers import to_2tuple, trunc_normal_
from einops import rearrange
import numbers

'''
--------------------------------------------
Wangquan He (github: https://github.com/Hewq77/LGCT)
20/November/2024
--------------------------------------------
# Reference
https://ieeexplore.ieee.org/abstract/document/10742406

# If you use this code, please consider the following citation:
@article{he2024lgct,
  title={LGCT: Local-Global Collaborative Transformer for Fusion of Hyperspectral and Multispectral Images},
  author={He, Wangquan and Fu, Xiyou and Li, Nanying and Ren, Qi and Jia, Sen},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE},
  volume={62},
  pages={1-14}
}
--------------------------------------------
'''

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
   
class Upsample(nn.Module):
    def __init__(self, n_feat, upscale_level=2):
        super(Upsample, self).__init__()
        
        if upscale_level == 1  :
            self.body = nn.Sequential(
                nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2)
            )
        elif upscale_level == 2:
            self.body = nn.Sequential(
                nn.Conv2d(n_feat, n_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2)
            )
        elif upscale_level == 3:
            self.body = nn.Sequential(
                nn.Conv2d(n_feat, n_feat * 16, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(4)
            )
        else:
            raise ValueError("Unsupported upscale_factor.")

    def forward(self, x):
        return self.body(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='BiasFree'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
## Spatial Regrouping Self-attention (Spe_RSA)
class Spa_RSA(nn.Module):
    def __init__(
            self,
            dim,
            window_size=8,
            dim_head=48,
            heads=8,
            img_size=128,
            only_local_branch=False,
            bias=False
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = to_2tuple(window_size)
        self.only_local_branch = only_local_branch
        self.img_size = img_size
        
        # position embedding
        if only_local_branch:
            seq_len = window_size[0] * window_size[1]
            self.pos_emb = nn.Parameter(torch.Tensor(1, heads, seq_len, seq_len))
            trunc_normal_(self.pos_emb)
        else:
            seq_len1 = self.window_size[0] * self.window_size[1]
            seq_len2 = (img_size // self.window_size[0]) ** 2
            self.pos_emb1 = nn.Parameter(torch.Tensor(1, 1, heads // 2, seq_len1, seq_len1))
            self.pos_emb2 = nn.Parameter(torch.Tensor(1, 1, heads // 2, seq_len2, seq_len2))
            trunc_normal_(self.pos_emb1)
            trunc_normal_(self.pos_emb2)

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.to_out = nn.Linear(inner_dim, dim)

    def local_branch(self, x, h, w):
        """Local branch computation."""
        w_size = self.window_size

        x_ = rearrange(x, 'b (h b0) (w b1) c -> (b h w) (b0 b1) c', b0=w_size[0], b1=w_size[1])
        q = self.to_q(x_)
        k, v = self.to_kv(x_).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        q *= self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.pos_emb
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = rearrange(out, '(b h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],
                        b0=w_size[0])
        return out

    def dual_branch(self, x, h, w):
        """Dual branch computation (local and global)."""
        w_size = self.window_size
        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        v1, v2 = v.chunk(2, dim=-1)

        # Local branch
        q1, k1, v1 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                             b0=w_size[0], b1=w_size[1]), (q1, k1, v1))
        q1, k1, v1 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads // 2), (q1, k1, v1))
        q1 *= self.scale
        sim1 = einsum('b n h i d, b n h j d -> b n h i j', q1, k1)
        sim1 = sim1 + self.pos_emb1
        attn1 = sim1.softmax(dim=-1)
        out1 = einsum('b n h i j, b n h j d -> b n h i d', attn1, v1)
        out1 = rearrange(out1, 'b n h mm d -> b n mm (h d)')

        # global branch
        q2, k2, v2 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                             b0=w_size[0], b1=w_size[1]), (q2, k2, v2))
        q2, k2, v2 = map(lambda t: t.permute(0, 2, 1, 3), (q2.clone(), k2.clone(), v2.clone()))
        q2, k2, v2 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads // 2), (q2, k2, v2))
        q2 *= self.scale
        sim2 = einsum('b n h i d, b n h j d -> b n h i j', q2, k2)
        sim2 = sim2 + self.pos_emb2
        attn2 = sim2.softmax(dim=-1)
        out2 = einsum('b n h i j, b n h j d -> b n h i d', attn2, v2)
        out2 = rearrange(out2, 'b n h mm d -> b n mm (h d)')
        out2 = out2.permute(0, 2, 1, 3)

        out = torch.cat([out1, out2], dim=-1).contiguous()
        out = self.to_out(out)
        out = rearrange(out, 'b (h w) (b0 b1) c  -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],
                        b0=w_size[0])
        return out

    def forward(self, x):
        """
        Args:
            x: [b, h, w, c]
        Returns:
            out: [b, h, w, c]
        """
        _, h, w, _ = x.shape
        w_size = self.window_size
        assert h % w_size[0] == 0 and w % w_size[1] == 0, 'fmap dimensions must be divisible by the window size'

        if self.only_local_branch:
            out = self.local_branch(x, h, w)
        else:
            out = self.dual_branch(x, h, w)
        return out
    
class FeedForwardSpa(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)
    
## Spatial Collaborative Transforme (Spa_CTB)
class Spa_CTB(nn.Module):
    def __init__(
            self,
            dim,
            window_size=8,
            dim_head=64,
            heads=8,
            num_blocks=1,
            img_size=128
    ):

        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, Spa_RSA(dim=dim, window_size=window_size, dim_head=dim_head, heads=heads, img_size=img_size,
                                    only_local_branch=(heads == 1))),
                PreNorm(dim, FeedForwardSpa(dim=dim))
            ]))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class FeedForwardSpe(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForwardSpe, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)

## Spectral Regrouping Self-attention (Spe_RSA)
class Spe_RSA(nn.Module):
    def __init__(self, dim, num_heads, win, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.win = win

        # Query, Key, Value 
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, 
                                    bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def local_branch(self, q, k, v, shape):
        b, c, h, w = shape
        q = q.view(b, c, h // self.win, self.win, w // self.win, self.win)
        k = k.view_as(q)
        v = v.view_as(q)

        q = q.permute(0, 2, 4, 1, 3, 5).reshape(-1, c, self.win * self.win)
        k = k.permute(0, 2, 4, 1, 3, 5).reshape(-1, c, self.win * self.win)
        v = v.permute(0, 2, 4, 1, 3, 5).reshape(-1, c, self.win * self.win)

        q = rearrange(q, 'n (head c) win -> n head c win', head=self.num_heads)
        k = rearrange(k, 'n (head c) win -> n head c win', head=self.num_heads)
        v = rearrange(v, 'n (head c) win -> n head c win', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.einsum('b h i d, b h d j -> b h i j', q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'n head c (h w) -> n (head c) h w', w=self.win)
        out = out.view(b, h // self.win, w // self.win, c, self.win, self.win)
        return out.permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)

    def global_branch(self, q, k, v, shape):
        _, _, h, w = shape
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = torch.einsum('b h i d, b h d j -> b h i j', q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        return rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
    
    def forward(self, x):
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)  

        out1 = self.local_branch(q, k, v, x.shape)
        out2 = self.global_branch(q, k, v, x.shape)

        out = torch.add(out1, out2).contiguous()
        return self.project_out(out)
    
## Spectral Collaborative Transforme (Spe_CTB)
class Spe_CTB(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 win, 
                 ffn_expansion_factor, 
                 bias, 
                 LayerNorm_type, 
                 num_blocks=1, 
                 conv_scale=1.0):
        super(Spe_CTB, self).__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                LayerNorm(dim, LayerNorm_type),
                Spe_RSA(dim, num_heads, win, bias),
                LayerNorm(dim, LayerNorm_type),
                FeedForwardSpe(dim, ffn_expansion_factor, bias)
            ]))
        
        self.conv_scale = conv_scale

    def forward(self, x):
        for (norm1, attn, norm2, ff) in self.blocks:
            res = x
            x = norm1(x)
            x = res + self.conv_scale * attn(x)
            x = x + ff(norm2(x))    
        return x
    
## MSI Feature Stream (MSI_FS)
class MSI_FS(nn.Module):
    def __init__(self, img_size=64, in_chans=34,
                 embed_dim=48,  dim_head=48, num_heads=[8,8,8],num_blocks=[1,1,1],
                 window_size=8, norm_layer=nn.LayerNorm):
        '''
        :param patch_size: for the embed conv
        :param in_chans: for the embed conv
        '''
        super(MSI_FS, self).__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.input_resolution = to_2tuple(img_size)
        self.in_chans = in_chans
        self.window_size = to_2tuple(window_size)

        self.conv = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=3, stride=1, padding=1)

        self.downsamples = nn.ModuleList([
            PatchMerging(dim=embed_dim, input_resolution=(self.img_size, self.img_size), norm_layer=norm_layer),
            PatchMerging(dim=embed_dim * 2, input_resolution=(self.img_size // 2, self.img_size // 2), norm_layer=norm_layer),
        ])

        self.Spa_CTBs = nn.ModuleList([
            Spa_CTB(dim=embed_dim, window_size=window_size, dim_head=dim_head,
                    heads=num_heads[0], num_blocks=num_blocks[0], img_size=self.img_size),
            Spa_CTB(dim=embed_dim * 2, window_size=window_size, dim_head=dim_head,
                    heads=num_heads[1], num_blocks=num_blocks[1], img_size=self.img_size // 2),
            Spa_CTB(dim=embed_dim * 4, window_size=window_size, dim_head=dim_head,
                    heads=num_heads[2], num_blocks=num_blocks[2], img_size=self.img_size // 4),
        ])

    def forward(self, x):
        _, _, H, _ = x.shape
        x = self.conv(x)  # B x em x H x W

        MSI_outputs = []
        for i, Spa_CTB in enumerate(self.Spa_CTBs):
            
            MSI_Out = Spa_CTB(x)
            MSI_outputs.append(MSI_Out)      

            if i < len(self.downsamples):
                MSI_Out = rearrange(MSI_Out, 'B C H W -> B (H W) C', H=H)  
                x = self.downsamples[i](MSI_Out)
                H //= 2 
                x = rearrange(x, 'B (H W) C -> B C H W', H=H)

        return MSI_outputs[0], MSI_outputs[1], MSI_outputs[2]

## HSI Feature Stream (HSI_FS)
class HSI_FS(nn.Module):
    def __init__(self, inp_channels=31, dim=32, num_heads=[8, 8, 8], group=8, ffn_expansion_factor=2.66,  LayerNorm_type = 'WithBias', bias=False):
        super(HSI_FS, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
       
        self.Spe_CTBs = nn.ModuleList([
            Spe_CTB(dim=dim, num_heads=num_heads[0], win=group, ffn_expansion_factor=ffn_expansion_factor, 
                    bias=bias, LayerNorm_type=LayerNorm_type),
            Spe_CTB(dim=dim // 2, num_heads=num_heads[1], win=group, ffn_expansion_factor=ffn_expansion_factor, 
                    bias=bias, LayerNorm_type=LayerNorm_type),
            Spe_CTB(dim=dim // 4, num_heads=num_heads[2], win=group, ffn_expansion_factor=ffn_expansion_factor, 
                    bias=bias, LayerNorm_type=LayerNorm_type),
        ])

        self.upsample_layers = nn.ModuleList([
            Upsample(dim, upscale_level=1),           # Upsample after stage 1
            Upsample(dim // 2, upscale_level=1)        # Upsample after stage 2
        ])

    def forward(self, x):
        emb = self.patch_embed(x)
        HSI_outputs = []
        for i, Spe_CTB in enumerate(self.Spe_CTBs):
            emb = Spe_CTB(emb)
            HSI_outputs.append(emb)

            # If not the last stage, upsample the output
            if i < len(self.upsample_layers):
                emb = self.upsample_layers[i](emb)

        return HSI_outputs[2], HSI_outputs[1], HSI_outputs[0]  
    
class MergeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(MergeBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.LeakyReLU(0.2, bias),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.LeakyReLU(0.2, bias)
        )

    def forward(self, x):
        return self.body(x)
    
class LGCT(nn.Module):
    def __init__(self, img_size=64, upscale=4, in_chans1=5, in_chans2=31, embed_dim=48, dim_head=48, num_heads1=[8, 8, 8],window_size=8,
                 group=8, dim=32, num_heads2=[8, 8, 8], ffn_expansion_factor=2.66,
                 LayerNorm_type = 'WithBias', bias=False, norm_layer=nn.LayerNorm,
                 ):
        super(LGCT, self).__init__()
        self.img_size = img_size
        self.scale_factor = upscale
        
        # Streams
        self.msi_stream = MSI_FS(img_size=img_size, in_chans=in_chans1, embed_dim=embed_dim ,
                                     dim_head=dim_head, num_heads=num_heads1, window_size=window_size)

        self.hsi_stream = HSI_FS(inp_channels=in_chans2, dim=dim*4, num_heads=num_heads2, group=group,
                                     ffn_expansion_factor=ffn_expansion_factor,  LayerNorm_type=LayerNorm_type, bias=bias)
        
        # Merge Blocks
        self.merge_d1 = MergeBlock(dim * 2, dim, bias=bias)
        self.merge_d2 = MergeBlock(dim * 4 + dim * 2, dim, bias=bias)
        self.merge_d3 = MergeBlock(dim * 8 + dim * 2, dim, bias=bias)

        # Upsampling
        self.up = Upsample(int(dim), upscale_level=2)

        # PatchMerging
        self.downsample1 = PatchMerging(dim=dim, input_resolution=(self.img_size, self.img_size),
                                      norm_layer=norm_layer)
        self.downsample2 = PatchMerging(dim=dim, input_resolution=(self.img_size // 2, self.img_size // 2),
                                      norm_layer=norm_layer)
        self.downsample3 = PatchMerging(dim=dim, input_resolution=(self.img_size // 4, self.img_size // 4),
                                      norm_layer=norm_layer)

        self.bottleneck = nn.Sequential(nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=bias),
                                nn.LeakyReLU(0.2, bias),
                                nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias),
                                nn.LeakyReLU(0.2, bias)
                                      )
        # Upsample merge blocks
        self.merge_u1 = MergeBlock(dim * 8 + dim + dim, dim, bias=bias)
        self.merge_u2 = MergeBlock(dim * 4 + dim + dim, dim, bias=bias)
        self.merge_u3 = nn.Sequential(
            nn.Conv2d(dim * 2 + dim + dim, dim, kernel_size=3, padding=1, bias=bias),
            nn.LeakyReLU(0.2, bias),
            nn.Conv2d(dim, in_chans2, kernel_size=1)
        )

    def forward(self, x, y):
        # x: LrHSI ; y: HrMSI
        x_U = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic')
        y_U_cat = torch.cat((y, x_U), 1)

        # MSI and HSI stream outputs
        M_stage1, M_stage2, M_stage3 = self.msi_stream(y_U_cat)
        H_stage3, H_stage2, H_stage1 = self.hsi_stream(x)

        # Multiscale Feature Symmetric Extraction
        merge1 = self.merge_d1(torch.cat((M_stage1, H_stage3), 1))
        merge1_down = rearrange(merge1, 'B C H W -> B (H W) C', H=merge1.shape[2])
        merge1_down = self.downsample1(merge1_down)  
        merge1_down = rearrange(merge1_down, 'B (H W) C -> B C H W', H=merge1.shape[2] // 2)

        merge2 = self.merge_d2(torch.cat((M_stage2, H_stage2, merge1_down), 1))
        merge2_down = rearrange(merge2, 'B C H W  -> B (H W) C', H=merge2.shape[2])  
        merge2_down = self.downsample2(merge2_down)  
        merge2_down = rearrange(merge2_down, 'B (H W) C -> B C H W', H=merge2.shape[2] // 2)

        merge3 = self.merge_d3(torch.cat((M_stage3, H_stage1, merge2_down), 1))
        merge3_down = rearrange(merge3, 'B C H W  -> B (H W) C', H=merge3.shape[2])  
        merge3_down = self.downsample3(merge3_down)  
        merge3_down = rearrange(merge3_down, 'B (H W) C -> B C H W', H=merge3.shape[2] // 2)

        bottleneck_feature = self.bottleneck(merge3_down)
        
        # Upsample and merge the features
        merge4 = self.merge_u1(torch.cat((M_stage3, H_stage1, self.up(bottleneck_feature), merge3), 1))
        merge5 = self.merge_u2(torch.cat((M_stage2, H_stage2, self.up(merge4), merge2), 1))
        output = self.merge_u3(torch.cat((M_stage1, H_stage3, self.up(merge5), merge1), 1))
       
        return output + x_U



