import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers.helpers import to_2tuple
from timm.models.layers import DropPath, trunc_normal_


class LayerNormChannel1d(nn.Module):
    """ 
    LayerNorm only for channel dimension
    Args: token shape [B, N, C]
    Return: [B, N, C]
    """
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.ones(num_channels))
        self.eps = eps
    
    def forward(self, x):
        u = x.mean(dim=-1, keepdim=True)
        s = (x - u).pow(2).mean(dim=-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1) * x + self.bias.unsqueeze(-1)
    
        return x


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None,
        out_features=None, act_layer=nn.GELU, drop=0.
    ):  
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Conv1d(in_features, hidden_features, kernel_size=1) 
        self.fc2 = nn.Conv1d(hidden_features, out_features, kernel_size=1)
        self.bn = nn.BatchNorm1d(hidden_features)
        self.act_layer = act_layer()
        self.act_layer = act_layer()
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.act_layer(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act_layer(x)
        x = self.drop(x)

        return x


class GeMPool(nn.Module):
    def __init__(self, p=3, pool_size=3, eps=1e-5):
        super().__init__()
        self.eps = eps
        # NOTE: Training or Constant
        # self.p = nn.Parameter(torch.ones(1) * p)
        self.p = p
        self.pool = nn.AvgPool1d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)
    
    def forward(self, x):
        gem_x = x.clamp(min=self.eps).pow(self.p)
        gem_x = self.pool(gem_x)
        gem_x = gem_x.pow(1./self.p)

        return gem_x - x


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class GeMPoolFormerBlock(nn.Module):
    def __init__(
        self, feature_dim, mlp_ratio=4,
        act_layer=nn.GELU, norm_layer=GroupNorm,
        drop=0., drop_path=0., use_layer_scale=True,
        layer_scale_init_value=1e-5
    ):
        super().__init__()
        self.norm1 = norm_layer(feature_dim)
        # NOTE: 
        self.token_mixer = Pooling()
        # self.token_mixer = GeMPool()
        self.norm2 = norm_layer(feature_dim)
        mlp_hidden_dim = int(feature_dim * mlp_ratio)
        self.mlp = MLP(in_features=feature_dim, 
                        hidden_features=mlp_hidden_dim,
                        act_layer=act_layer,
                        drop=drop)
                    
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((feature_dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((feature_dim)), requires_grad=True)
    

    def forward(self, x):
        if self.use_layer_scale:
            ### TEST
            tm = self.token_mixer(self.norm1(x))
            # print('tm max min:\n', torch.max(tm), torch.min(tm))
            scale_tm = self.layer_scale_1.unsqueeze(-1) * tm
            # print('scale tm max min:\n', torch.max(scale_tm), torch.min(scale_tm))
            # print()
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1) * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1) * self.mlp(self.norm2(x)))
        else:
            # x = self.drop_path(self.token_mixer(self.norm1(x))) + x
            x = self.drop_path(self.mlp(self.norm2(x))) + x
        # print('x max min:\n', torch.max(x), torch.min(x))
        return x
    

class GeMPoolFormer(nn.Module):
    def __init__(self, feature_dim, num_layers, drop=0., drop_path=0., use_layer_scale=True):
        super().__init__()
        self.layers = nn.ModuleList([
                GeMPoolFormerBlock(feature_dim=feature_dim,  drop=drop, drop_path=drop_path, use_layer_scale=use_layer_scale)
                for _ in range(num_layers)
        ])
        # self.norm_layers = nn.ModuleList([
        #     GroupNorm(feature_dim) for _ in range(num_layers)
        # ])
    
    def forward(self, desc):
        # for layer, norm in zip(self.layers, self.norm_layers):
        #     delta = layer(desc)
        #     norm_delta = norm(delta)
        #     desc = desc + norm_delta

        for layer in self.layers:
            # delta = layer(desc)
            # desc = desc + delta
            desc = layer(desc)

        return desc
        

if __name__ == '__main__':
    im = torch.randn((1, 3, 480, 640))
    feats = torch.randn((4, 256, 1024))
    desc0 = torch.randn((4, 256, 1024)).to('cuda')
    desc1 = torch.randn((4, 256, 1024)).to('cuda')

    """ Test LayerNormChannel 1d """
    # lnc = LayerNormChannel1d(256)
    # out = lnc(feats)

    """ Test Group Norm """
    # gn = GroupNorm(256)
    # out = gn(feats)

    """ Test MLP """
    # mlp = MLP(256, 256, 256)
    # out = mlp(feats)

    """ Test GeM Pool """
    # gem = GeMPool(p=3)
    # out = gem(feats)

    """ Test GeMPoolFormerBlock """
    # gemformer = GeMPoolFormerBlock(feature_dim=256, use_layer_scale=True)
    # out = gemformer(feats)

    """ Test GeMPoolFormer """
    # gpf = GeMPoolFormer(256, 2).to('cuda')
    # mdesc0 = gpf(desc0)
    
    print()