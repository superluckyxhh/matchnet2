import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from copy import deepcopy

from configs import dynamic_load
from modules.GeMPoolFormer1d import GeMPoolFormer

###### 
def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1
###########

def MLP(channels, do_bn=True):
    layers = []
    n = len(channels)
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    
    return nn.Sequential(*layers)


def norm_kpts(kpts, image_shape_list):
    height, width = image_shape_list[:, 0], image_shape_list[:, 1]
    one = kpts.new_tensor(1)
    size = torch.cat([(one*width).unsqueeze(-1), (one*height).unsqueeze(-1)], dim=-1)
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.8
    return (kpts - center[:, None, :]) / scaling[:, None, :]


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def log_dual_softmax(scores, temperature=0.1, eps=1e-10):
    scores = scores / temperature
    scores_col = F.softmax(scores, dim=1)
    #NOTE: DEBUG
    # print('scores col:\n', scores_col)
    scores_row = F.softmax(scores, dim=2)
    #NOTE: DEBUG
    # print('scores row:\n', scores_row)
    scores = torch.log(scores_col * scores_row + eps)
    return scores


class KeypointEncoder(nn.Module):
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)
    
    def forward(self, kpts, scores):
        # kpts shape: [b, n, 2] score shape:[b, n]
        inputs = torch.cat([kpts.transpose(1, 2), scores.unsqueeze(1)], dim=1)
        return self.encoder(inputs)


class MatchingNet(nn.Module):
    """
    Like SuperGlue :
    Inputs: kpts desc scores
    Change: GeMpoolFormer1d  No Cross Attention
    Return: scores (with dustbin)
    """
    def __init__(
        self, feature_dim, 
        kpt_encoder, num_layers,
        score_type, use_scale=False):

        super().__init__()
        self.feature_dim = feature_dim
        self.score_type = score_type
        self.kenc = KeypointEncoder(feature_dim, kpt_encoder)

        # self.gemformer0 = GeMPoolFormer(feature_dim, num_layers, use_layer_scale=use_scale)

        # self.gemformer1 = GeMPoolFormer(feature_dim, num_layers, use_layer_scale=use_scale)

        #TODO: 确定是否是pool的问题
        self.transformer = AttentionalGNN(feature_dim,
                        ['self','cross']*6)

        self.final_proj = nn.Conv1d(feature_dim, feature_dim,
                                    kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)
        
    
    def forward(self, data):
        # Take out Inputs
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        desc0, desc1 = data['descriptors0'].transpose(1, 2), data['descriptors1'].transpose(1, 2)
        scores0, scores1 = data['scores0'], data['scores1']
        ori_im0_shape = data['ori_im0_shapes'].squeeze(1)
        ori_im1_shape = data['ori_im1_shapes'].squeeze(1)

        # Keypoint normalization.
        kpts0 = norm_kpts(kpts0, ori_im0_shape)
        kpts1 = norm_kpts(kpts1, ori_im1_shape)

        # Keypoint MLP encoder.
        tmp_0 = self.kenc(kpts0, scores0)
        tmp_1 = self.kenc(kpts1, scores1)
        # print('ken0 max min:\n', torch.max(tmp_0), torch.min(tmp_0))
        # print('ken1 max min:\n', torch.max(tmp_1), torch.min(tmp_1))
        desc0 = desc0 + self.kenc(kpts0, scores0)
        desc1 = desc1 + self.kenc(kpts1, scores1)

        # print('kenc add desc0 max min:\n', torch.max(desc0), torch.min(desc0))
        # print('kenc add desc1 max min:\n', torch.max(desc1), torch.min(desc1))
    
        # TODO: GeMPoolFormer network. 
        # desc0 = self.gemformer0(desc0)
        # desc1 = self.gemformer1(desc1)
        # print('pool desc0 max min:\n', torch.max(desc0), torch.min(desc0))
        # print('pool desc1 max min:\n', torch.max(desc1), torch.min(desc1))

        #TODO: Superglue GNN
        desc0, desc1 = self.transformer(desc0, desc1)
        # print('gnn desc0 max min:\n', torch.max(desc0), torch.min(desc0))
        # print('gnn desc1 max min:\n', torch.max(desc1), torch.min(desc1))
        ##################

        # Final MLP projection.
        desc0, desc1 = self.final_proj(desc0), self.final_proj(desc1)
        # print('proj desc0 max min:\n', torch.max(desc0), torch.min(desc0))
        # print('proj desc1 max min:\n', torch.max(desc1), torch.min(desc1))

        if self.score_type == 'OT':
            # Compute matching descriptor distance.
            scores = torch.einsum('bdn,bdm->bnm', desc0, desc1)
            # print('cosine matrix:\n', scores)
            scores = log_optimal_transport(
                        scores, self.bin_score,
                        iters=50)
            # print('OT scores:\n', scores)
            # scores = scores / self.feature_dim ** .5
        elif self.score_type == 'DS':
            desc0 = desc0 / (desc0.shape[1] ** .5)
            desc1 = desc1 / (desc1.shape[1] ** .5)
            # desc0 = F.normalize(desc0, p=2, dim=1)
            # desc1 = F.normalize(desc1, p=2, dim=1)
            # NOTE: DEBUG
            # print('norm desc0 max min:', torch.max(desc0), torch.min(desc0))
            # print('norm desc1 max min:', torch.max(desc1), torch.min(desc1))
            scores = torch.einsum('bdn,bdm->bnm', desc0, desc1) 
            # NOTE: DEBUG
            # print('DS cosine matrix:\n', scores)
            scores = log_dual_softmax(scores)
            # NOTE: DEBUG
            tmp_max_v = torch.max(scores)
            tmp_min_v = torch.min(scores)
            # print('DS scores max:', tmp_max_v)
            # print('DS scores min:', tmp_min_v)
            # scores = scores / self.feature_dim ** .5
        else:
            print('Invalid score typr')
            exit(1)
        
        ##
        ### NOTE: DEBUG
        # GemPooling , max scores :3000, min scores:-75
        # Pooling, max scores:  min scores：
        # Transformer， max scores: min scores:
        # print(f'{self.score_type} scores:', scores)
        # print()
        # exit(1)
        return scores