import torch
import torch.nn as nn
import torch.nn.functional as F
from common import NoGradientError

class MeanCriterion(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps
    
    def forward(self, scores, assigns):
        batch_dim = scores.shape[0]
        scores = -scores
        losses = 0
        for b in range(batch_dim):
            assign = assigns[b]
            score = scores[b]
            assign = assign[:-1, :-1]
            pos_score = score[assign]
            losses += torch.mean(pos_score)

        losses = losses / float(batch_dim)

        return {'loss': losses}