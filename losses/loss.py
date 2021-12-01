import torch
import torch.nn as nn
import torch.nn.functional as F
from common import NoGradientError

class Criterion(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps
    
    def forward(self, scores, assigns):
        #### Test how many unmatches in assignment ####
        # batch_id = 2
        # tmp_score = scores[batch_id]
        # bin_score_col = tmp_score[:, -1]
        # bin_score_row = tmp_score[-1, :]

        # tmp_assign = assigns[batch_id]
        # bin_col = tmp_assign[:, -1]
        # bin_row = tmp_assign[-1, :]
        
        # unmatched_row = bin_score_row[bin_row]
        # unmatched_col = bin_score_col[bin_col]
        # print(f'numatched nums row:{unmatched_row.shape[0]}')
        # print(f'numatched nums col:{unmatched_col.shape[0]}')
        ###############################################
        batch = scores.shape[0]
        scores = torch.clamp(scores, self.eps, 1-self.eps)
        # print('clamp scores:\n', scores)
        losses = 0
        for b in range(batch):
            score = scores[b]
            assign = assigns[b]
            if assign.sum() == 0:
                print('Found zero GT matches.')
                raise NoGradientError

            p = score[assign]
            loss = - torch.log(p)
            losses += loss.mean()

        losses = losses / float(batch)

        return {'loss': losses}