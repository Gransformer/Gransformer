import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


# class EnsembleScaledDotProductAttention(nn.Module):
#     ''' Ensemble Scaled Dot-Product Attention '''
#
#     def __init__(self, temperature, attn_dropout=0.1):
#         super().__init__()
#         self.temperature = temperature
#         self.dropout = nn.Dropout(attn_dropout)
#
#     def forward(self, q, k, v, mask=None):
#
#         len_q, len_k, len_v = q.size(3), k.size(3), v.size(3)
#
#         attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
#
#         if mask is not None:
#             ens_1 = len_q // mask.size(-2)
#             ens_2 = len_k // mask.size(-1)
#             mask = mask.unsqueeze(-3).unsqueeze(-1).repeat(1, 1, ens_1, 1, 1, ens_2)
#             mask = mask.transpose(2,3).transpose(4,5)
#             mask = mask.reshape(mask.size(0), mask.size(1), mask.size(2) * mask.size(3), mask.size(4) * mask.size(5))
#             attn = attn.masked_fill(mask == 0, -1e9)
#
#         attn = self.dropout(F.softmax(attn, dim=-1))
#         output = torch.matmul(attn, v)
#
#         return output, attn
