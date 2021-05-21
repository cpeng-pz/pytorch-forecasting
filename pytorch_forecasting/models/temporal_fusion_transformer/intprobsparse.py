import torch
import torch.nn as nn
import numpy as np

from math import sqrt


class PaddedProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super(PaddedProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    # Algorithm 1 Steps 2--6 (return QK^T)
    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top


    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        # U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        # u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 
        U_part = self.factor * torch.ceil(torch.log(torch.tensor(L_K).float())).int() # c*ln(L_k)
        u = self.factor * (torch.ceil(torch.log(torch.tensor(L_Q).float())).int()) # c*ln(L_q)
        
        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        # scores = torch.zeros(B, H, L_Q, scores_top.shape[-1])
        scores = torch.zeros(B, H, L_Q, L_K).to(scores_top.device)
        scores[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :] = scores_top

        if self.mask_flag: # decoder's self-attention
            _mask = torch.ones(L_Q, scores.shape[-1], dtype=torch.bool).to(scores.device).triu(1)
            scores.masked_fill_(_mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)
        
        return attn


class InterpretableAttentionLayer(nn.Module):
    def __init__(self, n_head: int, d_model: int, dropout: float = 0.0):
        super(InterpretableAttentionLayer, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head

        self.inner_attention = PaddedProbAttention(False, factor=3, attention_dropout=dropout)
        self.query_projection = nn.Linear(d_model, self.d_q * n_head)
        self.key_projection = nn.Linear(d_model, self.d_k * n_head)
        self.value_projection = nn.Linear(d_model, self.d_v) # TFT 论文公式(14-16)中的 W_V
        self.out_projection = nn.Linear(self.d_v, d_model) # TFT 论文中公式（13）中的 W_H

    # 复制于 sub_modules.py 里面的 InterpretableMultiHeadAttention，应该可以优化初始权重
    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, q, k, v, mask):
        B, L_Q, _ = q.shape
        _, S, _ = k.shape
        H = self.n_head

        queries_h = self.query_projection(q).view(B, L_Q, H, -1)
        keys_h = self.key_projection(k).view(B, S, H, -1)
        v = self.value_projection(v) # value 不用分heads，直接就映射一下，对应 TFT 论文公式(14-16)中的 W_V

        # attns.shape == (B, H, L, S), 即 (B, H, L_Q, L_K)，因为里面 transpose(1,2) 了
        attns = self.inner_attention(
            queries_h,
            keys_h,
            v,
            mask
        )
        
        attn = attns.mean(dim=1)
        out =  attn @ v

        return self.out_projection(out), attn

