import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import math
import sklearn
import dgl

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, positional_number = 0, dropout = 0.1, return_att=False):
        super(MultiHeadAttention, self).__init__()
        self.return_att = return_att
        self.p_k = positional_number
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        if self.p_k != 0:
            self.relative_k = nn.Parameter(torch.randn(self.p_k, self.d_k))
            self.relative_v = nn.Parameter(torch.randn(self.p_k, self.d_k))
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.gating = nn.Linear(d_model, d_model)
        self.to_out = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.reset_parameters()
        
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.constant_(self.gating.weight, 0.)
        nn.init.constant_(self.gating.bias, 1.)
        
    def one_hot_embedding(self, labels):
        y = torch.eye(self.p_k)
        return y[labels]
    
    def forward(self, x_q, x_k, mask=None):
        bs = x_q.size(0)
        x_q = self.layer_norm(x_q)
        x_k = self.layer_norm(x_k)
        q = self.q_linear(x_q)
        k = self.k_linear(x_k)
        v = self.v_linear(x_k)

        k1 = k.view(bs, -1, self.h, self.d_k).transpose(1,2)
        q1 = q.view(bs, -1, self.h, self.d_k).transpose(1,2)
        v1 = v.view(bs, -1, self.h, self.d_k).transpose(1,2)
        attn1 = torch.matmul(q1, k1.permute(0, 1, 3, 2))
        
        if self.p_k == 0:
            attn = attn1/math.sqrt(self.d_k)
        else:
            gpms = self.one_hot_embedding(gpm.unsqueeze(1).repeat(1, self.h, 1, 1)).to(x.device)
            attn2 = torch.matmul(q1, self.relative_k.transpose(0, 1))
            attn2 = torch.matmul(gpms, attn2.unsqueeze(-1)).squeeze(-1)
            attn = (attn1 + attn2) /math.sqrt(self.d_k)
        
        if mask:
            mask = mask.bool()
            mask = mask.unsqueeze(1).repeat(1,mask.size(-1),1)
            mask = mask.unsqueeze(1).repeat(1,attn.size(1),1,1)
            attn[~mask] = float(-9e9)

        if self.return_att:
            return attn[:, 0]
            
        attn = torch.softmax(attn, dim=-1) * torch.softmax(attn, dim=-2)
        v1 = v.view(bs, -1, self.h, self.d_k).permute(0, 2, 1, 3)
        output = torch.matmul(self.dropout1(attn), v1)
        output = output.transpose(1,2).contiguous().view(bs, -1, self.d_model).squeeze(-1)
        output = self.to_out(output * self.gating(x_q).sigmoid()) # gate attention
        return self.dropout2(output)

        
    
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * torch.pow(x, 3)))) 
    
class FeedForward(nn.Module):
    def __init__(self, d_model, dropout = 0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*2, d_model))
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.layer_norm(x)
        return self.net(x)

class CrossReactivityAttention(nn.Module):
    def __init__(self, d_model, heads, n_layers = 3, positional_number = 0, dropout = 0.1):
        super(CrossReactivityAttention, self).__init__()
        self.positional_number = positional_number
        self.n_layers = n_layers
        att_stack = []
        pff_stack = []
        for _ in range(n_layers):
            att_stack.append(MultiHeadAttention(heads, d_model, positional_number, dropout))
            pff_stack.append(FeedForward(d_model, dropout))
#         att_stack.append(MultiHeadAttention(1, d_model, positional_number, dropout, return_att=True))
        self.att_stack = nn.ModuleList(att_stack)
        self.pff_stack = nn.ModuleList(pff_stack)
        
    def forward(self, x_q, x_k, mask=None):
        for n in range(self.n_layers):
            m = self.att_stack[n](x_q, x_k, mask)
            x_q = x_q + self.pff_stack[n](x_q+m)
#         score = self.att_stack[-1](x_q, x_k, mask)
        return x_q
    