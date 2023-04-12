import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import math
import sklearn
import dgl


def expand_vision(radm, padm, max_distance = 4):
    rsize, psize = radm.size(0), padm.size(0)
    vision = torch.ones((rsize+psize, rsize+psize))
    vision[:rsize, :rsize] = radm
    vision[rsize:, rsize:] = padm
    vision[:rsize, rsize:] = max_distance + 2 # r -> p
    vision[rsize:, :rsize] = max_distance + 3 # p -> r
    return vision.long()
    
def batch_att(att_nn, linear_nn, rbg, pbg, feats_r, feats_p, radms, padms):
    mapping_feats = []
    rbg.ndata['h'], pbg.ndata['h'] = feats_r, feats_p
    gs_r, gs_p = dgl.unbatch(rbg), dgl.unbatch(pbg)
    for g_r, g_p, radm, padm in zip(gs_r, gs_p, radms, padms):
        x_r, x_p = g_r.ndata['h'], g_p.ndata['h']
        x_rxn = torch.cat([x_r, x_p], dim = 0)
        rxn_adm = expand_vision(radm, padm)
        x_rxn, att = att_nn(x_rxn, rxn_adm)
        x_r, x_p = x_rxn[:x_r.size(0)], x_rxn[x_r.size(0):]
        x_pr = torch.cat([x_p.unsqueeze(1).repeat(1, x_r.size(0), 1), x_r.unsqueeze(0).repeat(x_p.size(0), 1, 1)], dim = -1)
        map_out = linear_nn(x_pr)
        mapping_feats.append(map_out.reshape(x_p.size(0), -1))
    return mapping_feats


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, positional_number = 5, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()
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

    def forward(self, x_q, x_k, gpm = None, mask=None):
        atom_size = x_q.size(0)
        x_q, x_k = self.layer_norm(x_q), self.layer_norm(x_k)
        q = self.q_linear(x_q)
        k = self.k_linear(x_k)
        v = self.v_linear(x_k)
        
        q1 = q.view(-1, self.h, self.d_k).transpose(0, 1) # h, atom_size, d_k
        k1 = k.view(-1, self.h, self.d_k).transpose(0, 1) # h, atom_size, d_k
        v1 = v.view(-1, self.h, self.d_k).transpose(0 ,1) # h, atom_size, d_k
        attn1 = torch.matmul(q1, k1.transpose(1 ,2)) # h, atom_size, atom_size
        if self.p_k == 0:
            attn = attn1/math.sqrt(self.d_k)
        else:
            gpms = self.one_hot_embedding(gpm.unsqueeze(0).repeat(self.h, 1, 1)).to(q.device) # atom_size, head, atom_size, head
            attn2 = torch.matmul(q1, self.relative_k.transpose(0, 1))
            attn2 = torch.matmul(gpms, attn2.unsqueeze(-1)).squeeze(-1)
            attn = (attn1 + attn2) /math.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(0).repeat(self.h, 1, 1)
            
            attn[~mask] = float(-9e9)
            attn_pxr = torch.softmax(attn, dim=-1)
            attn_rxp = torch.softmax(attn, dim=-2)
            attn = attn_pxr * attn_rxp
            
        else:
            attn = torch.softmax(attn, dim=-1)
        output = torch.matmul(self.dropout1(attn), v1).transpose(0, 1).contiguous().view(-1, self.d_model).squeeze(-1)
        output = self.to_out(output * self.gating(x_q).sigmoid()) # gate self attention
        return self.dropout2(output), attn
        
    
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

class Global_Reactivity_Attention(nn.Module):
    def __init__(self, d_model, heads, n_layers = 3, positional_number = 8, dropout = 0.1):
        super(Global_Reactivity_Attention, self).__init__()
        self.positional_number = positional_number
        self.n_layers = n_layers
        att_stack = []
        pff_stack = []
        for _ in range(n_layers):
            att_stack.append(MultiHeadAttention(heads, d_model, positional_number, dropout))
            pff_stack.append(FeedForward(d_model, dropout))
        self.att_stack = nn.ModuleList(att_stack)
        self.pff_stack = nn.ModuleList(pff_stack)
        
    def forward(self, x, adm):
        att_scores = []
        for n in range(self.n_layers):
            m, att_score = self.att_stack[n](x, x, adm)
            x = x + self.pff_stack[n](x+m)
            att_scores.append(att_score)
        return x, att_scores
    