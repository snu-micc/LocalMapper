import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import sklearn
import dgl
import dgllife
from dgllife.model import MPNNGNN

from .model_utils import GELU, MultiHeadAttention, CrossReactivityAttention
    
class LocalMapper(nn.Module):
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats,
                 edge_hidden_feats,
                 num_step_message_passing,
                 attention_heads,
                 attention_layers
                ):
        super(LocalMapper, self).__init__()
           
        self.mpnn = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        
        self.cross_att = CrossReactivityAttention(node_out_feats, attention_heads, attention_layers)
        self.amm_att = MultiHeadAttention(1, node_out_feats, dropout=0.2, return_att=True)


    def batch_att(self, rbg, pbg, feats_r, feats_p):
        device = feats_r.device
        mapping_scores = []
        rbg.ndata['h'], pbg.ndata['h'] = feats_r, feats_p
        rgs, pgs = dgl.unbatch(rbg), dgl.unbatch(pbg)
        batch_p = pad_sequence([g.ndata['h'] for g in pgs], batch_first=True, padding_value= 0) # batch, patom_n, hidden_n
        batch_r = pad_sequence([g.ndata['h'] for g in rgs], batch_first=True, padding_value= 0) # batch, ratom_n, hidden_n
        
        batch_p = self.cross_att(batch_p, batch_r)
        aam_score = self.amm_att(batch_p, batch_r)
        aam_score = [score[:pg.num_nodes(), :rg.num_nodes()] for pg, rg, score in zip(pgs, rgs, aam_score)]
        return aam_score

    def forward(self, rbg, pbg, rnode_feats, pnode_feats, redge_feats, pedge_feats):
        rnode_feats = self.mpnn(rbg, rnode_feats, redge_feats)
        pnode_feats = self.mpnn(pbg, pnode_feats, pedge_feats)
        mapping_scores = self.batch_att(rbg, pbg, rnode_feats, pnode_feats)
        return mapping_scores
    
    
    
    