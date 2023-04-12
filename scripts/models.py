import torch
import torch.nn as nn

import sklearn
import dgl
import dgllife
from dgllife.model import MPNNGNN

from model_utils import batch_att, GELU, Global_Reactivity_Attention

class LocalMapper(nn.Module):
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats,
                 edge_hidden_feats,
                 num_step_message_passing,
                 attention_heads,
                 attention_layers):
        super(LocalMapper, self).__init__()
           
        self.mpnn = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        
        self.att = Global_Reactivity_Attention(node_out_feats, attention_heads, attention_layers)
        
        self.linear =  nn.Sequential(
                           nn.Linear(node_out_feats*2, node_out_feats), 
                           GELU(),
                           nn.Dropout(0.2),
                           nn.Linear(node_out_feats, 1))

    def forward(self, rbg, pbg, rnode_feats, pnode_feats, redge_feats, pedge_feats, radms, padms):
        rnode_feats, pnode_feats = self.mpnn(rbg, rnode_feats, redge_feats), self.mpnn(pbg, pnode_feats, pedge_feats)
        mapping_feats = batch_att(self.att, self.linear, rbg, pbg, rnode_feats, pnode_feats, radms, padms)
        return mapping_feats