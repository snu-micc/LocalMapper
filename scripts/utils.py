import torch
import sklearn
import dgl
import json
import os
import copy
import numpy as np
import glob
from functools import partial
from collections import defaultdict
import sys

import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, lr_scheduler

from dgl.data.utils import Subset
from dgllife.utils import WeaveAtomFeaturizer, CanonicalBondFeaturizer, mol_to_bigraph, EarlyStopping

from models import LocalMapper
from dataset import ReactionDataset, mkdir_p

def get_user_name(args):
    if 'chemist_name' in args:
        return args['chemist_name']
    else:
        return glob.glob('../manual/*.user')[0].split('\\')[-1].split('.')[0]

def init_featurizer(args):
    atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
             'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
             'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
             'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir',
             'Ce', 'Gd', 'Ga', 'Cs']
    args['node_featurizer'] = WeaveAtomFeaturizer(atom_types = atom_types)
    args['edge_featurizer'] = CanonicalBondFeaturizer(self_loop=True)
    args['mol_to_graph'] = partial(mol_to_bigraph, add_self_loop=True, node_featurizer=args['node_featurizer'], edge_featurizer=args['edge_featurizer'], canonical_atom_order=False)
    return args

def get_configure(args):
    with open(args['config_path'], 'r') as f:
        config = json.load(f)
    config['in_node_feats'] = args['node_featurizer'].feat_size()
    config['in_edge_feats'] = args['edge_featurizer'].feat_size()
    return config
    
def load_dataloader(args, test = False):
    dataset = ReactionDataset(args)
    train_set, val_set, test_set = Subset(dataset, dataset.train_idx), Subset(dataset, dataset.val_idx), Subset(dataset, dataset.test_idx)
    
    if test:
        test_loader = DataLoader(dataset=test_set, batch_size=args['batch_size'], shuffle=False, collate_fn=collate_molgraphs)
        return test_loader
    else:
        train_loader = DataLoader(dataset=train_set, batch_size=args['batch_size'], shuffle=True, collate_fn=collate_molgraphs)
        val_loader = DataLoader(dataset=val_set, batch_size=args['batch_size'], shuffle=False, collate_fn=collate_molgraphs)
    return train_loader, val_loader
    
def Twoway_CrossEntropyLoss(pred, true):
    normalized_pred = torch.nn.LogSoftmax(dim = 0)(pred) + torch.nn.LogSoftmax(dim = 1)(pred)
    loss = nn.NLLLoss(reduction = 'none')
    return loss(normalized_pred, true)
    
def load_model(args):
    exp_config = get_configure(args)
    model = LocalMapper(
        node_in_feats=exp_config['in_node_feats'],
        edge_in_feats=exp_config['in_edge_feats'],
        node_out_feats=exp_config['node_out_feats'],
        edge_hidden_feats=exp_config['edge_hidden_feats'],
        num_step_message_passing=exp_config['num_step_message_passing'],
        attention_heads = exp_config['attention_heads'],
        attention_layers = exp_config['attention_layers'])
    model = model.to(args['device'])

    if args['mode'] == 'train':
        loss_criterion = Twoway_CrossEntropyLoss
        optimizer = Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, min_lr=1e-4)
        stopper = EarlyStopping(mode = 'lower', patience=args['patience'], filename=args['model_path'])
        return model, loss_criterion, optimizer, scheduler, stopper
    
    else:
        model.load_state_dict(torch.load(args['model_path'], map_location=args['device'])['model_state_dict'])
        return model            

def mask_labels(labels_list):
    fixed_labels_list = []
    masks_list = []
    for labels in labels_list:
        fixed_labels = [label if label >= 0 else 0 for label in labels]
        masks = [int(label >= 0) for label in labels]
        fixed_labels_list.append(torch.LongTensor(fixed_labels))
        masks_list.append(torch.LongTensor(masks))
        
    return fixed_labels_list, masks_list

def collate_molgraphs(data):
    idxs, rxns, rgraphs, pgraphs, labels_list, weights_list = map(list, zip(*data))
    labels_list, masks_list = mask_labels(labels_list)
    rbg, pbg = dgl.batch(rgraphs),  dgl.batch(pgraphs)
    rbg.set_n_initializer(dgl.init.zero_initializer) 
    pbg.set_n_initializer(dgl.init.zero_initializer)
    rbg.set_e_initializer(dgl.init.zero_initializer)
    pbg.set_e_initializer(dgl.init.zero_initializer)
    return idxs, rxns, rbg, pbg, labels_list, masks_list, weights_list

def predict(args, model, rbg, pbg):
    rbg, pbg = rbg.to(args['device']), pbg.to(args['device'])
    rnode_feats, pnode_feats = rbg.ndata.pop('h').to(args['device']), pbg.ndata.pop('h').to(args['device'])
    redge_feats, pedge_feats = rbg.edata.pop('e').to(args['device']), pbg.edata.pop('e').to(args['device'])
    return model(rbg, pbg, rnode_feats, pnode_feats, redge_feats, pedge_feats)