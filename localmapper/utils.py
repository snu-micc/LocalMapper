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

from .models import LocalMapper

atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
             'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
             'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
             'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir',
             'Ce', 'Gd', 'Ga', 'Cs']

def clean_reactant_map(rxn):
    r, p = rxn.split('>>')
    r_mol = Chem.MolFromSmiles(r)
    [atom.SetAtomMapNum(0) for atom in r_mol.GetAtoms()]
    r = Chem.MolToSmiles(r_mol, canonical = False)
    return '>>'.join([r, p])

def get_adm(mol, max_distance = 4):
    dm = Chem.GetDistanceMatrix(mol)
    dm[dm > 100] = -1 # remote (different molecule)
    dm[dm > max_distance] = max_distance + 1 # remote (same molecule)
    dm[dm == -1] = max_distance + 2 # remote (different molecule)
    return dm

def pad_atom_distance_matrix(adm_list):
    max_size = max([adm.shape[0] for adm in adm_list])
    adm_list = [torch.LongTensor(np.pad(adm, (0, max_size - adm.shape[0]), 'maximum')).unsqueeze(0) for adm in adm_list]
    return torch.cat(adm_list, dim = 0)

def init_featurizer():
    node_featurizer = WeaveAtomFeaturizer(atom_types = atom_types)
    edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
    graph_function = partial(mol_to_bigraph, add_self_loop=True, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer, canonical_atom_order=False)
    return node_featurizer, edge_featurizer, graph_function

def get_configure(config_path, node_featurizer, edge_featurizer):
    with open(config_path, 'r') as f:
        config = json.load(f)
    config['in_node_feats'] = node_featurizer.feat_size()
    config['in_edge_feats'] = edge_featurizer.feat_size()
    return config

def load_model(exp_config, model_path, device):
    model = LocalMapper(
        node_in_feats=exp_config['in_node_feats'],
        edge_in_feats=exp_config['in_edge_feats'],
        node_out_feats=exp_config['node_out_feats'],
        edge_hidden_feats=exp_config['edge_hidden_feats'],
        num_step_message_passing=exp_config['num_step_message_passing'],
        attention_heads = exp_config['attention_heads'],
        attention_layers = exp_config['attention_layers'])
    model = model.to(device)

    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    return model            

def predict(model, device, rgraphs, pgraphs):
    rbg, pbg = dgl.batch(rgraphs),  dgl.batch(pgraphs)
    rbg.set_n_initializer(dgl.init.zero_initializer), pbg.set_n_initializer(dgl.init.zero_initializer)
    rbg.set_e_initializer(dgl.init.zero_initializer), pbg.set_e_initializer(dgl.init.zero_initializer)
    rbg, pbg = rbg.to(device), pbg.to(device)
    rnode_feats, pnode_feats = rbg.ndata.pop('h').to(device), pbg.ndata.pop('h').to(device)
    redge_feats, pedge_feats = rbg.edata.pop('e').to(device), pbg.edata.pop('e').to(device)
    with torch.no_grad():
        predicitons = model(rbg, pbg, rnode_feats, pnode_feats, redge_feats, pedge_feats)
    return predicitons
