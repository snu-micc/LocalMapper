import json
from functools import partial
import numpy as np
import pandas as pd
import torch
from torch import nn
import sklearn

from rdkit import Chem
import dgl
from dgllife.utils import mol_to_bigraph, WeaveAtomFeaturizer, CanonicalBondFeaturizer
from functools import partial

import sys
sys.path.append('scripts')
from Sample import load_fixed_templates
from dataset import get_adm, clean_reactant_map
from utils import init_featurizer, load_model
from atom_mapper import AtomMapper, prediction2map

def init_AtomMapper(dataset, chemist_name, iteration, device):
    data_dir = 'data/%s/' % dataset
    sample_dir = '%s/%s' % (data_dir, chemist_name)
    config_path = 'data/configs/default_config.json'
    model_path = 'models/%s/%s/LocalMapper_%d.pth' % (dataset, chemist_name, iteration)
    args = {'mode': 'test', 'iteration': iteration, 'data_dir': data_dir, 'sample_dir': sample_dir, 'config_path': config_path, 'model_path': model_path, 'device': device}
    args = init_featurizer(args)
    model = load_model(args)
    atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
             'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
             'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
             'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir',
             'Ce', 'Gd', 'Ga', 'Cs']
    node_featurizer = WeaveAtomFeaturizer(atom_types = atom_types)
    edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
    graph_function = partial(mol_to_bigraph, add_self_loop=True, node_featurizer=args['node_featurizer'], edge_featurizer=args['edge_featurizer'], canonical_atom_order=False)
    accepted_templates, rejected_idxs = load_fixed_templates(args)
    return model, graph_function, accepted_templates
    
def predict(model, rbg, pbg, radm, padm, device):
    rbg, pbg = rbg.to(device), pbg.to(device)
    rnode_feats, pnode_feats = rbg.ndata.pop('h').to(device), pbg.ndata.pop('h').to(device)
    redge_feats, pedge_feats = rbg.edata.pop('e').to(device), pbg.edata.pop('e').to(device)
    return model(rbg, pbg, rnode_feats, pnode_feats, redge_feats, pedge_feats, radm, padm)

def pred_pxr(rxn, model, graph_function, device, verbose = False):
    reactant, product = rxn.split('>>')
    reactant, product = Chem.MolFromSmiles(reactant), Chem.MolFromSmiles(product)
    rgraph, pgraph = graph_function(reactant), graph_function(product)
    radm, padm = [get_adm(reactant)], [get_adm(product)]
    rbg, pbg = dgl.batch([rgraph]),  dgl.batch([pgraph])
    rbg.set_n_initializer(dgl.init.zero_initializer), pbg.set_n_initializer(dgl.init.zero_initializer)
    rbg.set_e_initializer(dgl.init.zero_initializer), pbg.set_e_initializer(dgl.init.zero_initializer)
    
    model.eval()
    with torch.no_grad():
        predicitons = predict(model, rbg, pbg, radm, padm, device)
    
    return predicitons[0]

def get_atom_map(model, graph_function, mapping_setting, rxn):
    preds = []
    reactant, product = rxn.split('>>')
    prediction = pred_pxr(rxn, model, graph_function, mapping_setting['device'])
    prediction = (torch.softmax(prediction, dim = 0)*torch.softmax(prediction, dim = 1)).cpu().numpy()
    mapped_rxn, temp, mapper = prediction2map(rxn, prediction, mapping_setting)
    return {'mapped_rxn': mapped_rxn, 'template': temp, 'mapper': mapper}