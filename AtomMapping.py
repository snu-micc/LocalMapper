import json
import copy
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

from time import time

atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
                 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
                 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
                 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir',
                 'Ce', 'Gd', 'Ga', 'Cs']

def pad_atom_distance_matrix(adm_list):
    max_size = max([adm.shape[0] for adm in adm_list])
    adm_list = [torch.LongTensor(np.pad(adm, (0, max_size - adm.shape[0]), 'maximum')).unsqueeze(0) for adm in adm_list]
    return torch.cat(adm_list, dim = 0) 

class AtomMapper:
    def __init__(self, device, dataset='USPTO_FULL', chemist_name='pretrained', iteration=3):
        data_dir = 'data/%s/' % dataset
        sample_dir = '%s/%s' % (data_dir, chemist_name)
        config_path = 'data/configs/default_config.json'
        model_path = 'models/%s/%s/LocalMapper.pth' % (dataset, chemist_name)
        
        args = {'mode': 'test', 
                'iteration': iteration, 
                'data_dir': data_dir, 
                'sample_dir': sample_dir, 
                'config_path': config_path, 
                'model_path': model_path, 
                'device': device
               }
        
        args = init_featurizer(args)
        node_featurizer = WeaveAtomFeaturizer(atom_types = atom_types)
        edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
        
        self.device = device
        self.model = load_model(args)
        self.model.eval()
        self.graph_function = partial(mol_to_bigraph, add_self_loop=True, node_featurizer=args['node_featurizer'], edge_featurizer=args['edge_featurizer'], canonical_atom_order=False)
        self.accepted_templates, self.rejected_idxs = load_fixed_templates(args)
            
    def pred_pxr(self, rxn):
        tos = time()
        reactant, product = rxn.split('>>')
        reactant, product = Chem.MolFromSmiles(reactant), Chem.MolFromSmiles(product)
        rgraph, pgraph = self.graph_function(reactant), self.graph_function(product)
        
        rbg, pbg = dgl.batch([rgraph]),  dgl.batch([pgraph])
        rbg.set_n_initializer(dgl.init.zero_initializer), pbg.set_n_initializer(dgl.init.zero_initializer)
        rbg.set_e_initializer(dgl.init.zero_initializer), pbg.set_e_initializer(dgl.init.zero_initializer)
        rbg, pbg = rbg.to(self.device), pbg.to(self.device)
        rnode_feats, pnode_feats = rbg.ndata.pop('h').to(self.device), pbg.ndata.pop('h').to(self.device)
        redge_feats, pedge_feats = rbg.edata.pop('e').to(self.device), pbg.edata.pop('e').to(self.device)
        use_time = time()-tos
        with torch.no_grad():
            predicitons = self.model(rbg, pbg, rnode_feats, pnode_feats, redge_feats, pedge_feats)
        return predicitons[0]

    def get_atom_map(self, rxn, fix_product_mapping=False, return_confidence=False, neighbor_weight=10):
        reactant, product = rxn.split('>>')
        prediction = self.pred_pxr(rxn)
        prediction = torch.softmax(prediction, dim = 1).cpu().numpy()
        result = prediction2map(rxn, prediction, fix_product_mapping, True, neighbor_weight)
        if return_confidence:
            result['confidence'] = result['template'] in self.accepted_templates
            return result
        else:
            return result
    