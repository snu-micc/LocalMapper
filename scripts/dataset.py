import os
import errno
import numpy as np
import pandas as pd
import json
import pickle
import glob
from tqdm import tqdm

from rdkit import Chem

import torch
import sklearn
import dgl.backend as F
from dgl.data.utils import save_graphs, load_graphs
            
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_adm(mol, max_distance = 4):
    dm = Chem.GetDistanceMatrix(mol)
    dm[dm > 100] = -1 # remote (different molecule)
    dm[dm > max_distance] = max_distance # remote (same molecule)
    dm[dm == -1] = max_distance + 1
    return torch.LongTensor(dm)

def get_mapping_label(rxn):
    rsmi, psmi = rxn.split('>>')
    rmol = Chem.MolFromSmiles(rsmi)
    pmol = Chem.MolFromSmiles(psmi)
    r_atom_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in rmol.GetAtoms()}
    return [r_atom_dict[atom.GetAtomMapNum()] for atom in pmol.GetAtoms()]

def clean_reactant_map(rxn):
    r, p = rxn.split('>>')
    r_mol = Chem.MolFromSmiles(r)
    [atom.SetAtomMapNum(0) for atom in r_mol.GetAtoms()]
    r = Chem.MolToSmiles(r_mol, canonical = False)
    return '>>'.join([r, p])

class ReactionDataset(object):
    def __init__(self, args, val_rate = 0.1):
        df = pd.read_csv('%s/raw_data.csv' % (args['data_dir']))
        self.rxns = df['mapped_rxn'].tolist()
        self.idxs = [idx for idx in range(len(self.rxns))]
        self.labels = [[] for _ in range(len(self.rxns))]
        self.rgraph_path = '%s/saved_graphs/reactants.bin' % (args['data_dir'])
        self.pgraph_path = '%s/saved_graphs/products.bin' % (args['data_dir'])
        self.radm_path = '%s/saved_graphs/radm.pkl' % (args['data_dir'])
        self.padm_path = '%s/saved_graphs/padm.pkl' % (args['data_dir'])
        
        self._split_data(args, val_rate)
        self._pre_process(args)
        
    def _load_train_conf_rxns(self, args):
        idxs = {'fixed_train': [], 'conf_pred': []}
        for i in range(args['iteration']):
            for train_type in ['fixed_train', 'conf_pred']:
                file_path = '%s/%s/%s_%d.csv' % (args['data_dir'], args['chemist_name'], train_type, i+1)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    idxs[train_type] += df['data_idx'].tolist()
                    if args['mode'] == 'train':
                        for idx, rxn in zip(df['data_idx'], df['mapped_rxn']):
                            self.rxns[idx] = rxn
                            self.labels[idx] = get_mapping_label(rxn)
        return idxs['fixed_train'], idxs['conf_pred']

    def _split_data(self, args, val_rate):
        train_idxs, learned_idxs = self._load_train_conf_rxns(args)
        val_size = int(len(learned_idxs)*val_rate)
        np.random.shuffle(learned_idxs)
        self.val_idx = learned_idxs[:val_size]
        self.train_idx = train_idxs + learned_idxs[val_size:]
        if args['mode'] == 'train' or args['skip']:
            self.test_idx = [idx for idx in self.idxs if idx not in train_idxs+learned_idxs]
        else:
            self.test_idx = self.idxs
        print ('Loaded %d train reaction, %d val reactions, %d test reactions' % (len(self.train_idx), len(self.val_idx), len(self.test_idx)))
        return 

    def _pre_process(self, args):
        if args['mode'] == 'test' and os.path.exists(self.rgraph_path) and os.path.exists(self.pgraph_path):
            print('Loading previously saved dgl graphs...')
            self.rgraphs, _ = load_graphs(self.rgraph_path)
            self.pgraphs, _ = load_graphs(self.pgraph_path)
            self.adms_r = pickle.load(open(self.radm_path,'rb'))
            self.adms_p = pickle.load(open(self.padm_path,'rb'))
        else:
            mkdir_p('%s/saved_graphs/' % args['data_dir'])
            self.mol_to_graph = args['mol_to_graph']
            print('Processing dgl graphs from scratch...')
            self.rgraphs, self.pgraphs, self.adms_r, self.adms_p = [], [], [], []
            for rxn in tqdm(self.rxns, total = len(self.rxns), desc = 'Generating molecule graphs...'):
                r, p = rxn.split('>>')
                r, p = Chem.MolFromSmiles(r), Chem.MolFromSmiles(p)
                self.adms_r.append(get_adm(r))
                self.adms_p.append(get_adm(p))
                self.rgraphs.append(self.mol_to_graph(r))
                self.pgraphs.append(self.mol_to_graph(p))
                
#             save_graphs(self.rgraph_path, self.rgraphs)
#             save_graphs(self.pgraph_path, self.pgraphs)
#             with open(self.radm_path, 'wb') as f1, open(self.padm_path, 'wb') as f2:
#                 pickle.dump(self.adms_r, f1)
#                 pickle.dump(self.adms_p, f2)
        return 

    def __getitem__(self, item):
        return self.idxs[item], self.rxns[item], self.rgraphs[item], self.pgraphs[item], self.adms_r[item], self.adms_p[item], self.labels[item]

    def __len__(self):
            return len(self.rxns)
