import os
import errno
import numpy as np
import pandas as pd
import json
import pickle
import glob
from tqdm import tqdm
from collections import defaultdict

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
    dm[dm > max_distance] = max_distance + 1 # remote (same molecule)
    dm[dm == -1] = max_distance + 2 # remote (different molecule)
    return dm

def product_is_unmapped(rxn):
    r, p = [Chem.MolFromSmiles(smi) for smi in rxn.split('>>')]
    rmaps = [atom.GetAtomMapNum() for atom in r.GetAtoms()]
    pmaps = [atom.GetAtomMapNum() for atom in p.GetAtoms()]
    return 0 in pmaps or sum([m not in rmaps for m in pmaps]) > 0

def get_mapping_label(rxn):
#     print (rxn)
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

def canonicalize_map_rxn(rxn):
    new_rxn = []
    for smi in rxn.split('>>'):
        mol = Chem.MolFromSmiles(smi)
        index2mapnums = {}
        for atom in mol.GetAtoms():
            index2mapnums[atom.GetIdx()] = atom.GetAtomMapNum()
        mol_cano = Chem.RWMol(mol)
        [atom.SetAtomMapNum(0) for atom in mol_cano.GetAtoms()]
        smi_cano = Chem.MolToSmiles(mol_cano)
        mol_cano = Chem.MolFromSmiles(smi_cano)
        matches = mol.GetSubstructMatches(mol_cano)
        if matches:
            for atom, mat in zip(mol_cano.GetAtoms(), matches[0]):
                atom.SetAtomMapNum(index2mapnums[mat])
            smi = Chem.MolToSmiles(mol_cano, canonical=False)
        new_rxn.append(smi)
    return '>>'.join(new_rxn)

class ReactionDataset(object):
    def __init__(self, args):
        df = pd.read_csv('%s/raw_data.csv' % (args['data_dir']))
        self.mode = args['mode']
        self.rxns = df['mapped_rxn'].tolist()
        self.idxs = [idx for idx in range(len(self.rxns))]
        self.labels = [[] for _ in range(len(self.rxns))]
        self.weights = [1]*len(self.rxns)
        self.mol_to_graph = args['mol_to_graph']
        if self.mode == 'train':
            self._load_conf_rxns(args)
            self._make_graphs()
        else:
            self.train_idx, self.val_idx, self.test_idx = [], [], self.idxs
        
        
    def _load_conf_rxns(self, args):
        print ('Preparing AAM labels...')
        train_idx, val_idx = set(), set()
        mapped_rxns = {}
        for i in range(args['iteration']):
            manual_df = pd.read_csv('%s/%s/fixed_train_%d.csv' % (args['data_dir'], args['chemist_name'], i+1))
            for idx, rxn in zip(manual_df['data_idx'], manual_df['mapped_rxn']):
                train_idx.add(idx)
                mapped_rxns[idx] = rxn
                self.weights[idx] = 100
        print ('Load %d reactions from fixed predictions' % len(mapped_rxns))
               
        conf_df = pd.read_csv('%s/%s/conf_pred_%d.csv' % (args['data_dir'], args['chemist_name'], args['iteration']))
        templates_idx = defaultdict(list)
        for i in conf_df.index:
            data_idx = conf_df['data_idx'][i]
            template = conf_df['template'][i]
            rxn = conf_df['mapped_rxn'][i]
            if data_idx not in train_idx and not product_is_unmapped(rxn):
                templates_idx[template].append(i)
                mapped_rxns[data_idx] = rxn
        print ('Load %d templates with total %d reactions from confident predictions' % (len(templates_idx), len(conf_df)))
               
            
        for template, idx_list in templates_idx.items():
            if len(idx_list) > 100:
                sample_size = 100
                add_to_val = True
                template_weight = 1
            else:
                sample_size = len(idx_list)
                add_to_val = False
                template_weight = 100/sample_size
            np.random.shuffle(idx_list)
            sampled_idx_list = idx_list[:sample_size]
            data_idxs = [conf_df['data_idx'][i] for i in sampled_idx_list]
            if add_to_val:
                train_idx.update(data_idxs[:90])
                val_idx.update(data_idxs[90:])
            else:
                train_idx.update(data_idxs)
            for idx in data_idxs:
                self.weights[idx] = template_weight
        print ('Sampled %d train reaction, %d val reactions' % (len(train_idx), len(val_idx)))
               
        for idx, rxn in mapped_rxns.items():
            self.rxns[idx] = rxn
            self.labels[idx] = get_mapping_label(rxn)
        self.train_idx, self.val_idx, self.test_idx = list(train_idx), list(val_idx), []
        return 

    def _make_graphs(self):
        self.rgraphs, self.pgraphs = [], []
        for i, rxn in tqdm(enumerate(self.rxns), total = len(self.rxns), desc = 'Generating molecule graphs...'):
            if i not in self.train_idx+self.val_idx:
                self.rgraphs.append(None)
                self.pgraphs.append(None)
            else:
                r, p = rxn.split('>>')
                r, p = Chem.MolFromSmiles(r), Chem.MolFromSmiles(p)
                self.rgraphs.append(self.mol_to_graph(r))
                self.pgraphs.append(self.mol_to_graph(p))
        return 

    def __getitem__(self, item):
        rxn = self.rxns[item]
        if self.mode == 'train':
            rgraph, pgraph = self.rgraphs[item], self.pgraphs[item]
        else:
            if len(rxn.split('>>')) != 2: # in case of bad reaction
                item = 1
                rxn = self.rxns[item]
            r, p = rxn.split('>>')
            r, p = Chem.MolFromSmiles(r), Chem.MolFromSmiles(p)
            rgraph, pgraph = self.mol_to_graph(r), self.mol_to_graph(p)
        return self.idxs[item], rxn, rgraph, pgraph, self.labels[item], self.weights[item]

    def __len__(self):
            return len(self.rxns)
