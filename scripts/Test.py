from argparse import ArgumentParser
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import sys

sys.path.append('../')

import torch
import sklearn
import torch.nn as nn

from Sample import load_fixed_templates
from utils import mkdir_p, get_user_name, init_featurizer, get_configure, load_model, load_dataloader, predict
from atom_mapper import AtomMapper, prediction2map

def get_atom_map(args, model, data_loader):
    model.eval()
    file_path = args['output_dir']+'pred_%s.txt' % (args['iteration'])
    accepted_templates, _ = load_fixed_templates(args)
    with open(file_path, 'w') as f:
        f.write('Reaction_id\tMapped_reaction\tTemplate\n')
        with torch.no_grad():
            for batch_data in tqdm(data_loader, total = len(data_loader), desc = 'Predicting AAM...'):
                idxs, rxns, rbg, pbg, _, _, _ = batch_data
                logits_list = predict(args, model, rbg, pbg)
                for rxn, logits, idx in zip(rxns, logits_list, idxs):
                    prediction = torch.softmax(logits, dim = 1).cpu().numpy()
                    result = prediction2map(rxn, prediction)
                    if args['try_twice'] and result['template'] not in accepted_templates:
                        result = prediction2map(rxn, prediction, neighbor_weight=90)
                    f.write('%s\t%s\t%s\n' % (idx, result['mapped_rxn'], result['template']))
    return

def main(args):
    args['mode'] = 'test'
    args['chemist_name'] = get_user_name(args)
    args['device'] = torch.device(args['gpu']) if torch.cuda.is_available() else torch.device('cpu')
    print ('Testing with device %s, chemist name %s' % (args['device'], args['chemist_name']))
    
    args['data_dir'] = '../data/%s/' % args['dataset']
    args['output_dir'] = '../outputs/%s/%s/' % (args['dataset'], args['chemist_name'])
    args['model_path'] =  '../models/%s/%s/LocalMapper_%d.pth' % (args['dataset'], args['chemist_name'], args['iteration'])
    args['config_path'] = '../data/configs/%s' % args['config']
    mkdir_p(args['output_dir'])
    
    args = init_featurizer(args)
    test_loader = load_dataloader(args, test = True)
    model = load_model(args)
    get_atom_map(args, model, test_loader)
    
    
if __name__ == '__main__':
    parser = ArgumentParser('LocalMapper testing arguements')
    parser.add_argument('-g', '--gpu', default='cuda:0', help='GPU device to use')
    parser.add_argument('-c', '--config', default='default_config.json', help='Configuration of model')
    parser.add_argument('-b', '--batch-size', default=20, help='Batch size of dataloader')     
    parser.add_argument('-i', '--iteration', type=int, default=1, help='Iteration of active learning')
    parser.add_argument('-d', '--dataset', default='USPTO_50K', help='Dataset to predict')
    parser.add_argument('-tt', '--try-twice', type=bool, default=False, help='Try AAM twice')
    
    args = parser.parse_args().__dict__
    main(args)