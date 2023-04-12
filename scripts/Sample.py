from argparse import ArgumentParser
from collections import defaultdict
import glob
import os, sys
import numpy as np
import pandas as pd

from utils import get_user_name, mkdir_p

def reject_template(template): # decomposition reaction without reagent
    return not isinstance(template, str) or template == 'mapped' or template == 'None'
                
def load_raw_data(args, fixed = False):
    if fixed:
        df = pd.read_csv('%s/fixed_data.csv' % args['data_dir'])
    else:
        df = pd.read_csv('%s/raw_data.csv' % args['data_dir'])
    trues = []
    for i, (rxn, temp) in enumerate(zip(df['mapped_rxn'], df['template'])):
        trues.append([rxn, temp])
    return trues

def load_train_data(args):
    train_rxns = []
    train_temps = []
    for i, file in enumerate(glob.glob('%s/fixed_train_*.csv' % args['sample_dir'])):
        if i >= args['iteration']:
            break
        df = pd.read_csv(file)
        train_rxns += df['mapped_rxn'].tolist()
        train_temps += df['template'].tolist()
    return train_rxns, train_temps

def load_prediction(args, load_prev = False, skip = False):
    predictions = []
    if load_prev:
        file = '%s/pred_%d.txt' % (args['output_dir'], args['iteration']-1)
    else:
        if not skip:
            file = '%s/pred_%d_full.txt' % (args['output_dir'], args['iteration'])
        else:
            file = '%s/pred_%d_full.txt' % (args['output_dir'], args['iteration'])
    with open(file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            predictions.append(line.split('\n')[0].split('\t'))
    return predictions

def load_templates(args, files, load_prev):
    loaded_templates = dict()
    for i, file in enumerate(files):
        iteration = int(file.split('_')[-1].split('.')[0])
        if iteration > args['iteration'] or (load_prev and iteration == args['iteration']):
            continue
        df = pd.read_csv(file)
        for idx, template in zip(df.data_idx, df.template):
            loaded_templates[idx] = template
    return loaded_templates
    
def load_fixed_templates(args, load_prev = False):
    pred_dict = load_templates(args, glob.glob('%s/pred_train_*.csv' % args['sample_dir']), load_prev)
    accepted_dict = load_templates(args, glob.glob('%s/fixed_train_*.csv' % args['sample_dir']), load_prev)
    rejected_idxs = [idx for idx in pred_dict if idx not in accepted_dict]
    accepted_templates = set(accepted_dict.values())
    return accepted_templates, rejected_idxs

def sample_reactions(args):
    trues = load_raw_data(args)
    if os.path.exists(args['sample_dir']) and os.path.exists('%s/fixed_train_%d.csv' % (args['sample_dir'], args['iteration']-1)):
        if os.path.exists('%s/fixed_train_%d.csv' % (args['sample_dir'], args['iteration'])):
            print ('Train data for iteration %d is already sampled and fixed.' % args['iteration'])
            return
        elif len(glob.glob('%s/fixed_train_*.csv' % args['sample_dir'])) > 0:
            predictions = load_prediction(args, True)
            accepted_templates, rejected_idxs= load_fixed_templates(args, True)
            new_templates = defaultdict(list)
            conf_idxs, conf_rxns, conf_temps = [], [], []
            for prediction in predictions:
                idx, mapped_rxn, temp = prediction
                if reject_template(temp) or temp in rejected_idxs:
                    continue
                elif temp in accepted_templates:
                    conf_idxs.append(int(idx))
                    conf_rxns.append(mapped_rxn)
                    conf_temps.append(temp)
                else:
                    new_templates[temp].append(int(idx))
                    
            sampled_rxn_n = 9e9
            template_freq = 0
            while sampled_rxn_n > args['sample_limit']:
                template_freq += 1
                sampled_idxs = []
                for template, rxn_idxs in new_templates.items():
                    n_rxn = len(rxn_idxs)
                    if n_rxn >= template_freq:
                        sampled_idx = np.random.choice(rxn_idxs, min([n_rxn, args['sample_n']]), replace=False)
                        sampled_idxs += list(sampled_idx)
                sampled_rxn_n = len(sampled_idxs)
            
            template_freq -= 1
            sampled_idxs = []
            for template, rxn_idxs in new_templates.items():
                n_rxn = len(rxn_idxs)
                if n_rxn >= template_freq:
                    sampled_idx = np.random.choice(rxn_idxs, min([n_rxn, args['sample_n']]), replace=False)
                    sampled_idxs += list(sampled_idx)                    
            
            sampled_idxs = list(np.random.choice(sampled_idxs, args['sample_limit'], replace=False))
            
            sampled_rxns = [trues[i][0] for i in sampled_idxs]
            sampled_temps = [trues[i][1] for i in sampled_idxs]
            
            print ('Sampled %d reactions showing rxns >= %d times' % (len(sampled_idxs), template_freq))
            
            
    else:
        mkdir_p(args['sample_dir'])
        conf_idxs, conf_rxns, conf_temps = [], [], []
        sampled_idxs = list(np.random.choice(np.arange(len(trues)), args['sample_limit'], replace=False))
        sampled_rxns = [trues[i][0] for i in sampled_idxs]
        sampled_temps = [trues[i][1] for i in sampled_idxs]
        print ('Sampled %d random rxns' % len(sampled_idxs))
    
    conf_df = pd.DataFrame({'data_idx': conf_idxs, 'mapped_rxn': conf_rxns, 'template': conf_temps} )
    conf_df.to_csv('%s/conf_pred_%d.csv' % (args['sample_dir'], args['iteration']), index = None)
    sample_df = pd.DataFrame({'data_idx': sampled_idxs, 'mapped_rxn': sampled_rxns, 'template': sampled_temps})
    sample_df.to_csv('%s/pred_train_%d.csv' % (args['sample_dir'], args['iteration']), index = None)
    return 

def main(args):
    args['chemist_name'] = get_user_name(args)
    print ('Sampling... chemist name: %s' % (args['chemist_name']))
    
    args['data_dir'] = '../data/%s/' % args['dataset']
    args['sample_dir'] = '%s/%s' % (args['data_dir'], args['chemist_name'])
    args['output_dir'] = '../outputs/%s/%s/' % (args['dataset'], args['chemist_name'])
    sample_reactions(args)
    
    
if __name__ == '__main__':
    parser = ArgumentParser('LocalMapper testing arguements')
    parser.add_argument('-d', '--dataset', default='USPTO_50K', help='Dataset to use')
    parser.add_argument('-i', '--iteration', type=int, default=1, help='Iteration of active learning')
    parser.add_argument('-s', '--skip', type=int, default=0, help='Skip confidence prediction or not')
    parser.add_argument('-e', '--epsilon', type=float, default=0.05, help='The ratio of sampling low prediction score reaction')
    parser.add_argument('-sn', '--sample-n', type=int, default=1, help='Number of reaction sampled for each template')
    parser.add_argument('-sl', '--sample-limit', type=int, default=200, help='The limit of sampling')
    
    args = parser.parse_args().__dict__
    main(args)