import glob
import sys
import pandas as pd
from rdkit import Chem

sys.path.append('../')
from LocalTemplate.template_extractor import extract_from_reaction

def get_user_name():
    return glob.glob('*.user')[0].split('\\')[-1].split('.')[0]
    
def load_sampled_data(dataset, chemist_name, samp_iter):
    return pd.read_csv('../data/%s/%s/pred_train_%d.csv' % (dataset, chemist_name, samp_iter))

def save_fixed_data(df, dataset, chemist_name, samp_iter):
    df.to_csv('../data/%s/%s/fixed_train_%d.csv' % (dataset, chemist_name, samp_iter), index = None)
    return 

def load_templates(files, samp_iter):
    loaded_templates = set()
    for i, file in enumerate(files):
        iteration = int(file.split('_')[-1].split('.')[0])
        if iteration > samp_iter:
            continue
        df = pd.read_csv(file)
        loaded_templates.update(df['template'].tolist())
    return loaded_templates
    
def load_fixed_templates(dataset, chemist_name, samp_iter):
    pred_temps = load_templates(glob.glob('../data/%s/%s/pred_train_*.csv' % (dataset, chemist_name)), samp_iter-1)
    accepted_temps = load_templates(glob.glob('../data/%s/%s/fixed_train_*.csv' % (dataset, chemist_name)), samp_iter-1)
    conf_temps = load_templates(glob.glob('../data/%s/%s/conf_pred_*.csv' % (dataset, chemist_name)), samp_iter)
    accepted_temps = accepted_temps.union(conf_temps)
    rejected_temps = set([temp for temp in pred_temps if temp not in accepted_temps])
    return accepted_temps, rejected_temps

def demap(smiles):
    mol = Chem.MolFromSmiles(smiles)
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
    return Chem.MolToSmiles(mol)

def is_valid_mapping(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atom_maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0]
    return len(atom_maps) == len(set(atom_maps))

def save_reaction(rxn, path = 'mol.png'):
    img = Chem.Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in rxn.split('>>')], returnPNG=False, molsPerRow=2, subImgSize=(400, 300))
    path = 'mol.png'
    img.save(path)
    return
    
