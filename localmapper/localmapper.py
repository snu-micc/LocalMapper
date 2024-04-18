import pickle
import numpy as np
import torch

from pkg_resources import resource_filename
from matplotlib import pyplot as plt

from rdkit import Chem
from rdkit.Chem import Draw

from .utils import *
from .mapper import AtomMapper, prediction2map

def get_rdkit_rxn(rxn):
    def demap(smi):
        mol = Chem.MolFromSmiles(smi)
        [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
        return Chem.MolToSmiles(mol)
    reactant, product = rxn.split('>>')
    rxn = demap(reactant) + '>>' + demap(product)
    rdkit_rxn = Chem.rdChemReactions.ReactionFromSmarts(rxn, useSmiles=True)
    return Chem.Draw.ReactionToImage(rdkit_rxn, subImgSize=(320,700), )

class localmapper:
    def __init__(self, device='cpu', model_version='202403'):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        print ('Loaded LocalMapper (version=%s) at device %s' % (model_version, self.device))
    
        config_path = resource_filename('localmapper', 'data/default_config.json')
        template_path = resource_filename('localmapper', 'data/templates_%s.pkl' % model_version)
        model_path = resource_filename('localmapper', 'data/LocalMapper_%s.pth' % model_version)

        with open(template_path, 'rb') as f:
            self.accepted_templates = pickle.load(f)
        node_featurizer, edge_featurizer, self.graph_function = init_featurizer()
        exp_config = get_configure(config_path, node_featurizer, edge_featurizer)
        self.model = load_model(exp_config, model_path, device)
        self.model.eval()
        
    def pred_pxr(self, rxns):
        rgraphs, pgraphs = [], []
        for rxn in rxns:
            reactant, product = rxn.split('>>')
            reactant, product = Chem.MolFromSmiles(reactant), Chem.MolFromSmiles(product)
            rgraph, pgraph = self.graph_function(reactant), self.graph_function(product)
            rgraphs.append(rgraph)
            pgraphs.append(pgraph)
        predictions = predict(self.model, self.device, rgraphs, pgraphs)
        return [torch.softmax(pred, dim = 1).cpu().numpy() for pred in predictions]

    def get_atom_map(self, rxns):
        if isinstance(rxns, str):
            rxns = [rxns]
        results = {}
        predictions = self.pred_pxr(rxns)
        for rxn, prediction in zip(rxns, predictions):
            outputs, mapped_result = prediction2map(rxn, prediction)
            results[rxn] = mapped_result
            results[rxn]['AAM_matrix'] = outputs
            confident = results[rxn]['template'] in self.accepted_templates
            if not confident:
                outputs, mapped_result = prediction2map(rxn, prediction, 90)
                results[rxn].update(mapped_result)
                confident = results[rxn]['template'] in self.accepted_templates
            results[rxn]['confident'] = confident
        return results
    
    def plot_output(self, outputs):
        plt.plot()
        plt.imshow(outputs, cmap = 'coolwarm')
        plt.colorbar()
        plt.xlabel('Atom index in reactant')
        plt.ylabel('Atom index in product')
        plt.show()
        return
    
    def plot_rxn(self, rxn):
        rdkit_rxn = Chem.rdChemReactions.ReactionFromSmarts(rxn, useSmiles=True)
        return Chem.Draw.ReactionToImage(rdkit_rxn)
    