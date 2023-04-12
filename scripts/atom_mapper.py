'''
This python script is modified from RXNMapper (Schwaller et al., Sci. Adv. 2021)
https://github.com/rxn4chemistry/rxnmapper/blob/main/rxnmapper/attention.py
'''
import copy
import numpy as np
import pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt
from LocalTemplate.template_extractor import extract_from_reaction

from rdkit import Chem
    
def prediction2map(rxn, prediction, args):
    reactant, product = rxn.split('>>')
    product_mapping = {atom.GetIdx(): atom.GetAtomMapNum() for atom in Chem.MolFromSmiles(product).GetAtoms()}
    if sum(product_mapping.values()) == 0:
        product_mapping = {atom.GetIdx(): atom.GetIdx()+1 for atom in Chem.MolFromSmiles(product).GetAtoms()}
    mapper = AtomMapper(rxn, prediction, product_mapping)
    output = mapper.generate_atom_mapping()
    mapped_rxn = mapper.mapped_rxn
    try:
        template = extract_from_reaction(mapped_rxn)
    except:
        template = None
    return mapped_rxn, template, mapper

class AtomMapper:
    def __init__(
        self,
        rxn_smiles: str,
        predictions: np.ndarray,
        fixed_product_mapping,
        neighbor_weight: float = 10,
        use_atom_mask = True,
        hard_selection = True,
        mask_mapped_product_atoms: bool = True,
        mask_mapped_reactant_atoms: bool = True):
        
        self.reactant, self.product = rxn_smiles.split('>>')
        self.predictions = predictions
        self.fixed_product_mapping = fixed_product_mapping
        self.reactant_mapping = {}
        self.product_mapping = {}
        
        self.neighbor_weight = neighbor_weight
        self.use_atom_mask = use_atom_mask
        self.hard_selection = hard_selection
        self.mask_mapped_product_atoms = mask_mapped_product_atoms
        self.mask_mapped_reactant_atoms = mask_mapped_reactant_atoms
        self.map_steps = None
        
        self.get_adj_matrix()
        self.mask_prediction()
        
        
    def get_adj_matrix(self):
        self.reactant_mol, self.product_mol = Chem.MolFromSmiles(self.reactant), Chem.MolFromSmiles(self.product)
        self.p_adj_matrix = np.array(Chem.GetAdjacencyMatrix(self.product_mol), "bool")
        self.r_adj_matrix = np.array(Chem.GetAdjacencyMatrix(self.reactant_mol), "bool")
        self.p_dist_matrix = np.array(Chem.GetDistanceMatrix(self.product_mol))
        self.r_dist_matrix = np.array(Chem.GetDistanceMatrix(self.reactant_mol))
        return 
    
    def mask_prediction(self):
        self.patoms, self.ratoms = Chem.MolFromSmiles(self.product).GetAtoms(),  Chem.MolFromSmiles(self.reactant).GetAtoms()
        self.n_patoms, self.n_ratoms = len(self.patoms), len(self.ratoms)
        psymbols, rsymbols = [atom.GetSymbol() for atom in self.patoms], [atom.GetSymbol() for atom in self.ratoms]
        self.atom_mask = np.array([[rsymbol == psymbol for rsymbol in rsymbols] for psymbol in psymbols]).astype(int)
        self.neighbor_weight_matrix = np.ones_like(self.atom_mask).astype(float)
        self.combined_masked_att = np.multiply(self.predictions, self.atom_mask)
        return 
    
    def _get_isolate_ringmember(self):
        return [atom.IsInRing() and sum([n.GetIdx() in self.product_mapping for n in atom.GetNeighbors()]) == 0 for atom in self.patoms]
    
    def _is_isolated_carbon(self, atom, mapping):
        return atom.GetSymbol() == 'C' and sum([n.GetIdx() in mapping for n in atom.GetNeighbors()]) == 0

    def _get_isolated_carbons(self):
        reactant_carbons = [self._is_isolated_carbon(atom, self.reactant_mapping) for atom in self.ratoms]
        product_carbons = [self._is_isolated_carbon(atom, self.product_mapping) for atom in self.patoms]
        return reactant_carbons, product_carbons
    
    def _get_normailzed_attntions(self, predictions):
        row_sums = predictions.sum(axis=1)
        normalized_predictions = np.divide(
                                predictions,
                                row_sums[:, np.newaxis],
                                out=np.zeros_like(predictions),
                                where=row_sums[:, np.newaxis] != 0,
                                )
        
        reactant_carbons, product_carbons = self._get_isolated_carbons()
        normalized_predictions[product_carbons, :] *= 0.1
        normalized_predictions[:, reactant_carbons] *= 0.1
        return normalized_predictions
    
    def _get_combined_normalized_predictions(self):
        if self.use_atom_mask:
            combined_predictions = np.multiply(
                self.combined_masked_att, self.neighbor_weight_matrix
            )
        else:
            combined_predictions = np.multiply(
                self.combined_att, self.neighbor_weight_matrix
            )
        return self._get_normailzed_attntions(combined_predictions)

    def generate_atom_mapping(self):
        pxr_mapping_vector = (np.ones(self.n_patoms) * -1).astype(int)
        scores = np.ones(self.n_patoms)
        mapping_steps = []
        for i in range(self.n_patoms):
            normalized_predictions = self._get_combined_normalized_predictions()
            product_atom_to_map = np.argmax(np.max(normalized_predictions, axis=1))
            if self.hard_selection:
                corresponding_reactant_atom = np.argmax(normalized_predictions, axis=1)[product_atom_to_map]
                score = np.max(normalized_predictions, axis=1)[product_atom_to_map]
            else:
                p_row = normalized_predictions[product_atom_to_map] + np.random.rand(normalized_predictions.shape[1])*1e-6
                corresponding_reactant_atom = np.random.choice(list(range(len(p_row))), p = p_row/sum(p_row))
                score = normalized_predictions[product_atom_to_map][corresponding_reactant_atom]
                
            if np.isclose(score, 0.0):
                score = 1.0
                corresponding_reactant_atom = pxr_mapping_vector[product_atom_to_map]  # either -1 or already mapped
                break

            pxr_mapping_vector[product_atom_to_map] = corresponding_reactant_atom
            scores[product_atom_to_map] = score
            
            self._update_neighbor_weight_matrix(product_atom_to_map, corresponding_reactant_atom)
            if self.fixed_product_mapping:
                map_number = self.fixed_product_mapping[product_atom_to_map]
            else:
                map_number = i+1
                
            mapping_steps.append([str(map_number), str(product_atom_to_map), str(corresponding_reactant_atom), round(score, 3)])
            self.product_mapping[int(product_atom_to_map)], self.reactant_mapping[int(corresponding_reactant_atom)] = map_number, map_number
        
        self.map_steps = pd.DataFrame({t[0]: t[1:] for t in mapping_steps}, index=['product_idx', 'reactant_idx', 'score'])
        self.map_on_rxn(self.reactant_mapping, self.product_mapping)

        return
    
    def _update_neighbor_weight_matrix(self, product_atom, reactant_atom):
        if reactant_atom != -1:
            # Neighbor atoms
            neighbors_in_products = self.p_adj_matrix[product_atom]
            neighbors_in_reactants = self.r_adj_matrix[reactant_atom]
            self.neighbor_weight_matrix[np.ix_(neighbors_in_products, neighbors_in_reactants)] *= float(self.neighbor_weight)
            
        if self.mask_mapped_product_atoms:
            self.neighbor_weight_matrix[product_atom] = np.zeros(self.n_ratoms)
        if self.mask_mapped_reactant_atoms:
            self.neighbor_weight_matrix[:,reactant_atom] = np.zeros(self.n_patoms)
        return 
    
    def map_on_rxn(self, reactant_mapping, product_mapping):
        self.mapped_reactant = copy.copy(self.reactant_mol)
        self.mapped_product = copy.copy(self.product_mol)
        [atom.SetAtomMapNum(reactant_mapping[atom.GetIdx()]) if atom.GetIdx() in reactant_mapping else atom.SetAtomMapNum(0) for atom in self.mapped_reactant.GetAtoms()]
        [atom.SetAtomMapNum(product_mapping[atom.GetIdx()])  if atom.GetIdx() in product_mapping else atom.SetAtomMapNum(0) for atom in self.mapped_product.GetAtoms()]
        self.mapped_rxn = '>>'.join(Chem.MolToSmiles(smiles, canonical = False) for smiles in [self.mapped_reactant, self.mapped_product])
        return 
        
    def plot_prediction(self, plot_raw = False):
        plt.plot()
        if plot_raw:
            plt.imshow(self.predictions, cmap = 'coolwarm')
        else:
            plt.imshow(self.combined_masked_att, cmap = 'coolwarm')
        plt.colorbar()
        plt.xlabel('Atom idx in reactant')
        plt.ylabel('Atom idx in product')
        plt.show()
        return