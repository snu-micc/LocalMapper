'''
This python script is modified from RXNMapper (Schwaller et al., Sci. Adv. 2021)
https://github.com/rxn4chemistry/rxnmapper/blob/main/rxnmapper/attention.py
'''
import copy
import numpy as np
import pandas as pd
from collections import defaultdict
from rdkit import Chem

from .LocalTemplate.template_extractor import extract_from_reaction
    
def prediction2map(rxn, prediction, neighbor_weight=10):
    reactant, product = rxn.split('>>')
    product_mapping = {atom.GetIdx(): atom.GetAtomMapNum() for atom in Chem.MolFromSmiles(product).GetAtoms()}
    if 0 in product_mapping.values():
        product_mapping = {atom.GetIdx(): atom.GetIdx()+1 for atom in Chem.MolFromSmiles(product).GetAtoms()}
    mapper = AtomMapper(rxn, prediction, product_mapping, neighbor_weight)
    mapped_rxn = mapper.generate_atom_mapping()
    try:
        template = extract_from_reaction(mapped_rxn)
    except:
        template = None
    return {'mapped_rxn': mapped_rxn, 'template': template}
        

class AtomMapper:
    def __init__(
        self,
        rxn_smiles: str,
        predictions: np.ndarray,
        fixed_product_mapping,
        neighbor_weight,
        mask_mapped_product_atoms: bool = True,
        mask_mapped_reactant_atoms: bool = True):
        
        self.reactant, self.product = rxn_smiles.split('>>')
        self.predictions = predictions
        self.fixed_product_mapping = fixed_product_mapping
        self.reactant_mapping = {}
        self.product_mapping = {}
        
        self.neighbor_weight = neighbor_weight
        self.mask_mapped_product_atoms = mask_mapped_product_atoms
        self.mask_mapped_reactant_atoms = mask_mapped_reactant_atoms
        self.map_steps = None
        
        self.mask_prediction()
        self.get_adj_matrix()
        

    def mask_prediction(self):
        self.reactant_mol, self.product_mol = Chem.MolFromSmiles(self.reactant), Chem.MolFromSmiles(self.product)
        self.patoms, self.ratoms = Chem.MolFromSmiles(self.product).GetAtoms(),  Chem.MolFromSmiles(self.reactant).GetAtoms()
        self.n_patoms, self.n_ratoms = len(self.patoms), len(self.ratoms)
        psymbols, rsymbols = [atom.GetSymbol() for atom in self.patoms], [atom.GetSymbol() for atom in self.ratoms]
        self.atom_mask = np.array([[rsymbol == psymbol for rsymbol in rsymbols] for psymbol in psymbols]).astype(int)
        self.masked_predictions = np.multiply(self.predictions, self.atom_mask)
        self.neighbor_weight_matrix = np.ones_like(self.atom_mask).astype(float)
        return 
    
    def get_adj_matrix(self):
        self.p_adj_matrix = np.array(Chem.GetAdjacencyMatrix(self.product_mol), "bool")
        self.r_adj_matrix = np.array(Chem.GetAdjacencyMatrix(self.reactant_mol), "bool")
        self.isolated_carbons = np.zeros(self.n_ratoms, dtype=bool)
        self.isolated_carbons[[atom.GetSymbol() == 'C' for atom in self.ratoms]] = True
        
        self.r_mol_matrix = np.array(Chem.GetDistanceMatrix(self.reactant_mol)) < 100
        self.r_mol_matrix[[atom.GetSymbol() == 'C' for atom in self.ratoms]] = False
        self.mapped_molecules = np.zeros(self.n_ratoms, dtype=bool)
        return 

    def _get_normailzed_predictions(self, predictions):
        
        row_sums = predictions.sum(axis=1)
        column_sums = predictions.sum(axis=0)
        row_normalized = np.divide(
            predictions, row_sums[:, np.newaxis], 
            out=np.zeros_like(predictions), where=row_sums[:, np.newaxis] != 0
        )
        column_normalized = np.divide(
            predictions, column_sums[np.newaxis, :],
            out=np.zeros_like(predictions), where=column_sums[np.newaxis, :] != 0
        )
        normalized_predictions = row_normalized*column_normalized
        
        # prevent hopping AAM
        low_score = np.max(predictions, axis=0) < 0.5
        if self.neighbor_weight < 90:
            normalized_predictions[:, low_score+self.isolated_carbons] *= 0.01
        else:
            normalized_predictions[:, low_score] *= 0.01
        
        # in case of multiple reagents
        normalized_predictions[:, self.mapped_molecules] *= 0.99
        
        return normalized_predictions
    
    def _get_combined_normalized_predictions(self):
        combined_predictions = np.multiply(self.masked_predictions, self.neighbor_weight_matrix)
        return self._get_normailzed_predictions(combined_predictions)

    def generate_atom_mapping(self):
        pxr_mapping_vector = (np.ones(self.n_patoms) * -1).astype(int)
        scores = np.ones(self.n_patoms)
        mapping_steps = []
        for i in range(self.n_patoms):
            normalized_predictions = self._get_combined_normalized_predictions()
            product_atom_to_map = np.argmax(np.max(normalized_predictions, axis=1))
            corresponding_reactant_atom = np.argmax(normalized_predictions, axis=1)[product_atom_to_map]
            score = np.max(normalized_predictions, axis=1)[product_atom_to_map]
                
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
        mapped_rxn = self.map_on_rxn(self.reactant_mapping, self.product_mapping)

        return mapped_rxn
    
    def _update_neighbor_weight_matrix(self, product_atom, reactant_atom):
        if reactant_atom != -1:
            # Neighbor atoms
            neighbors_in_products = self.p_adj_matrix[product_atom]
            neighbors_in_reactants = self.r_adj_matrix[reactant_atom]
            self.neighbor_weight_matrix[np.ix_(neighbors_in_products, neighbors_in_reactants)] *= float(self.neighbor_weight)
            
            self.isolated_carbons[reactant_atom] = False
            self.isolated_carbons[neighbors_in_reactants] = False
            # Mapped molecules
            self.mapped_molecules[self.r_mol_matrix[reactant_atom]] = True
            
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
        
        return '>>'.join(Chem.MolToSmiles(smiles, canonical = False) for smiles in [self.mapped_reactant, self.mapped_product])
