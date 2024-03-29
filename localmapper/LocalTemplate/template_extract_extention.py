'''
This python script is modified from rdchiral template extractor 
https://github.com/connorcoley/rdchiral/blob/master/rdchiral/template_extractor.py
'''
import re
from numpy.random import shuffle
from collections import defaultdict
from pprint import pprint 
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import ChiralType


def get_special_groups(mol):
    '''Given an RDKit molecule, this function returns a list of tuples, where
    each tuple contains the AtomIdx's for a special group of atoms which should 
    be included in a fragment all together. This should only be done for the 
    reactants, otherwise the products might end up with mapping mismatches
    We draw a distinction between atoms in groups that trigger that whole
    group to be included, and "unimportant" atoms in the groups that will not
    be included if another atom matches.'''

    # Define templates
    group_templates = [ 
        # Functional groups
#         (range(2), '[OH0,SH0]=C',), # carbonyl
        (range(2), '[C,O,N]=[C,O,N]',), # alkene/imine
        (range(2), '[C,N]#[C,N]',), # alkyne/nitrile
#         (range(2), '[S,O,N,P]-[S,O,N,P]',), # hetero bond
        (range(3), 'O-C-O',), # acetal group
        (range(3), '[OH0,SH0]=C-O',), # carbonyl acid
        
        # Acidic carbon/oxygen
        ((2,), '[*]=[*]-[C;X4;!$([CX4][F,Cl,Br,I,OH])]',), # acidic carbon
        ((2,), '[*]#[*]-[C;X4;!$([CX4][F,Cl,Br,I,OH])]',), # acidic carbon
        ((1,), 'a-[C;X4;!$([CX4][F,Cl,Br,I,OH])]',), # acidic carbon
        ((2,), 'a-C-O',), # adjacency to aromatic ring
        ((1,), 'a-O',), # adjacency to aromatic ring
        
    ]
    
    # Build list
    groups = []
    for (add_if_match, template) in group_templates:
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(template), useChirality=True)
        for match in matches:
            add_if = []
            for pattern_idx, atom_idx in enumerate(match):
                if pattern_idx in add_if_match:
                    add_if.append(atom_idx)
            groups.append((add_if, match, template))
    return groups

def expand_atoms_to_use(mol, atoms_to_use, groups=[], symbol_replacements=[]):
    '''Given an RDKit molecule and a list of AtomIdX which should be included
    in the reaction, this function expands the list of AtomIdXs to include one 
    nearest neighbor with special consideration of (a) unimportant neighbors and
    (b) important functional groupings'''

    # Copy
    new_atoms_to_use = atoms_to_use[:]
    # Look for all atoms in the current list of atoms to use
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in atoms_to_use: continue
        # Ensure membership of changed atom is checked against group
        for group in groups:
            if int(atom.GetIdx()) in group[0]:
                for idx in group[1]:
                    if idx not in new_atoms_to_use:
                        new_atoms_to_use.append(idx)
    return new_atoms_to_use, symbol_replacements, mol

# def get_strict_smarts_for_special_atom(atom):
#     '''
#     For an RDkit atom object, generate a SMARTS pattern that
#     matches the atom as strictly as possible
#     '''
    
#     symbol = '[%s:%s]' % (atom.GetSymbol(), atom.GetAtomMapNum())
#     if 'H' in symbol and 'Hg' not in symbol:
#         symbol = symbol.replace('H', '')
#     if atom.GetSymbol() == 'H':
#         symbol = '[#1]'
        
#     if '[' not in symbol:
#         symbol = '[' + symbol + ']'
            
#     return symbol