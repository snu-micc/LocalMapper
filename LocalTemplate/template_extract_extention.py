'''
This python script is modified from rdchiral template extractor 
https://github.com/connorcoley/rdchiral/blob/master/rdchiral/template_extractor.py
'''
import re
from rdkit import Chem

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
        (range(3), '[OH0,SH0]=C-O',), # carbonyl acid
        (range(2), '[C,N]=[C,N]',), # alkene/imine
        (range(2), '[C,N]#[C,N]',), # alkyne/nitrile
        (range(3), 'O-C-O',), # acetal group
        
        # Acidic carbon/oxygen
        ((2,), '[*]=[*]-[C;X4;!$([CX4][F,Cl,Br,I,OH])]',), # acidic carbon
        ((2,), '[*]#[*]-[C;X4;!$([CX4][F,Cl,Br,I,OH])]',), # acidic carbon
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
        if atom.GetIdx() not in atoms_to_use or atom.GetIsAromatic(): continue
        # Ensure membership of changed atom is checked against group
        for group in groups:
            if int(atom.GetIdx()) in group[0]:
                for idx in group[1]:
                    if idx not in atoms_to_use:
                        n_atom = mol.GetAtomWithIdx(idx)
                        n_atom.SetAtomMapNum(0)
                        new_atoms_to_use.append(idx)
            
    return new_atoms_to_use, symbol_replacements

def expand_atoms_to_use_atom(mol, atoms_to_use, atom_idx, groups=[], symbol_replacements=[]):
    '''Given an RDKit molecule and a list of AtomIdx which should be included
    in the reaction, this function extends the list of atoms_to_use by considering 
    a candidate atom extension, atom_idx'''

    # See if this atom belongs to any special groups (highest priority)
    found_in_group = False
    for group in groups: # first index is atom IDs for match, second is what to include
        if int(atom_idx) in group[0]: # int correction
            # Add the whole list, redundancies don't matter 
            # *but* still call convert_atom_to_wildcard!
            for idx in group[1]:
                if idx not in atoms_to_use:
                    atoms_to_use.append(idx)
            found_in_group = True
    if found_in_group:  
        return atoms_to_use, symbol_replacements

    # Skip current candidate atom if it is already included
    if atom_idx in atoms_to_use:
        return atoms_to_use, symbol_replacements

    # Include this atom
    atoms_to_use.append(atom_idx)

    return atoms_to_use, symbol_replacements