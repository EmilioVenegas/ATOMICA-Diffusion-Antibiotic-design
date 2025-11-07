import numpy as np
from rdkit import Chem
import torch

# ------------------------------------------------------------------------------
# Computational
# ------------------------------------------------------------------------------
FLOAT_TYPE = torch.float32
INT_TYPE = torch.int64

# --- START: ATOMICA Vocabulary Definition ---
# Source: mims-harvard/atomica/ATOMICA-0f9ae56c23f59920b44caac28a7762f535c3c15d/data/pdb_utils.py

# ATOMICA Atom Vocabulary for A field (atoms)
ATOMS = [ # Periodic Table
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og'
]
# ATOMICA's atom_decoder includes special tokens 'p', 'm', 'g' at the start
atomica_atom_decoder = ['p', 'm', 'g'] + ATOMS
atomica_atom_encoder = {atom: i for i, atom in enumerate(atomica_atom_decoder)}

# --- ATOMICA Block Definitions (from ATOMICA/data/pdb_utils.py) ---
atomica_block_vocab = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'NAG', 'ADP', 'FAD', 'NAD', 'NDP', 'ATP', 'GTP', 'FMN', 'HEM', 'MAN',
    'GLC', 'BGC', 'GDP', 'SAH', 'SAM', 'AMP', 'COA', 'ACT', 'NAP', 'GNP',
    'HES', 'SO4', 'PO4', 'EDO', 'GOL', 'CL', 'MG', 'CA', 'MN', 'ZN',
    'FE', 'FE2', 'NA', 'K', 'IOD', 'ACY', 'MES', 'NO3', 'PCA', 'NH4',
    'EPE', 'MLI', 'DMS', 'CIT', 'BME', 'MPD', 'PGE', 'FUC', 'PEG', 'PG4',
    'PLP', 'PLM', 'PCW', 'PMP', 'SF4', 'LIG', 'UNK'
]
atomica_block_encoder = {j: i for i, j in enumerate(atomica_block_vocab)}
atomica_block_decoder = atomica_block_vocab

# Placeholder histogram for atomica blocks
atomica_aa_hist = np.ones(len(atomica_block_vocab))
atomica_aa_hist = atomica_aa_hist / atomica_aa_hist.sum()



#  Helper function to build new bond matrices
def build_matrix(raw_dict, decoder, default_val=0.0):
    n = len(decoder)
    matrix = [[default_val] * n for _ in range(n)]
    for i in range(n):
        atom1 = decoder[i]
        if atom1 not in raw_dict:
            continue
        for j in range(i, n): 
            atom2 = decoder[j]
            
            val = raw_dict.get(atom1, {}).get(atom2)
            if val is None:
                val = raw_dict.get(atom2, {}).get(atom1)
            
            if val is not None:
                matrix[i][j] = val
                matrix[j][i] = val # Matrix is symmetric
    return matrix



# ------------------------------------------------------------------------------
# Bond parameters
# ------------------------------------------------------------------------------

# margin1, margin2, margin3 = 10, 5, 3
margin1, margin2, margin3 = 3, 2, 1

allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'B': 3, 'Al': 3,
                 'Si': 4, 'P': [3, 5],
                 'S': 4, 'Cl': 1, 'As': 3, 'Br': 1, 'I': 1, 'Hg': [1, 2],
                 'Bi': [3, 5]}

# Bond lengths from:
# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
# And:
# http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
bonds1 = {'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
                'B': 119, 'Si': 148, 'P': 144, 'As': 152, 'S': 134,
                'Cl': 127, 'Br': 141, 'I': 161},
          'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
                'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194,
                'I': 214},
          'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
                'Cl': 175, 'Br': 214, 'S': 168, 'I': 222, 'P': 177},
          'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
                'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164,
                'I': 194},
          'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
                'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156,
                'I': 187},
          'B': {'H':  119, 'Cl': 175},
          'Si': {'Si': 233, 'H': 148, 'C': 185, 'O': 163, 'S': 200,
                 'F': 160, 'Cl': 202, 'Br': 215, 'I': 243 },
          'Cl': {'Cl': 199, 'H': 127, 'C': 177, 'N': 175, 'O': 164,
                 'P': 203, 'S': 207, 'B': 175, 'Si': 202, 'F': 166,
                 'Br': 214},
          'S': {'H': 134, 'C': 182, 'N': 168, 'O': 151, 'S': 204,
                'F': 158, 'Cl': 207, 'Br': 225, 'Si': 200, 'P': 210,
                'I': 234},
          'Br': {'Br': 228, 'H': 141, 'C': 194, 'O': 172, 'N': 214,
                 'Si': 215, 'S': 225, 'F': 178, 'Cl': 214, 'P': 222},
          'P': {'P': 221, 'H': 144, 'C': 184, 'O': 163, 'Cl': 203,
                'S': 210, 'F': 156, 'N': 177, 'Br': 222},
          'I': {'H': 161, 'C': 214, 'Si': 243, 'N': 222, 'O': 194,
                'S': 234, 'F': 187, 'I': 266},
          'As': {'H': 152}
          }

bonds2 = {'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
          'N': {'C': 129, 'N': 125, 'O': 121},
          'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150},
          'P': {'O': 150, 'S': 186},
          'S': {'P': 186, 'C': 160}}


bonds3 = {'C': {'C': 120, 'N': 116, 'O': 113},
          'N': {'C': 116, 'N': 110},
          'O': {'C': 113}}

bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
             Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

# https://en.wikipedia.org/wiki/Covalent_radius#Radii_for_multiple_bonds
# (2022/08/14)
covalent_radii = {'H': 32, 'C': 60, 'N': 54, 'O': 53, 'F': 53, 'B': 73,
                  'Al': 111, 'Si': 102, 'P': 94, 'S': 94, 'Cl': 93, 'As': 106,
                  'Br': 109, 'I': 125, 'Hg': 133, 'Bi': 135}

# ------------------------------------------------------------------------------

# Generate the new ATOMICA-compatible matrices
atomica_bonds1_matrix = build_matrix(bonds1, atomica_atom_decoder)
atomica_bonds2_matrix = build_matrix(bonds2, atomica_atom_decoder)
atomica_bonds3_matrix = build_matrix(bonds3, atomica_atom_decoder)

atomica_lennard_jones_matrix = atomica_bonds1_matrix


# Backbone geometry
# Taken from: Bhagavan, N. V., and C. E. Ha.
# "Chapter 4-Three-dimensional structure of proteins and disorders of protein misfolding."
# Essentials of Medical Biochemistry (2015): 31-51.
# https://www.sciencedirect.com/science/article/pii/B978012416687500004X
# ------------------------------------------------------------------------------
N_CA_DIST = 1.47
CA_C_DIST = 1.53
N_CA_C_ANGLE = 110 * np.pi / 180

# --- START: NEW DRUG-LIKE VOCABULARY DEFINITION ---

# 1. Define the small set of atoms we want our model to learn about.
DRUGLIKE_ATOMS_DECODER = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
DRUGLIKE_ATOMS_ENCODER = {atom: i for i, atom in enumerate(DRUGLIKE_ATOMS_DECODER)}
DRUGLIKE_VOCAB_SIZE = len(DRUGLIKE_ATOMS_DECODER)

# 2. Create a mapping from the large ATOMICA index to our new small index.
#    This is used by process_atomica_jsonl.py
ATOMICA_TO_DRUGLIKE_MAP = np.full(len(atomica_atom_decoder), -1, dtype=int)
for atomica_idx, atom_symbol in enumerate(atomica_atom_decoder):
    if atom_symbol in DRUGLIKE_ATOMS_ENCODER:
        druglike_idx = DRUGLIKE_ATOMS_ENCODER[atom_symbol]
        ATOMICA_TO_DRUGLIKE_MAP[atomica_idx] = druglike_idx

# --- Programmatically generate data for the *NEW* 9-atom dataset params ---

# 1. Define colors for our 9 atoms
atom_colors = {}
atom_colors['C'] = '#33ff33'  # Green
atom_colors['N'] = '#3333ff'  # Blue
atom_colors['O'] = '#ff4d4d'  # Red
atom_colors['S'] = '#e6c540'  # Yellow
atom_colors['P'] = '#ff8000'  # Orange
atom_colors['F'] = '#B3FFFF'  # Light Blue
atom_colors['Cl'] = "#0D700D" # Bright Green
atom_colors['Br'] = '#A62929' # Dark Red
atom_colors['I'] = '#940094'  # Purple
atom_colors['H'] = '#FFFFFF'  # White (not in list, but good to have)
DEFAULT_COLOR = '#ffb5b5'     # Pink (for any missing)

# 2. Create color, radius, and histogram lists *for the 9-atom vocab*
druglike_colors_list = []
druglike_radius_list = []
druglike_atom_hist_array = np.array([0.58447313, 0.09319977, 0.28838374, 0.00662685, 0.02323252, 0.00200347, 0.00138702, 0.00023117, 0.00046234])
for i, atom_symbol in enumerate(DRUGLIKE_ATOMS_DECODER):
    druglike_colors_list.append(atom_colors.get(atom_symbol, DEFAULT_COLOR))
    # Get radius in pm, default to 30pm, convert to Angstrom for plotting
    radius_in_pm = covalent_radii.get(atom_symbol, 30) # e.g., 60
    radius_in_angstroms = radius_in_pm / 100.0         # e.g., 0.6
    druglike_radius_list.append(radius_in_angstroms)
# 3. Create the new 9x9 bond matrices
#    (This uses your 'build_matrix' helper function and 'bonds1' raw dict)
druglike_bonds1_matrix = build_matrix(bonds1, DRUGLIKE_ATOMS_DECODER)
druglike_bonds2_matrix = build_matrix(bonds2, DRUGLIKE_ATOMS_DECODER)
druglike_bonds3_matrix = build_matrix(bonds3, DRUGLIKE_ATOMS_DECODER)
# Create the Lennard-Jones r_m matrix based on the sum of covalent radii.
# This approximates the van der Waals minimum energy distance (r_m) for 
# non-bonded pairs, which is what the LJ potential is for.
# This prevents the "atom clumping" bug if auxiliary_loss is turned on.

n_atoms_druglike = len(DRUGLIKE_ATOMS_DECODER)
druglike_lennard_jones_matrix = np.zeros((n_atoms_druglike, n_atoms_druglike))

for i in range(n_atoms_druglike):
    atom_i = DRUGLIKE_ATOMS_DECODER[i]
    # Get the radius for atom i from the covalent_radii dict
    # Default to 60 (Carbon) if somehow missing
    radius_i = covalent_radii.get(atom_i, 60) 
    
    for j in range(i, n_atoms_druglike): # Start from i to fill the symmetric matrix
        atom_j = DRUGLIKE_ATOMS_DECODER[j]
        radius_j = covalent_radii.get(atom_j, 60)
        
        # The minimum energy distance (r_m) is the sum of their radii
        rm_ij = radius_i + radius_j
        
        druglike_lennard_jones_matrix[i, j] = rm_ij
        druglike_lennard_jones_matrix[j, i] = rm_ij # Make symmetric



# ------------------------------------------------------------------------------
# Dataset-specific constants
# ------------------------------------------------------------------------------
dataset_params = {}

# --- THIS IS THE CORRECTED DEFINITION ---
dataset_params['atomica_PL'] = {
    # Use the 9-atom vocab
    'atom_encoder': DRUGLIKE_ATOMS_ENCODER,
    'atom_decoder': DRUGLIKE_ATOMS_DECODER,
    
    # Use the 9-element lists
    'colors_dic': druglike_colors_list,
    'radius_dic': druglike_radius_list,
    'atom_hist': druglike_atom_hist_array,
          
    # Use the 9x9 matrices
    'bonds1': druglike_bonds1_matrix,
    'bonds2': druglike_bonds2_matrix,
    'bonds3': druglike_bonds3_matrix,
    'lennard_jones_rm': druglike_lennard_jones_matrix,

    # These are for the pocket (receptor) and can stay the same
    'aa_encoder': atomica_block_encoder,
    'aa_decoder': atomica_block_decoder,
    'aa_hist': atomica_aa_hist,
}

dataset_params['bindingmoad'] = {
    'atom_encoder': {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9},
    'atom_decoder': ['C', 'N', 'O', 'S', 'B', 'Br', 'Cl', 'P', 'I', 'F'],
    'aa_encoder': {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19},
    'aa_decoder': ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
    # PyMOL colors, see: https://pymolwiki.org/index.php/Color_Values#Chemical_element_colours
    'colors_dic': ['#33ff33', '#3333ff', '#ff4d4d', '#e6c540', '#ffb5b5', '#A62929', '#1FF01F', '#ff8000', '#940094', '#B3FFFF', '#b3e3f5'],
    'radius_dic': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    'bonds1': [
        [154.0, 147.0, 143.0, 182.0, 0.0, 194.0, 177.0, 184.0, 214.0, 135.0],
        [147.0, 145.0, 140.0, 168.0, 0.0, 214.0, 175.0, 177.0, 222.0, 136.0],
        [143.0, 140.0, 148.0, 151.0, 0.0, 172.0, 164.0, 163.0, 194.0, 142.0],
        [182.0, 168.0, 151.0, 204.0, 0.0, 225.0, 207.0, 210.0, 234.0, 158.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 175.0, 0.0, 0.0, 0.0],
        [194.0, 214.0, 172.0, 225.0, 0.0, 228.0, 214.0, 222.0, 0.0, 178.0],
        [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 0.0, 166.0],
        [184.0, 177.0, 163.0, 210.0, 0.0, 222.0, 203.0, 221.0, 0.0, 156.0],
        [214.0, 222.0, 194.0, 234.0, 0.0, 0.0, 0.0, 0.0, 266.0, 187.0],
        [135.0, 136.0, 142.0, 158.0, 0.0, 178.0, 166.0, 156.0, 187.0, 142.0]],
    'bonds2': [[134.0, 129.0, 120.0, 160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [129.0, 125.0, 121.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [120.0, 121.0, 121.0, 0.0, 0.0, 0.0, 0.0, 150.0, 0.0, 0.0],
               [160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 186.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 150.0, 186.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    'bonds3': [[120.0, 116.0, 113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [116.0, 110.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    'lennard_jones_rm': [
        [120.0, 116.0, 113.0, 160.0, 133.0, 194.0, 177.0, 184.0, 214.0, 135.0],
        [116.0, 110.0, 121.0, 168.0, 127.0, 214.0, 175.0, 177.0, 222.0, 136.0],
        [113.0, 121.0, 121.0, 151.0, 126.0, 172.0, 164.0, 150.0, 194.0, 142.0],
        [160.0, 168.0, 151.0, 204.0, 167.0, 225.0, 207.0, 186.0, 234.0, 158.0],
        [133.0, 127.0, 126.0, 167.0, 146.0, 182.0, 175.0, 167.0, 198.0, 126.0],
        [194.0, 214.0, 172.0, 225.0, 182.0, 228.0, 214.0, 222.0, 234.0, 178.0],
        [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 218.0, 166.0],
        [184.0, 177.0, 150.0, 186.0, 167.0, 222.0, 203.0, 221.0, 219.0, 156.0],
        [214.0, 222.0, 194.0, 234.0, 198.0, 234.0, 218.0, 219.0, 266.0, 187.0],
        [135.0, 136.0, 142.0, 158.0, 126.0, 178.0, 166.0, 156.0, 187.0, 142.0]],
    'atom_hist': {'C': 545542, 'N': 90205, 'O': 132965, 'S': 9342, 'B': 109,
                  'Br': 1424, 'Cl': 5516, 'P': 5154, 'I': 445, 'F': 9742},
    'aa_hist': {'A': 109798, 'C': 31556, 'D': 83921, 'E': 79405, 'F': 97083,
                'G': 139319, 'H': 62661, 'I': 99008, 'K': 62403, 'L': 155105,
                'M': 59977, 'N': 70437, 'P': 58833, 'Q': 48254, 'R': 74215,
                'S': 103286, 'T': 90972, 'V': 119954, 'W': 42017, 'Y': 90596},
}

dataset_params['crossdock_full'] = {
      'atom_encoder': {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9, 'others': 10},
      'atom_decoder': ['C', 'N', 'O', 'S', 'B', 'Br', 'Cl', 'P', 'I', 'F', 'others'],
      'aa_encoder': {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9, 'others': 10},
      'aa_decoder': ['C', 'N', 'O', 'S', 'B', 'Br', 'Cl', 'P', 'I', 'F', 'others'],
      'colors_dic': ['#33ff33', '#3333ff', '#ff4d4d', '#e6c540', '#ffb5b5', '#A62929', '#1FF01F', '#ff8000', '#940094', '#B3FFFF', '#ffb5b5'],
      'radius_dic': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
      'bonds1': [[154.0, 147.0, 143.0, 182.0, 0.0, 194.0, 177.0, 184.0, 214.0, 135.0, 0.0], [147.0, 145.0, 140.0, 168.0, 0.0, 214.0, 175.0, 177.0, 222.0, 136.0, 0.0], [143.0, 140.0, 148.0, 151.0, 0.0, 172.0, 164.0, 163.0, 194.0, 142.0, 0.0], [182.0, 168.0, 151.0, 204.0, 0.0, 225.0, 207.0, 210.0, 234.0, 158.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 175.0, 0.0, 0.0, 0.0, 0.0], [194.0, 214.0, 172.0, 225.0, 0.0, 228.0, 214.0, 222.0, 0.0, 178.0, 0.0], [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 0.0, 166.0, 0.0], [184.0, 177.0, 163.0, 210.0, 0.0, 222.0, 203.0, 221.0, 0.0, 156.0, 0.0], [214.0, 222.0, 194.0, 234.0, 0.0, 0.0, 0.0, 0.0, 266.0, 187.0, 0.0], [135.0, 136.0, 142.0, 158.0, 0.0, 178.0, 166.0, 156.0, 187.0, 142.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
      'bonds2': [[134.0, 129.0, 120.0, 160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [129.0, 125.0, 121.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [120.0, 121.0, 121.0, 0.0, 0.0, 0.0, 0.0, 150.0, 0.0, 0.0, 0.0], [160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 186.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 150.0, 186.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
      'bonds3': [[120.0, 116.0, 113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [116.0, 110.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
      'lennard_jones_rm': [[120.0, 116.0, 113.0, 160.0, 133.0, 194.0, 177.0, 184.0, 214.0, 135.0, 0.0], [116.0, 110.0, 121.0, 168.0, 127.0, 214.0, 175.0, 177.0, 222.0, 136.0, 0.0], [113.0, 121.0, 121.0, 151.0, 126.0, 172.0, 164.0, 150.0, 194.0, 142.0, 0.0], [160.0, 168.0, 151.0, 204.0, 167.0, 225.0, 207.0, 186.0, 234.0, 158.0, 0.0], [133.0, 127.0, 126.0, 167.0, 146.0, 182.0, 175.0, 167.0, 198.0, 126.0, 0.0], [194.0, 214.0, 172.0, 225.0, 182.0, 228.0, 214.0, 222.0, 234.0, 178.0, 0.0], [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 218.0, 166.0, 0.0], [184.0, 177.0, 150.0, 186.0, 167.0, 222.0, 203.0, 221.0, 219.0, 156.0, 0.0], [214.0, 222.0, 194.0, 234.0, 198.0, 234.0, 218.0, 219.0, 266.0, 187.0, 0.0], [135.0, 136.0, 142.0, 158.0, 126.0, 178.0, 166.0, 156.0, 187.0, 142.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
      'atom_hist': {'C': 1570767, 'N': 273858, 'O': 396837, 'S': 26352, 'B': 0, 'Br': 0, 'Cl': 15058, 'P': 25994, 'I': 0, 'F': 30687, 'others': 0},
      'aa_hist': {'C': 23302704, 'N': 6093090, 'O': 6701210, 'S': 276805, 'B': 0, 'Br': 0, 'Cl': 0, 'P': 0, 'I': 0, 'F': 0, 'others': 0},
}

dataset_params['crossdock'] = {
      'atom_encoder': {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9},
      'atom_decoder': ['C', 'N', 'O', 'S', 'B', 'Br', 'Cl', 'P', 'I', 'F'],
      'aa_encoder': {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19},
      'aa_decoder': ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
      # PyMOL colors, see: https://pymolwiki.org/index.php/Color_Values#Chemical_element_colours
      'colors_dic': ['#33ff33', '#3333ff', '#ff4d4d', '#e6c540', '#ffb5b5', '#A62929', '#1FF01F', '#ff8000', '#940094', '#B3FFFF'],
      'radius_dic': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
      'bonds1': [[154.0, 147.0, 143.0, 182.0, 0.0, 194.0, 177.0, 184.0, 214.0, 135.0], [147.0, 145.0, 140.0, 168.0, 0.0, 214.0, 175.0, 177.0, 222.0, 136.0], [143.0, 140.0, 148.0, 151.0, 0.0, 172.0, 164.0, 163.0, 194.0, 142.0], [182.0, 168.0, 151.0, 204.0, 0.0, 225.0, 207.0, 210.0, 234.0, 158.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 175.0, 0.0, 0.0, 0.0], [194.0, 214.0, 172.0, 225.0, 0.0, 228.0, 214.0, 222.0, 0.0, 178.0], [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 0.0, 166.0], [184.0, 177.0, 163.0, 210.0, 0.0, 222.0, 203.0, 221.0, 0.0, 156.0], [214.0, 222.0, 194.0, 234.0, 0.0, 0.0, 0.0, 0.0, 266.0, 187.0], [135.0, 136.0, 142.0, 158.0, 0.0, 178.0, 166.0, 156.0, 187.0, 142.0]],
      'bonds2': [[134.0, 129.0, 120.0, 160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [129.0, 125.0, 121.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [120.0, 121.0, 121.0, 0.0, 0.0, 0.0, 0.0, 150.0, 0.0, 0.0], [160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 186.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 150.0, 186.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
      'bonds3': [[120.0, 116.0, 113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [116.0, 110.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
      'lennard_jones_rm': [[120.0, 116.0, 113.0, 160.0, 133.0, 194.0, 177.0, 184.0, 214.0, 135.0], [116.0, 110.0, 121.0, 168.0, 127.0, 214.0, 175.0, 177.0, 222.0, 136.0], [113.0, 121.0, 121.0, 151.0, 126.0, 172.0, 164.0, 150.0, 194.0, 142.0], [160.0, 168.0, 151.0, 204.0, 167.0, 225.0, 207.0, 186.0, 234.0, 158.0], [133.0, 127.0, 126.0, 167.0, 146.0, 182.0, 175.0, 167.0, 198.0, 126.0], [194.0, 214.0, 172.0, 225.0, 182.0, 228.0, 214.0, 222.0, 234.0, 178.0], [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 218.0, 166.0], [184.0, 177.0, 150.0, 186.0, 167.0, 222.0, 203.0, 221.0, 219.0, 156.0], [214.0, 222.0, 194.0, 234.0, 198.0, 234.0, 218.0, 219.0, 266.0, 187.0], [135.0, 136.0, 142.0, 158.0, 126.0, 178.0, 166.0, 156.0, 187.0, 142.0]],
      'atom_hist': {'C': 1570032, 'N': 273792, 'O': 396623, 'S': 26339, 'B': 0, 'Br': 0, 'Cl': 15055, 'P': 25975, 'I': 0, 'F': 30673},
      'aa_hist': {'A': 277175, 'C': 92406, 'D': 254046, 'E': 201833, 'F': 234995, 'G': 376966, 'H': 147704, 'I': 290683, 'K': 173210, 'L': 421883, 'M': 157813, 'N': 174241, 'P': 148581, 'Q': 120232, 'R': 173848, 'S': 274430, 'T': 247605, 'V': 326134, 'W': 88552, 'Y': 226668},
}
