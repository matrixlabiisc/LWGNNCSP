import numpy as np
import pandas as pd
from pymatgen.core import Structure
import matbench
import os
from matminer.datasets import load_dataset
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

df = load_dataset('matbench_mp_e_form')

path_to_data = './matbench_allcifs_unperturbed_testremoved/'

if not os.path.exists(path_to_data):
    os.makedirs(path_to_data)
    

materials_to_remove = ['LiF', 'NaF', 'KF', 'RbF', 'CsF', 'LiCl', 'NaCl', 'KCl', 'RbCl', 'CsCl',
                      'BeO', 'MgO', 'CaO', 'SrO', 'BaO', 'ZnO', 'CdO', 'BeS', 'MgS', 'CaS', 'SrS',
                      'BaS', 'ZnS', 'CdS', 'C', 'Si', 'GaAs', 'CdTe', 'CsPbI3',  'BN', 'GaN', 'ZrC',
                      'SiC', 'WC', 'MnO', 'GaP', 'AlP', 'InP', 'LiBr', 'NaBr', 'KBr', 'RbBr', 'BeSe',
                      'MgSe', 'CaSe', 'SrSe', 'BaSe']

removed_cifs = []

for i in range(len(df)):
    SGA = SpacegroupAnalyzer(df.iloc[i, 0])
    conventional_structure = SGA.get_conventional_standard_structure()
    if conventional_structure.composition.reduced_formula not in materials_to_remove:
        conventional_structure.to(fmt = 'cif', filename = os.path.join(path_to_data, 't{}.cif'.format(i)))
    else:
        removed_cifs.append('t{}'.format(i))
        
indices = ['t{}'.format(i) for i in range(len(df))]
id_prop = pd.DataFrame()
id_prop['indices'] = indices
id_prop['form'] = df.iloc[:, 1]
id_prop = id_prop[~id_prop['indices'].isin(removed_cifs)]
id_prop.to_csv(os.path.join(path_to_data, 'id_prop.csv'), header = None, index = None)

