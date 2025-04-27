#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from pymatgen.core import Structure
import os
from matminer.datasets import load_dataset
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.structure_matcher import ElementComparator
from pymatgen.ext.matproj import MPRester
import shutil
import argparse
from dotenv import load_dotenv 

parser = argparse.ArgumentParser()
parser.add_argument('--compound', type = str)
args = parser.parse_args()

path_to_crystal = '../results/{}'.format(args.compound)
optimizer = 'tpe'
pretty_formula = args.compound
path_to_data = os.path.join(path_to_crystal, optimizer)


load_dotenv()
MP_API_KEY = os.getenv('MP_API_KEY')
properties = ["cifs.conventional_standard", "formation_energy_per_atom"]
criteria = {"formation_energy_per_atom": {"$exists": True}, "pretty_formula": pretty_formula}

with MPRester(MP_API_KEY) as mpr:
    ground_truth = mpr.query(criteria, properties)

print(len(ground_truth))

all_directories = sorted(os.listdir(path_to_data))

final_matching = {}


for directory in all_directories:
    path_to_curr_dir = os.path.join(path_to_data, directory)
    print(directory)
    energy_csv = pd.read_csv(path_to_curr_dir + '/results/energy_data.csv')
    min_energy = np.min(energy_csv['energy'])
    print(min_energy)
    #print(energy_csv[energy_csv['energy'] == min_energy]['step'])
    optimal_step = int(energy_csv[energy_csv['energy'] == min_energy]['step'].iloc[0])


    best_cif = [x for x in os.listdir(path_to_curr_dir + '/results/structures/') if x.split('_')[-2] == str(optimal_step)]
    shutil.copy(path_to_curr_dir + '/results/structures/' + best_cif[0], path_to_curr_dir + '/results/structures/' + 'best_cif.cif')
    
    pred = Structure.from_file(path_to_curr_dir + '/results/structures/' + 'best_cif.cif')
    for i in range(len(ground_truth)):
        true = Structure.from_str(ground_truth[i]['cifs.conventional_standard'], fmt = 'cif')
        sm = StructureMatcher(comparator = ElementComparator(), primitive_cell = False)
        if sm.fit(pred, true) == True:
            true_form = ground_truth[i]['formation_energy_per_atom']
            form_e_error = abs(true_form - min_energy)
            lattice_pred = np.array([pred.as_dict()['lattice']['a'], pred.as_dict()['lattice']['b'],
                                     pred.as_dict()['lattice']['c']])
            lattice_true = np.array([true.as_dict()['lattice']['a'], true.as_dict()['lattice']['b'],
                                     true.as_dict()['lattice']['c']])
            mape_lattice = np.mean((np.abs(lattice_true - lattice_pred) / lattice_true) * 100)
            final_matching[directory] = [True, form_e_error, mape_lattice]
            break

print(final_matching)
