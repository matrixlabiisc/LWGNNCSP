# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 12:47:30 2022

@author: ashro
"""

import os
from pymatgen.ext.matproj import MPRester
import csv
import pandas as pd
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import random
import pickle as pkl
from sklearn.mixture import GaussianMixture
from pymatgen.io.cif import CifParser
from pymatgen.core import Structure

np.random.seed(19)
random.seed(19)



def perturb(struct, low, high, scale):
    """Perturbs the atomic cordinates of a structure
    Parameters
    ----------
    struct: pymatgen.core.Structure
      pymatgen structure to be perturbed
    data: np.ndarray
      numpy array of possible magnitudes of perturbation
    Returns
    -------
    struct_per: pymatgen.core.Structure
      Perturbed structure
    """

    def get_rand_vec(dist):
        """Returns vector used to pertrub structure
        Parameters
        ----------
        dist: float
          Magnitude of perturbation
        Returns
        -------
        vector: np.ndarray
          Vector whos direction was randomly sampled from random sphere and magnitude is defined by dist
        """
        vector = np.random.randn(3)
        vnorm = np.linalg.norm(vector)
        return vector / vnorm * dist if vnorm != 0 else get_rand_vec(dist)

    struct_per = struct.copy()
    for i in range(len(struct_per._sites)):
        dist = np.random.uniform(low, high)*scale
        struct_per.translate_sites([i], get_rand_vec(dist), frac_coords=False)
    return struct_per


def write_structs(low, high, scale, num_per, input_dir, output_dir):
    """Write original and perturbed structure to a directory
    Parameters
    ----------
    results: List or np.ndarray
      query results from materials project
    dir_name: String
      Name of directory to write cif files to
    Returns
    -------
    E: List
      list in format to write id_prop.csv file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cif_files = [x for x in os.listdir(input_dir) if x[-4:] == '.cif']
    print(len(cif_files))
    id_prop = pd.read_csv(os.path.join(input_dir, 'id_prop.csv'), header = None)
    id_prop.columns = ['struct_idx', 'form_energy']
    #ind = 0
    E = []
    for cif in cif_files:
        #ind += 1
        curr_ind = cif.split('.')[0]
        curr_form_e = float(id_prop[id_prop['struct_idx'] == curr_ind]['form_energy'])
        struct = Structure.from_file(os.path.join(input_dir, cif))
        struct.to(fmt="cif", filename="{}/{}.cif".format(output_dir, curr_ind))
        E.append(['{}'.format(curr_ind), curr_form_e])
        #ind += 1
        for i in range(num_per):
            struct_per = perturb(struct, low, high, scale)
            struct_per.to(fmt="cif", filename='{}/{}_per{}.cif'.format(output_dir, curr_ind, i+1))
            E.append(['{}_per{}'.format(curr_ind, i+1), curr_form_e])
        
    return E


def write_dir(E, output_dir):
    """Writes id_prop.csv file for training of CGCNN
    Parameters
    ----------
    E: List
      list with the first dimension coresponding to the structures index and the
      second corresponding to the structures formation energy per atom
    dir_name: String
      Directory to write id_prop.csv to.
    """
    with open(f"{output_dir}/id_prop.csv", "w", newline="") as file:
        wr = csv.writer(file)
        wr.writerows(E)


if __name__ == "__main__":
    low = 0.0
    high = 0.3
    scale_per = 1 #Scaling the perturbation distance
    input_dir = 'matbench_allcifs_unperturbed_testremoved'
    num_per = 1
    output_dir = input_dir + '_per{}'.format(num_per) + '_scale{}'.format(scale_per) + '_uniform_low{}_high{}'.format(low, high)
    E = write_structs(low, high, scale_per, num_per, input_dir, output_dir)
    write_dir(E, output_dir)
