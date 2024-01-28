# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 12:52:37 2022

@author: ashro
"""
import argparse
import os
import shutil
import sys
import time
from tqdm import tqdm

import numpy as np
import pandas as pd
from pymatgen.core import Structure
# import torch
# import torch.nn as nn
# from sklearn import metrics
# from torch.autograd import Variable
# from torch.utils.data import DataLoader

# from src.data_prop import CIFData, collate_pool
# from src.model import CrystalGraphConvNet
import tensorflow as tf
from megnet.models import MEGNetModel, GraphModel
from megnet.data.crystal import CrystalGraph
from tensorflow.keras.models import load_model
from megnet.layers import _CUSTOM_OBJECTS


def string_to_float(x):
    try:
        return float(x.split('[')[1].split(']')[0])
    except:
        return float(x)
    
    
def predict_energy(model_dir, cifdir):
    #We can either pass structure or the cif file as input along with model.
    #Output must be a single number with energy value.
    
    
    if os.path.isdir(model_dir):
        print('Model path is a directory.')
        model_files = os.listdir(model_dir)
        model_files = [mf for mf in model_files if '.hdf5' in mf]
        #model_files.sort(key=lambda fn: os.path.getmtime(model_path + '/' + fn))
        val_errors = [float(x.split('_')[-1][:-5]) for x in model_files]
        min_val = min(val_errors)
        print(min_val)
        idx = val_errors.index(min_val)
        model_file_path = os.path.join(model_dir, model_files[idx])
    elif os.path.isfile(model_dir):
        model_file_path = model_dir
    else:
        print('Input a GN model path!')
        
    nfeat_bond = 100
    r_cutoff = 5
    gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
    gaussian_width = 0.5
    graph_converter = CrystalGraph(cutoff=r_cutoff)

    model = load_model(model_file_path, custom_objects = _CUSTOM_OBJECTS)

    gn_model = GraphModel(model=model,
                    graph_converter=graph_converter,
                    centers=gaussian_centers,
                    width=gaussian_width)
    
    
    cif_files = [x.split('.')[0] for x in os.listdir(cifdir) if x.endswith('.cif')]
    
    all_cif_ids = []
    all_preds = []
    
    for i in range(len(cif_files)):
        conventional_structure = Structure.from_file(os.path.join(cifdir, cif_files[i]+'.cif'))
        try:
            pred_energy = gn_model.predict_structure(conventional_structure).reshape(-1)[0]
            all_cif_ids.append(cif_files[i])
            all_preds.append(pred_energy)
        except:
            pass
    
    df = pd.DataFrame()
    df['struct_idx'] = all_cif_ids
    df['form_energy'] = all_preds
    
    print(min(all_preds), max(all_preds))
    return df

if __name__=='__main__':
    modelpath = './model_best_trainmpdata75-230_testmpdata75-230.pth.tar'
    cifdir = '../Data/One_Item/'
    predict_energy(modelpath, cifdir)