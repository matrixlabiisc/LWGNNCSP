import argparse
import csv
import os
import shutil
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from jarvis.core.atoms import Atoms
from jarvis.db.jsonutils import loadjson
from torch.utils.data import DataLoader
from alignn.data import get_torch_dataset
from alignn.config import TrainingConfig
from ignite.handlers import Checkpoint

from alignn.models.alignn import ALIGNN
from alignn.models.alignn_atomwise import ALIGNNAtomWise
from alignn.models.alignn_layernorm import ALIGNN as ALIGNN_LN
from alignn.models.modified_cgcnn import CGCNN
from alignn.models.dense_alignn import DenseALIGNN
from alignn.models.densegcn import DenseGCN
from alignn.models.icgcnn import iCGCNN
from alignn.models.alignn_cgcnn import ACGCNN



def generate_test_data_loader(cifdir, config):
    '''
    config = loadjson(config_file)
    print(config)
    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)
    '''
            
    id_prop_dat = os.path.join(cifdir, "id_prop.csv")

    with open(id_prop_dat, "r") as f:
        reader = csv.reader(f)
        data = [row for row in reader]

    dataset = []

    for i in tqdm(data):
        info = {}
        file_name = i[0] + '.cif'
        file_path = os.path.join(cifdir, file_name)
        try:
            atoms = Atoms.from_cif(file_path, get_primitive_atoms = False, use_cif2cell = False)
            #print(atoms.composition, atoms.num_atoms, atoms.get_spacegroup)
        except:
            raise NotImplementedError(
                "File format not implemented"
            )

        info["atoms"] = atoms.to_dict()
        info["jid"] = file_name

        tmp = [float(j) for j in i[1:]]  # float(i[1])
        if len(tmp) == 1:
            tmp = tmp[0]
        else:
            raise NotImplementedError('Currently supporting only one output at a time')
        info["target"] = tmp  
        dataset.append(info)
    
    all_targets = []

    for i in dataset:
        all_targets.append(i['target'])

    test_data = get_torch_dataset(
            dataset=dataset,
            id_tag=config.id_tag,
            atom_features=config.atom_features,
            target=config.target,
            target_atomwise='',
            target_grad='',
            target_stress='',
            neighbor_strategy=config.neighbor_strategy,
            use_canonize=config.use_canonize,
            name='test',
            line_graph=True,
            cutoff=config.cutoff,
            max_neighbors=config.max_neighbors,
            classification=config.classification_threshold is not None,
            output_dir='.', # Check back what it does
            tmp_name="test_data_geo_opt",
        )   
    
    collate_fn = test_data.collate_line_graph
    
    test_loader = DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
            num_workers=1,
            pin_memory=False,
        )
    
    return test_loader

def predict_energy(modelpath, cifdir, config_file, device):
    #Output must be a single number with energy value.
    config = loadjson(config_file)
    print(config)
    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)
    
    # Generating test data loader
    test_loader = generate_test_data_loader(cifdir, config)

    # Defining structure of the model
    _model = {
        "cgcnn": CGCNN,
        "icgcnn": iCGCNN,
        "densegcn": DenseGCN,
        "alignn": ALIGNN,
        "alignn_atomwise": ALIGNNAtomWise,
        "dense_alignn": DenseALIGNN,
        "alignn_cgcnn": ACGCNN,
        "alignn_layernorm": ALIGNN_LN,
    }
    net = _model.get(config.model.name)(config.model)
    to_save = {'model' : net}

    # Finding the best model checkpoint file
    output_files = os.listdir(modelpath)
    pt_files = [x for x in output_files if x.endswith('.pt')]
    val_neg_mae_list = [float(x.split('_')[3].split('=')[1][:-3]) for x in pt_files]
    min_mae = max(val_neg_mae_list)
    best_pt_file = [x for x in pt_files if str(min_mae) in x]
    best_pt_file = best_pt_file[0]

    print('Best Validation Model : {}'.format(best_pt_file))
    best_val_ckpt = torch.load(os.path.join(modelpath, best_pt_file), map_location=device)
    Checkpoint.load_objects(to_load=to_save, checkpoint=best_val_ckpt)

    # Sending model to GPU and setting it to evaluation mode.
    net.to(device)
    net.eval()
    #Checkpoint.load_objects(to_load=to_save, checkpoint=best_val_ckpt)
    cif_files = [x.split('.')[0] for x in test_loader.dataset.ids]
    ground_truths = []
    predictions = []
    with torch.no_grad():
        ids = test_loader.dataset.ids
        for dat, id in tqdm(zip(test_loader, ids), total=len(ids)):
            #print(id)
            g, lg, target = dat
            target = float(target.cpu().numpy()[0])
            #print(type(target), target)
            out_data = net([g.to(device), lg.to(device)])
            out_data = out_data.cpu().numpy().tolist()
            #print(type(out_data), out_data)
            ground_truths.append(target)
            predictions.append(out_data)

    df = pd.DataFrame()
    df['struct_idx'] = cif_files
    df['ground_truth_relaxed_mapping'] = ground_truths
    df['pred_form_energy'] = predictions
    return df
    

if __name__=='__main__':
    modelpath = '../trained_models/M2_trial5_modelmapped/'
    cifdir = './2unrelaxeddataperstructuremaxatomcount8_using132krelaxedstructure/'
    config_file = '../config_runs/M2_trial5_modelmapped.json'
    device = torch.device("cuda")
    base_output_dir = '../../results/Holdout_Unrelaxed/'
    pid = 'frommatbench132kmaxatoms200novolumeconstraint_2unrelaxedm2modelmappingtrial5'
    output_dir = os.path.join(base_output_dir, pid)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df = predict_energy(modelpath, cifdir, config_file, device)
    df.to_csv(os.path.join(output_dir, 'pred_results.csv'), header = None, index = None)
    
    
