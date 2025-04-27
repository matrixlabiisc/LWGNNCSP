import os
import re
import json
import shutil
import pandas as pd
import numpy as np
import random

import tensorflow as tf
import keras as ks
from kgcnn.data.crystal import CrystalDataset
from kgcnn.data.transform.scaler.serial import deserialize as deserialize_scaler
from kgcnn.models.serial import deserialize as deserialize_model
from kgcnn.utils.devices import check_device, set_cuda_device

# required for deserializing model and validating dataset
model_config= {
                "class_name": "make_crystal_model",
                "module_name": "kgcnn.literature.DimeNetPP",
                "config": {
                    "name": "DimeNetPP",
                    "inputs": [
                        {"shape": [None], "name": "node_number", "dtype": "int32", "ragged": True},
                        {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                        {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
                        {"shape": [None, 2], "name": "angle_indices", "dtype": "int64", "ragged": True},
                        {'shape': (None, 3), 'name': "range_image", 'dtype': 'int64', 'ragged': True},
                        {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False}
                    ],
                    "input_tensor_type": "ragged",
                    "input_embedding": None,
                    "input_node_embedding": {"input_dim": 95, "output_dim": 128,
                                            "embeddings_initializer": {"class_name": "RandomUniform",
                                                                        "config": {"minval": -1.7320508075688772,
                                                                                "maxval": 1.7320508075688772}}},
                    "emb_size": 128, "out_emb_size": 256, "int_emb_size": 64, "basis_emb_size": 8,
                    "num_blocks": 4, "num_spherical": 7, "num_radial": 6,
                    "cutoff": 5.0, "envelope_exponent": 5,
                    "num_before_skip": 1, "num_after_skip": 2, "num_dense_output": 3,
                    "num_targets": 1, "extensive": False, "output_init": "zeros",
                    "activation": "swish", "verbose": 10,
                    "output_embedding": "graph",
                    "use_output_mlp": False,
                    "output_mlp": {},
                        }
                }

def sort_per_filename(df):
    def extract_int(filename):
        match = re.search(r't(\d+)', filename)
        return int(match.group(1)) if match else float('inf')
    
    df.loc[:,'int']= df['filename'].apply(extract_int)
    df = df.sort_values('int', ascending=True, ignore_index=True)
    return df.drop(columns='int')

def seperate_data(data_path, output_path):
    df = pd.read_csv(os.path.join(data_path, 'id_prop.csv'))
    df.columns = ["filename","formation_energy"]
    df['filename'] = df['filename']+ '.cif'

    print("length of data: ", len(df))

    perturbed = df[df['filename'].str.contains("per")]
    relaxed = df[~df['filename'].str.contains("per")]

    perturbed = sort_per_filename(perturbed)
    relaxed = sort_per_filename(relaxed)

    print("length of perturbed data: ", len(perturbed))
    print("length of relaxed data: ", len(relaxed))

    # copy perturbed cifs in a temp dir
    os.makedirs(output_path, exist_ok=True)

    if os.path.exists(os.path.join(output_path, 'temp_perturbed_cifs')):
        print("Deleting existing temp_perturbed_cifs")
        shutil.rmtree(os.path.join(output_path, 'temp_perturbed_cifs'))
    
    print("Creating temp_perturbed_cifs dir")
    os.mkdir(os.path.join(output_path, 'temp_perturbed_cifs'))

    print("Copying perturbed cifs to temp_perturbed_cifs")
    for file in os.listdir(os.path.join(data_path, 'cif_files')):
        if 'per' in file:
            shutil.copy(os.path.join(data_path, 'cif_files', file), os.path.join(output_path, 'temp_perturbed_cifs'))
    
    perturbed.to_csv(os.path.join(output_path, 'perturbed.csv'), index=False, header=True)
    relaxed.to_csv(os.path.join(output_path, 'relaxed.csv'), index=False, header=True)
    print("perturbed and relaxed csv files saved succesfully")
    print("Data seperated successfully")

def load_dataset(data_directory, dataset_name,  file_name, file_directory):
    '''
    loads data, data should have below structure:

    data_directory
    ├── file_directory
    │   ├── *.cif
    │   ├── *.cif
    │   └── ...
    ├── file_name.csv
    └── file_name.pymatgen.json

    if file_name.pymatgen.json is not available, it will be generated (will take more time), if present it'll be
    used to load data faster.

    Returns:
    dataset: CrystalDataset object
    '''
    dataset = CrystalDataset(
    data_directory=data_directory,
    dataset_name=dataset_name,
    file_name=file_name,
    file_directory=file_directory,
    file_name_pymatgen_json=None)

    dataset.prepare_data(
        file_column_name="filename",
        overwrite=False,
    )
    # this makes sure that each formation energy is np.array of shape (1,)
    print(f"Applying additional callbacks to dataset")
    dataset.read_in_memory(
        additional_callbacks={"graph_labels": lambda st, ds: np.array([ds["formation_energy"]])})

    print(f"Setting methods to dataset")
    dataset.set_methods(
        [
            {"map_list": {"method": "set_range_periodic", "max_distance": 5.0, "max_neighbours": 17}},
            {"map_list": {"method": "set_angle", "allow_multi_edges": True, "allow_reverse_edges": True}},
        ]
    )

    # assert dataset has correct format
    print(f"Asserting dataset has valid model input")
    dataset.assert_valid_model_input(model_config["config"]["inputs"])

    print(f"Length of dataset before cleaning: {len(dataset)}")
    cleaned_idx = dataset.clean(model_config["config"]["inputs"])
    print("#####################################################################")
    print(f"Removed {len(cleaned_idx)} entries with missing data")
    print(f"Length of cleaned dataset: {len(dataset)}")
    print(f"Successfully loaded dataset")

    return dataset, cleaned_idx

def load_model_and_scaler(model_dir_prefix, model_type_trial):
    '''
    loads model and scaler

    Args:
    model_dir_prefix: str, path to model directory+ prefix, eg "../model/dimenet_pp" 
    model_type_trial: str: eg M1_trial1, M2_trial5

    these two args will form the path to model by concatenating them as f"{model_dir_prefix}_{model_type_trial}"
    the directory should have below structure:
    {model_dir_prefix}_{model_type_trial}
    ├── models
    │   ├── best_model_{model_type_trial}.weights.h5
    ├── scaler.json
    .
    .

    Returns:
    model: keras model
    scaler: deserialized scaler object
    '''
    model_dir = f"{model_dir_prefix}_{model_type_trial}"
    model = deserialize_model(model_config)
    model.load_weights(f"{model_dir}/models/best_model_{model_type_trial}.weights.h5")
    with open(f"{model_dir}/models/scaler.json", "r") as f:
        scaler = deserialize_scaler(json.load(f))
    
    return model, scaler

def predict_energy(model_dir_prefix, model_type_trial, dataset, cleaned_idx):
    '''
    Predicts formation energy of data using trained model

    Args:
    model_dir_prefix: str, path to model directory+ prefix, eg "../model/dimenet_pp" 
    model_type_trial: str: eg M1_trial1, M2_trial5
    dataset: CrystalDataset object

    Returns:
    df: pandas dataframe with columns struct_idx, ground_truth_relaxed_mapping, pred_form_energy
    '''
    # load model and scaler
    model, scaler = load_model_and_scaler(model_dir_prefix, model_type_trial)
    
    y_pred = np.array([])

    for i in range(14):
        # load  data
        print(f"Loading data from {10000*i}:{10000*(i+1)}")
        x = dataset[10000*i:10000*(i+1)].tensor(model_config["config"]["inputs"])  
        # predict
        print(f"Predicting formation energy")
        y_pred_batch = model.predict(x)
        # inverse transform to get actual values
        y_pred_batch = scaler.inverse_transform(y_pred_batch)
        y_pred = np.append(y_pred, y_pred_batch)

    df = dataset.data_frame.copy()
    
    dropped = None
    if len(cleaned_idx) > 0:
        dropped = df.loc[cleaned_idx]
        df = df.drop(cleaned_idx)

    assert len(df) == len(y_pred), f"Length of data and predictions do not match: {len(df)} != {len(y_pred)}"
    df["formation_energy"] = y_pred

    return df, dropped

if __name__ == "__main__":
    set_cuda_device(0)
    print("Device being used:", check_device())
    # Enable memory growth for the GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    np.random.seed(19)
    random.seed(19)
    ks.utils.set_random_seed(19)

    # seperate data
    data_path = './matbench_allcifs_unperturbed_testremoved_per1_scale1_uniform_low0.0_high0.3'
    output_path = './matbench_perturbed_unpurterbed_per1_scale1_uniform_low0.0_high0.3_modelmapped'

    print("Seperating data...")
    seperate_data(data_path, output_path)

    model_dir_prefix = "../model/dimenet_pp"
    model_type_trial = "M1_trial1"

    # load perturbed dataset
    purterbed_dataset, cleaned_idx = load_dataset(output_path, "perturbed", "perturbed.csv", "temp_perturbed_cifs")

    # predict energy
    perturbed_model_mapped_df, cleaned_df = predict_energy(model_dir_prefix, model_type_trial, purterbed_dataset, cleaned_idx)

    # delete cleaned cifs
    if cleaned_df is not None:
        count = 0
        for file in cleaned_df['filename']:
            if os.path.exists(os.path.join(output_path, 'temp_perturbed_cifs', file)):
                os.remove(os.path.join(output_path, 'temp_perturbed_cifs', file))
                count += 1
        print(f"{count} invalid cifs removed successfully")

    # load relaxed dataset
    relaxed_df = pd.read_csv(os.path.join(output_path, 'relaxed.csv'))

    complete_df = pd.concat([relaxed_df, perturbed_model_mapped_df],axis = 0, ignore_index=True)
    complete_df.to_csv(os.path.join(output_path, 'id_prop.csv'), index=False, header=True)

    # copy paste rest of relaxed cifs
    print("Copying rest of the relaxed cifs")
    for file in os.listdir(os.path.join(data_path, 'cif_files')):
            if 'per' in file:
                continue
            else:
                shutil.copy(os.path.join(data_path, 'cif_files', file), os.path.join(output_path, 'temp_perturbed_cifs'))
    print("All cifs copied successfully")

    # rename temp_perturbed_cifs to cif_files
    os.rename(os.path.join(output_path, 'temp_perturbed_cifs'), os.path.join(output_path, 'cif_files'))
    print("temp_perturbed_cifs renamed to cif_files")

    print("Data mapped successfully")