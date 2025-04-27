## Predicts formation energy of holdout data using trained model
import os
import json
import pandas as pd
import tensorflow as tf
import keras as ks
import numpy as np
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

def load_holdout_data():
    '''
    loads holdout data, holdout data should have below structure:

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
    holdout_dataset: CrystalDataset object
    cleaned_idx: list, indices of cleaned data
    '''
    holdout_dataset = CrystalDataset(
    data_directory="../dimenet_pp/data/2unrelaxeddataperstructuremaxatomcount8_using132krelaxedstructure",
    dataset_name="holdout_unrelaxed",
    file_name="id_prop.csv",
    file_directory="cif_files",
    file_name_pymatgen_json=None)

    holdout_dataset.prepare_data(
        file_column_name="filename",
        overwrite=False,
    )
    # this makes sure that each formation energy is np.array of shape (1,)
    print(f"Applying additional callbacks to holdout_dataset")
    holdout_dataset.read_in_memory(
        additional_callbacks={"graph_labels": lambda st, ds: np.array([ds["formation_energy"]])})

    print(f"Setting methods to holdout_dataset")
    holdout_dataset.set_methods(
        [
            {"map_list": {"method": "set_range_periodic", "max_distance": 5.0, "max_neighbours": 17}},
            {"map_list": {"method": "set_angle", "allow_multi_edges": True, "allow_reverse_edges": True}},
        ]
    )

    # assert holdout_dataset has correct format
    holdout_dataset.assert_valid_model_input(model_config["config"]["inputs"])

    print(f"Length of holdout_dataset before cleaning: {len(holdout_dataset)}")
    cleaned_idx = holdout_dataset.clean(model_config["config"]["inputs"])
    print("#####################################################################")
    print(f"Removed {len(cleaned_idx)} entries with missing data")
    print(f"Length of cleaned holdout_dataset: {len(holdout_dataset)}")

    return holdout_dataset, cleaned_idx
     

def load_model_and_scaler(model_dir_prefix, model_type_trial, model_config= model_config):
    '''
    loads model and scaler

    Args:
    model_dir_prefix: str, path to model directory+ prefix, eg "../diment_pp/model/diment_pp" 
    model_type_trial: str: eg M1_trial1, M2_trial5
    model_config: dict, model configuration

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


def predict_energy(model_dir_prefix, model_type_trial, holdout_dataset, df):
    '''
    Predicts formation energy of holdout data using trained model

    Args:
    model_dir_prefix: str, path to model directory+ prefix, eg "../diment_pp/model/diment_pp" 
    model_type_trial: str: eg M1_trial1, M2_trial5
    holdout_dataset: CrystalDataset object
    df: pandas dataframe with columns struct_idx, ground_truth_relaxed_mapping

    Returns:
    df: pandas dataframe with columns struct_idx, ground_truth_relaxed_mapping, pred_form_energy
    '''
    # load holdout data
    x_holdout = holdout_dataset.tensor(model_config["config"]["inputs"])  

    # load model and scaler
    model, scaler = load_model_and_scaler(model_dir_prefix, model_type_trial)

    # predict
    y_pred = model.predict(x_holdout)
    # inverse transform to get actual values
    y_pred = scaler.inverse_transform(y_pred)

    assert len(df) == len(y_pred), f"Length of holdout data and predictions do not match: {len(df)} != {len(y_pred)}"
    df["predicted_formation_energy"] = y_pred

    return df

if __name__ == "__main__":
    model_dir_prefix = "../dimenet_pp/model/dimenet_pp"
    model_type_trial_list = ["M1_trial1", "M1_trial2", "M1_trial3", "M1_trial4", "M1_trial5",
                             "M2_trial1", "M2_trial2", "M2_trial3", "M2_trial4", "M2_trial5"]

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

    holdout_dataset, cleaned_idx = load_holdout_data()
    data_df = holdout_dataset.data_frame.copy()
    if len(cleaned_idx):
        data_df = data_df.drop(cleaned_idx)
        data_df = data_df.reset_index(drop=True)


    failed_models = []
    # predict energy for each model
    for model_type_trial in model_type_trial_list:
        print(f"##################Predicting energy for model {model_type_trial}##############")
        output_dir = f"./holdout_unrelaxed_predictions/dimenetpp_{model_type_trial}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            raise Warning(f"Output directory {output_dir} already exists. Overwriting it.")
        try:
            df = predict_energy(model_dir_prefix, model_type_trial, holdout_dataset, data_df)
            df.to_csv(os.path.join(output_dir, 'pred_results.csv'), header = True, index = False)
            print(f"Predictions saved to {output_dir}/pred_results.csv")
        except Exception as e:
            failed_models.append(model_type_trial)
            print(f"Error in predicting energy for model {model_type_trial}: {e}")
            continue
    
    print('###################### Summary ######################')
    if failed_models:
        print(f"Failed models: {failed_models}")
    else:
        print("All models predicted successfully")
    print('#####################################################')
