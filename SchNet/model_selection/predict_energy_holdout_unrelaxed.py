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
                "module_name": "kgcnn.literature.Schnet",
                "class_name": "make_crystal_model",
                "config": {
                        "name": "Schnet",
                        "inputs": [
                            {'shape': (None,), 'name': "node_number", 'dtype': 'int32'},
                            {'shape': (None, 3), 'name': "node_coordinates", 'dtype': 'float32'},
                            {'shape': (None, 2), 'name': "range_indices", 'dtype': 'int64'},
                            {'shape': (None, 3), 'name': "range_image", 'dtype': 'int64'},
                            {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32'},
                            {"shape": (), "name": "total_nodes", "dtype": "int64"},
                            {"shape": (), "name": "total_ranges", "dtype": "int64"}
                        ],
                        "cast_disjoint_kwargs": {"padded_disjoint": False},
                        "input_node_embedding": {"input_dim": 95, "output_dim": 64},
                        "interaction_args": {
                            "units": 128, "use_bias": True,
                            "activation": {"class_name": "function", "config": "kgcnn>shifted_softplus"},
                            "cfconv_pool": "scatter_sum"
                        },
                        "node_pooling_args": {"pooling_method": "scatter_mean"},
                        "depth": 4,
                        "gauss_args": {"bins": 25, "distance": 5, "offset": 0.0, "sigma": 0.4}, "verbose": 10,
                        "last_mlp": {"use_bias": [True, True, True], "units": [128, 64, 1],
                                    "activation": [
                                        {"class_name": "function", "config": "kgcnn>shifted_softplus"},
                                        {"class_name": "function", "config": "kgcnn>shifted_softplus"},
                                        'linear']},
                        "output_embedding": "graph",
                        "use_output_mlp": False,
                        "output_mlp": None,  # Last MLP sets output dimension if None.
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
    '''
    holdout_dataset = CrystalDataset(
    data_directory="../schnet/data/2unrelaxeddataperstructuremaxatomcount8_using132krelaxedstructure",
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
            {"map_list": {"method": "set_range_periodic", "max_distance": 5, "max_neighbours": 32}},
            {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_ranges",
                                    "count_edges": "range_indices", "count_nodes": "node_number",
                                    "total_nodes": "total_nodes"}},
        ]
    )

    # assert holdout_dataset has correct format
    holdout_dataset.assert_valid_model_input(model_config["config"]["inputs"])

    print(f"Length of holdout_dataset: {len(holdout_dataset)}")

    return holdout_dataset
     

def load_model_and_scaler(model_dir_prefix, model_type_trial, model_config= model_config):
    '''
    loads model and scaler

    Args:
    model_dir_prefix: str, path to model directory+ prefix, eg "../schnet/model/schnet_kgcnn" 
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


def predict_energy(model_dir_prefix, model_type_trial, holdout_dataset):
    '''
    Predicts formation energy of holdout data using trained model

    Args:
    model_dir_prefix: str, path to model directory+ prefix, eg "../schnet/schnet_kgcnn" 
    model_type_trial: str: eg M1_trial1, M2_trial5
    holdout_dataset: CrystalDataset object

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

    df = holdout_dataset.data_frame.copy()
    assert len(df) == len(y_pred), f"Length of holdout data and predictions do not match: {len(df)} != {len(y_pred)}"
    df["predicted_formation_energy"] = y_pred

    return df

if __name__ == "__main__":
    model_dir_prefix = "../schnet/model/schnet_kgcnn"
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

    holdout_dataset = load_holdout_data()

    failed_models = []
    # predict energy for each model
    for model_type_trial in model_type_trial_list:
        print(f"##################Predicting energy for model {model_type_trial}##############")
        output_dir = f"./holdout_unrelaxed_predictions/schnet_kgcnn_{model_type_trial}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            raise Warning(f"Output directory {output_dir} already exists. Overwriting it.")
        try:
            df = predict_energy(model_dir_prefix, model_type_trial, holdout_dataset)
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
