# Load the trained model and predict the energy of a single crystal structure 
# Author: Rushikesh Pawar (Date : 18 Dec 2024)

import os
import json
from collections import defaultdict
from typing import Dict, Callable, List, Union

import pymatgen
from pymatgen.io.cif import CifParser

import numpy as np
import tensorflow as tf
import keras as ks

from kgcnn.data.transform.scaler.standard import StandardLabelScaler
from kgcnn.data.crystal import CrystalDataset
from kgcnn.data.transform.scaler.serial import deserialize as deserialize_scaler
from kgcnn.models.serial import deserialize as deserialize_model
from kgcnn.utils.devices import check_device, set_cuda_device

def load_model_and_scaler(model_dir_prefix, model_type_trial, model_config):
    '''
    loads model and scaler

    Args:
    model_dir_prefix: str, path to model directory+ prefix, eg "./schnet/model/schnet_kgcnn" 
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

class SingleCrystalDataset(CrystalDataset):
    '''
    Wrapper class for CrystalDataset to load single crystal data from cif file
    Capable of reading in memory and setting methods and returns x only

    written by: Rushikesh Pawar
    '''
    def __init__(self, cif_dir, methods):
        '''
        Args:
        cif_dir: str, path to cif file eg "./schnet/data/test_single_cif/cif_files/t0.cif"
        methods: list of str, methods to be set for the dataset
        '''
        super(SingleCrystalDataset, self).__init__(data_directory=None,
                                                        dataset_name=cif_dir,
                                                        file_name=None,
                                                        file_directory=None,
                                                        file_name_pymatgen_json=None)
        self.cif_dir = cif_dir
        self._structs = [self.load_structure_from_cif(cif_dir)]
        self.read_in_memory()
        self.set_methods(methods)
    

    def load_structure_from_cif(self, cif_dir):
        parser = CifParser(cif_dir)
        structure = parser.parse_structures(primitive = False)[0]
        return structure
    
    def read_in_memory(self):
        """Read structures from pymatgen json serialization and convert them into graph information.

        Returns:
            self
        """

        self.info("Making node features from structure...")
        callbacks = {"graph_labels": lambda st: None,
            "node_coordinates": lambda st: np.array(st.cart_coords, dtype="float"),
            "node_frac_coordinates": lambda st: np.array(st.frac_coords, dtype="float"),
            "graph_lattice": lambda st: np.ascontiguousarray(np.array(st.lattice.matrix), dtype="float"),
            "abc": lambda st: np.array(st.lattice.abc),
            "charge": lambda st: np.array([st.charge], dtype="float"),
            "volume": lambda st: np.array([st.lattice.volume], dtype="float"),
            "node_number": lambda st: np.array(st.atomic_numbers, dtype="int"),
            }

        self._map_callbacks(structs=self._structs, 
                            callbacks=callbacks)

        return self
          

    def _map_callbacks(self, structs: list,
                       callbacks: Dict[
                           str, Callable[[pymatgen.core.structure.Structure], Union[np.ndarray, None]]],
                       assign_to_self: bool = True) -> dict:
        """Map callbacks on a data series object plus structure list.

        Args:
            structs (list): List of pymatgen structures.
            callbacks (dict): Dictionary of callbacks that take a data object plus pymatgen structure as argument.
            assign_to_self (bool): Whether to already assign the output of callbacks to this class.

        Returns:
            dict: Values of callbacks.
        """

        # The dictionaries values are lists, one for each attribute defines in "callbacks" and each value in those
        # lists corresponds to one structure in the dataset.
        value_lists = defaultdict(list)
        for index, st in enumerate(structs):
            for name, callback in callbacks.items():
                if st is None:
                    value_lists[name].append(None)
                else:
                    value = callback(st)
                    value_lists[name].append(value)
        self.info(f"Succesfully read structure at {self.cif_dir}")

        # The string key names of the original "callbacks" dict are also used as the names of the properties which are
        # assigned
        if assign_to_self:
            for name, values in value_lists.items():
                self.assign_property(name, values)

        return value_lists
    
def predict_energy(cif_dir:str, model_dir_prefix:str, model_type_trial:str, model_config:Dict, methods: List[Dict], if_use_gpu:bool=False):
    '''
    Predicts energy of a single crystal structure

    Args:
    cif_dir: str, path to cif file eg "./schnet/data/test_single_cif/cif_files/t0.cif"
    model_dir_prefix: str, path to model directory+ prefix, eg "./schnet/model/schnet_kgcnn" 
    model_type_trial: str: eg M1_trial1, M2_trial5
    model_config: dict, model configuration
    methods: list of str, methods to be set for the dataset
    if_use_gpu: bool, whether to use gpu or not (default: False)

    these two args will form the path to model by concatenating them as f"{model_dir_prefix}_{model_type_trial}"
    the directory should have below structure:
    {model_dir_prefix}_{model_type_trial}
    ├── models
    │   ├── best_model_{model_type_trial}.weights.h5
    ├── scaler.json

    Returns:
    energy: float, predicted energy
    '''
    assert cif_dir.endswith(".cif"), f"{cif_dir} should be a cif file"
    assert os.path.exists(cif_dir), f"{cif_dir} does not exist"
    assert os.path.exists(model_dir_prefix+"_"+model_type_trial), f"{model_dir_prefix}_{model_type_trial} does not exist"

    gpus_available = tf.config.experimental.list_physical_devices('GPU')
    if if_use_gpu and gpus_available:
        set_cuda_device(0)
        print("Device being used:", check_device())
        # Enable memory growth for the GPU
        try:
            for gpu in gpus_available:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # load model and scaler
    model, scaler = load_model_and_scaler(model_dir_prefix, model_type_trial, model_config)
    # load data
    try:
        data = SingleCrystalDataset(cif_dir, methods)
        x = data.tensor(model_config["config"]["inputs"])  
        energy = model(x, training=False).numpy()
        energy = scaler.inverse_transform(energy)
        return energy.item()
    except Exception as e:
        print(f"Error while predicting energy for {cif_dir}")
        print(e)
        return 999

def predict_energy_with_model_scaler(model:ks.Model, scaler:StandardLabelScaler,  model_config:Dict, methods: List[Dict], cif_dir:str ):
    '''
    Predicts energy of a single crystal structure

    Args:
    cif_dir: str, path to cif file eg "./schnet/data/test_single_cif/cif_files/t0.cif"
    model: keras model
    scaler: deserialized scaler object
    model_config: dict, model configuration

    Returns:
    energy: float, predicted energy
    '''
    assert cif_dir.endswith(".cif"), f"{cif_dir} should be a cif file"
    assert os.path.exists(cif_dir), f"{cif_dir} does not exist"
    
    try:
        # load data
        data = SingleCrystalDataset(cif_dir, methods)
        x = data.tensor(model_config["config"]["inputs"])  

        energy = model(x, training=False).numpy()
        energy = scaler.inverse_transform(energy)
        return energy.item()
    except Exception as e:
        print(f"Error while predicting energy for {cif_dir}")
        print(e)
        return 999

if __name__ == "__main__":

    cif_dir = "./schnet/data/matbench_allcifs_unperturbed_testremoved/cif_files/t0.cif"
    model_dir_prefix = "./schnet/model/schnet_kgcnn"
    model_type_trial = "M1_trial1"

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

    methods = [
                {"map_list": {"method": "set_range_periodic", "max_distance": 5, "max_neighbours": 32}},
                {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_ranges",
                                        "count_edges": "range_indices", "count_nodes": "node_number",
                                        "total_nodes": "total_nodes"}},
            ]
    
    energy = predict_energy(cif_dir, model_dir_prefix, model_type_trial, model_config, methods, if_use_gpu=False)
    print()
    print(f"Predicted energy for crystal at {cif_dir} is: {energy} eV")