## Instructions to run the code
1. The file `matbench_demo.py` can be used to fetch Matbench formation energy dataset.
2. The file `augment_from_file_uniform.py` can be used to augment the dataset with controlled perturbation without affecting the energy labels.
3. The file `augment_from_file_modelmapped_uniform.py` can be used to augment the dataset with controlled perturbation while using an existing model to generate energy labels.

## data should have below structure:

    data_directory
    ├── file_directory
    │   ├── *.cif
    │   ├── *.cif
    │   └── ...
    ├── file_name.csv
    └── file_name.pymatgen.json

if file_name.pymatgen.json is not available, it will be generated (will take more time), if present it'll be used to load data faster.
file_name.csv should have two columns with headers `filename` and `formation_energy`, `filename` entries should end with `.cif` and should be relative to file_directory.

## Model directory should have below structure:

    {model_dir_prefix}_{model_type_trial}
    ├── models
    │   ├── best_model_{model_type_trial}.weights.h5
    ├── scaler.json
    .
    .
model_dir_prefix: str, path to model directory+ prefix, eg "../model/dimenet_pp" 
    model_type_trial: str: eg M1_trial1, M2_trial5