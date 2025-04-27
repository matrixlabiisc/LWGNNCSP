## SchNet Model Training
1. Use `train_schnet_kgcnn.py` to train SchNet model on Matbench dataset. The hyperparameters used for training are available in `hyperparams.json`.
2. Command line args are as below:<br>
    2.1. `--data_path` : Path to the dataset dir, the dir should contain subdir with name `cif_files` that has cif files and a csv file `id_prop.csv` with headers `filename` and `formation_energy`, `filename` should end with .cif. <br>
    2.2. `--model_name` : Name of the model to be trained, for example `M1_Trial1` <br>
    2.3. `--random_seed` : Random seed for used for training, for example `0`, different random seeds will give us different trials of the model <br>
    2.4. `--hyperparams_path` : Path to the hyperparams.json file, for example `hyperparams.json`<br>

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

For above example, `--data_path` should be path to `data_directory`.
`file_directory`, `file_name.csv` values should be updated in `hyperparams.json` file. `default` values are `cif_files`, `id_prop.csv` respectively.
