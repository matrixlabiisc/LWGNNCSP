# Crystal Structure Prediction with DimeNet++ and Bayesian Optimization
Crystal structure prediction with DimeNet++ as GNN backbone involves three key steps: (a) Training DimeNet++ model (b) Geometry Optimization with Bayesian Optimization (c) Performance Evaluation

## Training DimeNet++ model
DimeNet++ model was trained on Matbench dataset and its augmented counterpart as required for formation energy prediction. Use https://pypi.org/project/kgcnn/4.0.1/ on how to install kgcnn library, hyperparameters used are available in `/dimenet_pp/src/hyperparams.json`. Instructions to augment the dataset as mentioned in manuscript and select the best model as per our framework can be found in `data/README.md` and `model_selection/README.md` respectively. Instructions to train DimeNet++ are given in `/dimenet_pp/src/README.md`.

## Geometry Optimization
Update `gnoa.in` as required with composition, optimal GNN model path, geometry optimization output path and other parameters. Run `python predict_structure.py` for geometry optimization. The final structure predicted by the algorithm is in `best.cif` file in the geometry optimization output directory.

## Performance Evaluation
Code for evaluating the prediction is available in `post_processing`.
Run `python crystal_matcher.py --compound <compound-name>` for performance evaluation. For example, `<compound-name>` can be `NaCl`.