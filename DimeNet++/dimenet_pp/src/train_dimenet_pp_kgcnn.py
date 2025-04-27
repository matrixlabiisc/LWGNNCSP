# Description: Train a dimenetpp model with KGCNN on the dataset
# Author: Rushikesh Pawar (date: 13th dec 2024)
# Reference: https://github.com/aimat-lab/gcnn_keras/blob/master/training/train_graph.py#L16 
import os
import time
import logging
import json
from datetime import timedelta
import argparse

import tensorflow as tf
import keras as ks
import numpy as np
from sklearn.model_selection import train_test_split
import kgcnn.training.scheduler  # noqa
import kgcnn.training.schedule  # noqa
import kgcnn.losses.losses
import kgcnn.metrics.metrics
from kgcnn.data.crystal import CrystalDataset
from kgcnn.metrics.metrics import ScaledMeanAbsoluteError, ScaledRootMeanSquaredError
from kgcnn.data.transform.scaler.serial import deserialize as deserialize_scaler
from kgcnn.utils.plots import plot_train_test_loss, plot_predict_true
from kgcnn.models.serial import deserialize as deserialize_model
from kgcnn.utils.devices import check_device, set_cuda_device
from kgcnn.data.utils import save_pickle_file, load_pickle_file
from keras.saving import deserialize_keras_object
from keras.callbacks import ModelCheckpoint

def getlogger(expt_dir):
    os.makedirs(expt_dir, exist_ok=True)

    logpath = os.path.join(expt_dir, "logfile.log")
    logger = logging.getLogger(expt_dir)
    logger.setLevel(logging.DEBUG)


    #create a file handler
    handler = logging.FileHandler(logpath)
    handler.setLevel(logging.DEBUG)

    #create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    #add the handlers to the logger
    logger.addHandler(handler)

    logger.info("Logging to file: %s" % logpath)
    return logger

def get_args():
    argparser = argparse.ArgumentParser(description="Train a DimeNet++ model with KGCNN on given data")

    argparser.add_argument(
                            "--data_path",
                            type=str,
                            help="Path to the data directory, data dir should have folder named cif_files, id_prop.csv",
                            required=True)
    argparser.add_argument(
                            "--model_name",
                            type=str,
                            help="Name of the model to be trained, eg M1_trial1",
                            required=True)
    argparser.add_argument(
                            "--random_seed",
                            type=int,
                            help="Random seed for training, this will be different for different models of same type",
                            required=True)
    argparser.add_argument(
                            "--hyperparams_path",
                            type=str,
                            help="Path to the hyperparameters json file, this should be a json file",
                            default="./hyperparams.json")
    
    args = argparser.parse_args()
    return args

def run_train(args, hyper):
    # make directories
    expt_dir = hyper["info"]["postfix_file"]
    os.makedirs(expt_dir, exist_ok=True)

    model_dir = os.path.join(expt_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    result_dir = os.path.join(expt_dir, "results")
    os.makedirs(result_dir, exist_ok=True)


    # get logger
    logger = getlogger(expt_dir)
    logger.info(f"Starting expt: {args.model_name} with random seed: {args.random_seed}")

    #set cuda device
    set_cuda_device(0)
    logger.info("Device being used: " + str(check_device()))

    # Enable memory growth for the GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    # set seed
    np.random.seed(hyper["info"]["random_seed"])
    ks.utils.set_random_seed(hyper["info"]["random_seed"])
    logger.info(f"Random seed fixed: {hyper['info']['random_seed']}")

    ########################### Load dataset ############################
    dataset = CrystalDataset(
        data_directory=hyper["data"]["dataset"]["data_directory"],
        dataset_name=hyper["data"]["dataset"]["dataset_name"],
        file_name=hyper["data"]["dataset"]["file_name"],
        file_directory=hyper["data"]["dataset"]["file_directory"],
        file_name_pymatgen_json=None)

    dataset.prepare_data(
        file_column_name="filename",
        overwrite=False,
    )
    # this makes sure that each formation energy is np.array of shape (1,)
    logger.info(f"Applying additional callbacks to dataset")
    dataset.read_in_memory(
        additional_callbacks={"graph_labels": lambda st, ds: np.array([ds["formation_energy"]])})

    logger.info(f"Setting methods to dataset")
    dataset.set_methods(hyper["data"]["dataset"]["methods"])

    # assert dataset has correct format
    dataset.assert_valid_model_input(hyper["model"]["config"]["inputs"])

    logger.info(f"Length of dataset: {len(dataset)}")
    dataset.clean(hyper["model"]["config"]["inputs"])
    len_dataset = len(dataset)
    logger.info(f"Length of cleaned dataset: {len_dataset}")

    # get train test val split
    idx = np.arange(len_dataset)

    train_idx, val_test_idx = train_test_split(idx, test_size=0.3, random_state=hyper["data"]["split_random_seed"])
    val_idx, test_idx = train_test_split(val_test_idx, test_size=0.5, random_state=hyper["data"]["split_random_seed"])
    logger.info(f"using random seed: {hyper['data']['split_random_seed']} for train, val, test split")

    logger.info(f"Train data size: {len(train_idx)}, Val data size: {len(val_idx)}, Test data size: {len(test_idx)}")
    logger.info(f"Train data %: {len(train_idx)*100/len_dataset}, Val data %: {len(val_idx)*100/len_dataset}, Test data %: {len(test_idx)*100/len_dataset}")

    # save split
    np.savez(os.path.join(expt_dir, "data_train_val_test_idx.npz"),
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    logger.info(f"Saved train, val, test split indices at {expt_dir}/data_train_val_test_idx.npz")

    # get data
    data_train = dataset[train_idx]
    data_val = dataset[val_idx]
    data_test = dataset[test_idx]

    # scale data 
    # This only sclaes y (ie, formation energy) and not x (ie, crystal structure)
    scaler = deserialize_scaler(hyper["training"]["scaler"])
    scaler.fit_dataset(data_train)
    data_train = scaler.transform_dataset(data_train, copy_dataset=True, copy=True)
    data_val = scaler.transform_dataset(data_val,copy_dataset=True, copy=True)

    scaler_scale = scaler.get_scaling()
    logger.info(f"Scaler scale: {scaler_scale}")

    mae_metric = ScaledMeanAbsoluteError(scaler_scale.shape, name="scaled_mean_absolute_error")
    rms_metric = ScaledRootMeanSquaredError(scaler_scale.shape, name="scaled_root_mean_squared_error")

    if scaler_scale is not None:
        mae_metric.set_scale(scaler_scale)
        rms_metric.set_scale(scaler_scale)

    scaled_metrics = [mae_metric, rms_metric]
    scaled_predictions = True

    scaler.save(os.path.join(model_dir, f"scaler"))
    logger.info(f"Saved scaler at {model_dir}/scaler")

    # get train and val features and labels

    x_train = data_train.tensor(hyper["model"]["config"]["inputs"])
    y_train = np.array(data_train.get("graph_labels"))

    x_val = data_val.tensor(hyper["model"]["config"]["inputs"])
    y_val = np.array(data_val.get("graph_labels"))

    ########################## create a model ##########################
    dimenetpp_model = deserialize_model(hyper["model"])

    dimenetpp_model.compile(
        optimizer=hyper["training"]["compile"]["optimizer"],
        loss=hyper["training"]["compile"]["loss"],
        metrics=scaled_metrics)

    dimenetpp_model.summary()

    print(" Compiled with jit: %s" % dimenetpp_model._jit_compile)  # noqa
    print(" dimenetpp_model is built: %s, with unbuilt: %s" % (
            all([layer.built for layer in dimenetpp_model._flatten_layers()]),  # noqa
            [layer.name for layer in dimenetpp_model._flatten_layers() if not layer.built]
        ))
    ########################## Start training ##########################
    def deserialize_callbacks(callbacks):
        callbacks_objs = []
        for cb in callbacks:
            if isinstance(cb, (str, dict)):
                callbacks_objs += [deserialize_keras_object(cb)]
            else:
                callbacks_objs += [cb]
        return callbacks_objs

    checkpoints = ModelCheckpoint(
        os.path.join(model_dir, f"best_model_{args.model_name}.weights.h5"),
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,)

    logger.info("Starting training=======================================================================")
    start = time.time()
    hist = dimenetpp_model.fit(
        x = x_train,
        y = y_train,
        batch_size=hyper["training"]["fit"]["batch_size"],
        epochs=hyper["training"]["fit"]["epochs"],
        validation_data=(x_val, y_val),
        validation_freq=hyper["training"]["fit"]["validation_freq"],
        verbose=hyper["training"]["fit"]["verbose"],
        callbacks=[checkpoints],
        shuffle=True,
    )
    end = time.time()
    logger.info(f"Training took {timedelta(seconds=end - start)}")

    ########################## Save model  ##########################
    # save history
    logger.info("Saving history")
    save_pickle_file(hist.history, os.path.join(model_dir, "history.pickle"))
    save_pickle_file(str(timedelta(seconds=end - start)), os.path.join(model_dir, "time.pickle"))

    # save model wights
    dimenetpp_model.save_weights(os.path.join(model_dir, f"last_ep_model_seed_{hyper['info']['random_seed']}.weights.h5"))
    logger.info(f"Saved last epoch model at {model_dir}")
    logger.info(f"Saved last epoch model weights at {model_dir}")

    ########################## Evaluate model & save plots ##########################
    # load test data
    x_test = data_test.tensor(hyper["model"]["config"]["inputs"])  
    y_test = np.array(data_test.get("graph_labels"))    # not scaled

    logger.info("Evaluating last epoch model=======================================================================")
    logger.info("Evaluating on validation set")
    # plot predict true
    y_pred = dimenetpp_model.predict(x_val)
    y_true = y_val

    plot_predict_true(y_pred, y_true, filepath=result_dir,
                    model_name=f"dimenetpp_last_epoch_{args.model_name}",
                    scaled_predictions=scaled_predictions, target_names=["Formation Energy"],
                    file_name=f"predict_val_seed_{hyper['info']['random_seed']}")

    logger.info("Evaluating on test set")

    # plot predict true
    y_pred = dimenetpp_model.predict(x_test)
    y_pred = scaler.inverse_transform(y_pred)

    y_true = y_test
    plot_predict_true(y_pred, y_true, filepath=result_dir,
                    model_name=f"dimenetpp_last_epoch_{args.model_name}",
                    scaled_predictions=False, target_names=["Formation Energy"],
                    file_name=f"predict_test_seed_{hyper['info']['random_seed']}")


    logger.info("Evaluating best model=======================================================================")

    # plot predict true val for best model
    best_model = deserialize_model(hyper["model"])
    best_model.load_weights(os.path.join(model_dir, f"best_model_{args.model_name}.weights.h5"))

    logger.info("Evaluating on validation set")
    y_pred = best_model.predict(x_val)
    y_true = y_val
    plot_predict_true(y_pred, y_true, filepath=result_dir,
                    model_name=f"dimenetpp_best_model_{args.model_name}",
                    scaled_predictions=scaled_predictions, target_names=["Formation Energy"],
                    file_name=f"predict_val_seed_{hyper['info']['random_seed']}")

    # plot predict true test for best model
    logger.info("Evaluating on test set")
    y_pred = best_model.predict(x_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_true = y_test
    plot_predict_true(y_pred, y_true, filepath=result_dir,
                    model_name=f"dimenetpp_best_model_{args.model_name}",
                    scaled_predictions=False, target_names=["Formation Energy"],
                    file_name=f"predict_test_seed_{hyper['info']['random_seed']}")


    logger.info("plotting training curves=======================================================================")
    # plot train test loss
    history = load_pickle_file(os.path.join(model_dir, "history.pickle"))

    plot_train_test_loss([history], model_name=f"dimenetpp_kgcnn",filepath=result_dir,
                        file_name=f"train_val_loss_seed_{hyper['info']['random_seed']}.png",
                        )
    logger.info(f"Plots saved at {result_dir}")

    logger.info("Finished=======================================================================")

def main():
    args = get_args()
    data_path = args.data_path
    model_name = args.model_name
    random_seed = args.random_seed
    hyperparams_path = args.hyperparams_path

    # load hyperparameters
    with open(hyperparams_path, "r") as f:
        hyper = json.load(f)

    # update hyperparameters
    hyper["data"]["dataset"]["data_directory"] = data_path
    hyper["info"]["postfix_file"] = f"../model/dimenet_pp_{model_name}"
    hyper["info"]["random_seed"] = random_seed

    run_train(args, hyper)

if __name__ == "__main__":
    main()