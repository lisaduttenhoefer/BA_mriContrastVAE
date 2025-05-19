import sys

import matplotlib
import numpy as np
import pandas as pd
import argparse
import datetime as dt

sys.path.append("../src")
import os
import sys
import time
from typing import Any, Dict

import ray
import torch
from ray import tune
from ray.air import config
from ray.tune.logger import CSVLogger
from torch.cuda.amp import GradScaler

from models.ClassVAE_2D import ClassVAE_2D, train_ClassVAE_2D
from utils.config_utils_2D import Config_2D
from utils.helper_functions_2D import get_all_data, split_df
from utils.data_processing_2D import (
    load_checkpoint_model, 
    load_mri_data_2D, 
    load_mri_data_2D_all_atlases, 
    process_subjects,
    train_val_split_annotations,
    train_val_split_subjects
    )
from utils.logging_utils import (
    log_and_print,
    log_data_loading,
    log_atlas_mode,
    log_model_ready,
    log_model_setup,
    setup_logging,
    )

## To-Do: Think about wether ALL plotting should be deactivated during tuning.


# Use non-interactive plotting to avoid tmux crashes
matplotlib.use("Agg")

def create_arg_parser():  # For training several different models with run_training.sh file
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('atlas_name', help='Name of the desired atlas for training.')
    parser.add_argument('num_epochs', help='Number of epochs to be trained for')
    return parser

# Define tuning function
def tune_ClassVAE(
    config:Dict[str, Any]
):  

    # Unpack tuning Parameters for model training
    learning_rate = config["learning_rate"]
    kldiv_loss_weight = config["kldiv_loss_weight"]
    latent_dim = config["latent_dim"]
    weight_decay = config["weight_decay"]
    class_loss_weight = config["class_loss_weight"]
    recon_loss_weight = config["recon_loss_weight"]

    # setup paths to be absolute paths
    base_dir = os.path.dirname(__file__)  # Directory where the script is located

    # make a unique run name
    run_name = f"ClassVAE_tuning_{learning_rate}"

    # Config has default parameters you may want to check
    config = Config_2D(
        # General Parameters
        RUN_NAME=run_name,
        # Input / Output Paths
        TEST_CSV=["/raid/nschmidt/project/catatonia_VAE_2D/data/relevant_metadata/testing_metadata.csv"],  # For WhiteCAT, NSS, NU metadata
        TRAIN_CSV=["/raid/nschmidt/project/catatonia_VAE_2D/data/relevant_metadata/training_metadata.csv"],  # For all other metadata
        MRI_DATA_PATH="/raid/nschmidt/project/catatonia_VAE_2D/data/raw_extracted_xml_data/train_xml_data", # This is the h5 file!  
        ATLAS_NAME=atlas_name,
        PROC_DATA_PATH="/raid/nschmidt/project/catatonia_VAE_2D/data/proc_extracted_xml_data",
        OUTPUT_DIR="/raid/nschmidt/project/catatonia_VAE_2D/analysis",
        # Loading Model
        LATENT_DIM=latent_dim,
        LOAD_MODEL=False,
        PRETRAIN_MODEL_PATH=None,
        PRETRAIN_METRICS_PATH=None,
        CONTINUE_FROM_EPOCH=0,
        # Loss Parameters
        RECON_LOSS_WEIGHT=recon_loss_weight,
        KLDIV_LOSS_WEIGHT=kldiv_loss_weight,
        CLASS_LOSS_WEIGHT=class_loss_weight,
        # USE_SSIM=False,  # if TRUE memory reqs. increase. Halve batch size.
        # Learning and Regularization
        TOTAL_EPOCHS=num_epochs,
        LEARNING_RATE=learning_rate,
        WEIGHT_DECAY=weight_decay,
        EARLY_STOPPING=True,
        STOP_LEARNING_RATE=4e-8,
        SCHEDULE_ON_VALIDATION=True,  # regularization?
        SCHEDULER_PATIENCE=6,
        SCHEDULER_FACTOR=0.5,
        # Visualization
        CHECKPOINT_INTERVAL=5,
        DONT_PLOT_N_EPOCHS=0,
        UMAP_NEIGHBORS=30,
        UMAP_DOT_SIZE=20,
        METRICS_ROLLING_WINDOW=10,
        # Data Parameters
        BATCH_SIZE=64,
        DIAGNOSES=["HC", "MDD", "SCHZ"],
        # COVARS=["Dataset"],
        # Misc.
        SHUFFLE_DATA=False
    )

    # Set up logging
    setup_logging(config)

    # Set seed for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    ## 1. Load Data and Initialize Model --------------------------------
    # Load train data
    if config.ATLAS_NAME != "all":
        subjects, annotations = load_mri_data_2D(
            csv_paths=config.TRAIN_CSV,
            data_path=config.MRI_DATA_PATH,
            atlas_name=atlas_name,
            # covars=config.COVARS,
            diagnoses=config.DIAGNOSES,
            hdf5=True,
            train_or_test="train",
            save=False
        )
    else:
        all_data_paths = get_all_data(directory=config.MRI_DATA_PATH, ext="h5")
        
        subjects, annotations = load_mri_data_2D_all_atlases(
            csv_paths=config.TRAIN_CSV,
            data_paths=all_data_paths,
            # covars=config.COVARS,
            diagnoses=config.DIAGNOSES,
            hdf5=True,
            train_or_test="train"
        )

    len_atlas = len(subjects[0]["measurements"])

    train_annotations, valid_annotations = train_val_split_annotations(annotations=annotations, diagnoses=config.DIAGNOSES)

    train_subjects, valid_subjects = train_val_split_subjects(subjects=subjects, train_ann=train_annotations, val_ann=valid_annotations)

    train_annotations.insert(1, "Data_Type", "train")  # Inserts a new column called "Data_Type" with "train" for values at the second position.
    valid_annotations.insert(1, "Data_Type", "valid")  # Inserts a new column called "Data_Type" with "valid" for values at the second position.

    annotations = pd.concat([train_annotations, valid_annotations])  # Stacks the two data frames again after splitting.
    annotations.sort_values(by=["Data_Type", "Filename"], inplace=True)
    annotations.reset_index(drop=True, inplace=True)

    annotations = annotations.astype(
        {
            "Age": "float",
            "Dataset": "category",
            "Diagnosis": "category",
            "Sex": "category",
            "Data_Type": "category",
            "Filename": "category",
        }
    )

    log_and_print(annotations)


    train_loader = process_subjects(
        subjects=train_subjects,
        batch_size=config.BATCH_SIZE,
        shuffle_data=config.SHUFFLE_DATA,
    )

    valid_loader = process_subjects(
        subjects=valid_subjects,
        batch_size=config.BATCH_SIZE,
        shuffle_data=False,
    )

    # Log the used atlas and the number of ROIs
    log_atlas_mode(atlas_name=config.ATLAS_NAME, 
                num_rois=len_atlas
                )

    # Log data setup
    log_data_loading(
        datasets={
            "Training Data": len(train_subjects),
            "Validation Data": len(valid_subjects),
        }
    )

    # Initialize Model
    log_model_setup()
    model = ClassVAE_2D(
        num_classes=len(config.DIAGNOSES),
        learning_rate=learning_rate,
        weight_decay=config.WEIGHT_DECAY,
        device=config.DEVICE,
        scaler=GradScaler(),
        kldiv_loss_weight=kldiv_loss_weight,
        recon_loss_weight=recon_loss_weight,
        class_loss_weight=class_loss_weight,
        latent_dim=latent_dim,
        schedule_on_validation=config.SCHEDULE_ON_VALIDATION,
        scheduler_patience=config.SCHEDULER_PATIENCE,
        scheduler_factor=config.SCHEDULER_FACTOR,
        input_dim=len_atlas
    )

    if config.LOAD_MODEL:
        model = load_checkpoint_model(model, config.PRETRAIN_MODEL_PATH)
        model_metrics = pd.read_csv(config.PRETRAIN_METRICS_PATH)
    else:
        model_metrics = pd.DataFrame(
            {
                "train_loss": [],
                "t_class_loss": [],
                "t_recon_loss": [],
                "t_kldiv_loss": [],
                "valid_loss": [],
                "v_class_loss": [],
                "v_recon_loss": [],
                "v_kldiv_loss": [],
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1-score": [],
                "learning_rate": [],
            }
        )

    log_model_ready(model)

    ## 2. Train Model --------------------------------------------------
    train_ClassVAE_2D(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        annotations=annotations,
        model_metrics=model_metrics,
        config=config,
        accuracy_tune= True,
        no_plotting=True,  # Added to selectively prevent the computation and generation of UMAP at every epoch.
        no_val_plotting=False,  # Added to selectively prevent the generation of training metric plots. 
        no_saving=True  # Added to selectively prevent the saving of every checkpoints model.
    )

if __name__ == "__main__": 
    # Define search space
    search_space = {
        "learning_rate": tune.loguniform(1e-5, 5e-4),
        "kldiv_loss_weight": tune.loguniform(1, 12),
        "latent_dim": tune.choice([10, 20, 40, 60]),
        "weight_decay": tune.loguniform(1e-4, 1e-2),
        "class_loss_weight": tune.loguniform(1, 8),
        "recon_loss_weight": tune.loguniform(10, 40),
    }

    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    atlas_name = parsed_args.atlas_name
    atlas_name = str(atlas_name)
    num_epochs = parsed_args.num_epochs
    num_epochs = int(num_epochs)

    # Set the local_dir with the file:// scheme
    local_dir = f"file:///raid/nschmidt/project/catatonia_VAE_2D/analysis/ray_results"

    os.makedirs(f"/raid/nschmidt/project/catatonia_VAE_2D/analysis/ray_results", exist_ok=True)

    tuner = tune.Tuner(
        tune.with_resources(
            tune_ClassVAE,
            resources={"cpu": 10, "gpu": 2}
            ),
        tune_config=tune.TuneConfig(
                time_budget_s=3600,
                num_samples=-1,
                metric="accuracy",
                mode="max",
                max_concurrent_trials=1
            ),
        run_config=tune.RunConfig(
            name=atlas_name,
            storage_path=local_dir,
            log_to_file=True,
            stop={"accuracy":0.85},
            ),
        param_space=search_space
    )
        
    analysis = tuner.fit()

    for i in range(len(analysis)):
        result = analysis[i]
    if not result.error:
            print(f"Trial finishes successfully with metrics"
               f"{result.metrics}.")
    else:
            print(f"Trial failed with error {result.error}.")

    # print best hyperparameters
    time.sleep(10)

    best_result = analysis.get_best_result(metric="accuracy", mode="max")
    print(f"Best hyperparameters for atlas {atlas_name} found were: ", best_result )
    # print("Best accuracy for atlas {atlas_name} was: ", best_result.checkpoint)
    print()
    print()

    print(analysis.get_dataframe(filter_metric="accuracy", filter_mode="max"))

    results = analysis.get_dataframe(filter_metric="accuracy", filter_mode="max")
    results.to_csv(f"/raid/nschmidt/project/catatonia_VAE_2D/analysis/ray_results/ray_results_{atlas_name}.csv")