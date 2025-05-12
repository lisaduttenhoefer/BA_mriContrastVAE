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
from datetime import datetime
import ray
import torch
from ray import tune
from ray.air import config
from ray.train import RunConfig
from ray.tune.logger import CSVLogger
from torch.cuda.amp import GradScaler

from models.ContrastVAE_2D_f import ContrastVAE_2D, train_ContrastVAE_2D
from utils.config_utils_model import Config_2D
from utils.support_f import get_all_data, split_df
from module.data_processing_hc import (
    load_checkpoint_model, 
    load_mri_data_2D, 
    load_mri_data_2D_all_atlases, 
    process_subjects,
    train_val_split_annotations,
    train_val_split_subjects,
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
def tune_ContrastVAE(
    config:Dict[str, Any]
):  

    # Unpack tuning Parameters for model training
    learning_rate = config["learning_rate"]
    kldiv_loss_weight = config["kldiv_loss_weight"]
    latent_dim = config["latent_dim"]
    weight_decay = config["weight_decay"]
    contr_loss_weight = config["contr_loss_weight"]
    recon_loss_weight = config["recon_loss_weight"]
    # Set default output directory and other parameters
    output_dir = config.get('output_dir', None)
    atlas_name = config.get('atlas_name', 'all')
    num_epochs = config.get('num_epochs', 50)

    # setup paths to be absolute paths
    base_dir = os.path.dirname(__file__)  # Directory where the script is located

    # Set main paths
    path_original = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/metadata_20250110/full_data_train_valid_test.csv"
    path_to_dir = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training"
    TRAIN_CSV, TEST_CSV = split_df(path_original, path_to_dir)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        save_dir = f"/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/analysis/tuning_results/tuning_results_{timestamp}"
    else:
        save_dir = output_dir

    # make a unique run name
    run_name = f"ContrastVAE_tuning_{learning_rate}"

    # Config has default parameters you may want to check

    ## 1. Load Data and Initialize Model --------------------------------
    
    config = Config_2D(
        # General Parameters
        RUN_NAME=run_name,
        TRAIN_CSV=["/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training/training_metadata.csv"],
        TEST_CSV=["/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training/testing_metadata.csv"],
        MRI_DATA_PATH_TRAIN="/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/train_xml_data",
        MRI_DATA_PATH_TEST="/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/test_xml_data",
        # Input / Output Paths
        #MRI_DATA_PATH="/raid/nschmidt/project/catatonia_VAE_2D/data/raw_extracted_xml_data/train_xml_data", # This is the h5 file!  
        ATLAS_NAME=atlas_name,
        PROC_DATA_PATH="/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training/proc_extracted_xml_data",
        OUTPUT_DIR=save_dir,
        VOLUME_TYPE= "Vgm",
        VALID_VOLUME_TYPES=["Vgm", "Vwm", "csf"],
        # Loading Model
        LATENT_DIM=latent_dim,
        LOAD_MODEL=False,
        PRETRAIN_MODEL_PATH=None,
        PRETRAIN_METRICS_PATH=None,
        CONTINUE_FROM_EPOCH=0,
        # Loss Parameters
        RECON_LOSS_WEIGHT=recon_loss_weight,
        KLDIV_LOSS_WEIGHT=kldiv_loss_weight,
        CONTR_LOSS_WEIGHT=contr_loss_weight,
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
        DIAGNOSES=["HC"],
        # COVARS=["Dataset"],
        # Misc.
        SHUFFLE_DATA=False,
        SEED=42
        )

    # Set seed for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    accuracy_sum = 0

    all_output_dir = config.OUTPUT_DIR

    # Set up logging
    setup_logging(config)

    print(f"Currently training model on fold [{i+1}/{len(train_indices)}]")

    ## 1. Load Data --------------------------------
    # Load healthy control data
    log_and_print("Loading healthy control data...")
    if config.ATLAS_NAME != "all":
        subjects_hc, annotations_hc = load_mri_data_2D(
            csv_paths=config.TRAIN_CSV,
            data_path=config.MRI_DATA_PATH_TRAIN,
            atlas_name=config.ATLAS_NAME,
            diagnoses=["HC"],  # Only healthy controls for normative model
            hdf5=True,
            train_or_test="train",
            save=True,
            volume_type=config.VOLUME_TYPE,
            valid_volume_types=config.VALID_VOLUME_TYPES,
        )
    else:
        all_data_paths_train = get_all_data(directory=config.MRI_DATA_PATH_TRAIN, ext="h5")

        # We assume there's a function load_mri_data_2D_all_atlases similar to the one in your original code
        subjects_hc, annotations_hc = load_mri_data_2D_all_atlases(
            csv_paths=config.TRAIN_CSV,
            data_paths=all_data_paths_train,
            diagnoses=config.DIAGNOSES,
            hdf5=True,
            train_or_test="train",
            volume_type=config.VOLUME_TYPE,
            valid_volume_types=config.VALID_VOLUME_TYPES,
        )

    len_atlas = len(subjects_hc[0]["measurements"])
    log_and_print(f"Number of ROIs in atlas: {len_atlas}")

    # Split healthy controls into train and validation
    train_annotations_hc, valid_annotations_hc = train_val_split_annotations(
        annotations=annotations_hc, 
        diagnoses=["HC"]
    )
    
    train_subjects_hc, valid_subjects_hc = train_val_split_subjects(
        subjects=subjects_hc, 
        train_ann=train_annotations_hc, 
        val_ann=valid_annotations_hc
    )

    train_annotations_hc.insert(1, "Data_Type", "train")
    valid_annotations_hc.insert(1, "Data_Type", "valid")

    annotations = pd.concat([train_annotations_hc, valid_annotations_hc])
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


    train_loader_hc = process_subjects(
        subjects=train_subjects_hc,
        batch_size=config.BATCH_SIZE,
        shuffle_data=config.SHUFFLE_DATA,
    )

    valid_loader_hc = process_subjects(
        subjects=valid_subjects_hc,
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
            "Training Data": len(train_subjects_hc),
            "Validation Data": len(valid_subjects_hc),
        }
    )

    # Initialize Model
    log_model_setup()

    model = ContrastVAE_2D(
        num_classes=len(config.DIAGNOSES),
        learning_rate=learning_rate,
        weight_decay=config.WEIGHT_DECAY,
        device=config.DEVICE,
        scaler=GradScaler(),
        kldiv_loss_weight=kldiv_loss_weight,
        recon_loss_weight=recon_loss_weight,
        contr_loss_weight=contr_loss_weight,
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
                "t_contr_loss": [],
                "t_recon_loss": [],
                "t_kldiv_loss": [],
                "valid_loss": [],
                "v_contr_loss": [],
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
    best_model_accuracy = train_ContrastVAE_2D(
        model=model,
        train_loader=train_loader_hc,
        valid_loader=valid_loader_hc,
        annotations=annotations,
        model_metrics=model_metrics,
        config=config,
        # accuracy_tune= False,
        no_plotting=True,  # Added to selectively prevent the computation and generation of UMAP at every epoch.
        no_val_plotting=False,  # Added to selectively prevent the generation of training metric plots. 
        no_saving=True  # Added to selectively prevent the saving of every checkpoints model.
    )
    accuracy_sum += best_model_accuracy

    #fold_accuracy = best_model_accuracy
    #k_avg_accuracy_so_far = accuracy_sum / (i + 1)
    # # Report both current fold accuracy and running average
        # tune.report({
        #     "k_avg_accuracy": k_avg_accuracy_so_far,
        #     "fold_accuracy": fold_accuracy,
        #     "completed_folds": i + 1
        # })

    # Final average accuracy across all folds
    #k_avg_accuracy = accuracy_sum / k_fold
    
    # Final report with the complete k-fold cross-validation result
    tune.report({
        "accuracy": best_model_accuracy,
    })

    print(f"Model accuracy: {best_model_accuracy}")
    


if __name__ == "__main__": 
    # Define search space
    
    path_all = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/metadata_20250110/full_data_train_valid_test.csv"
    all_annotations = pd.read_csv(path_all, header=[0])

    #k_fold = 5

    # train_indices, valid_indices = cross_validation_split(path_original=path_all,
    #                                                       path_to_dir="./data/relevant_metadata",
    #                                                       k=k_fold)
    
    train_val_annotations = pd.read_csv("/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training/training_metadata.csv", index_col=0)

    search_space = {
        "learning_rate": tune.loguniform(1e-5, 5e-4),
        "kldiv_loss_weight": tune.loguniform(1, 12),
        "latent_dim": tune.choice([10, 20, 40, 60]),
        "weight_decay": tune.loguniform(1e-4, 1e-2),
        "contr_loss_weight": tune.loguniform(1, 8),
        "recon_loss_weight": tune.loguniform(10, 40),
    }

    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    atlas_name = parsed_args.atlas_name
    atlas_name = str(atlas_name)
    num_epochs = parsed_args.num_epochs
    num_epochs = int(num_epochs)

    # Set the local_dir with the file:// scheme
    local_dir = f"file:///raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/analysis/tuning_results"

    os.makedirs(f"/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/analysis/tuning_results", exist_ok=True)

    tuner = tune.Tuner(
        tune.with_resources(
            tune_ContrastVAE,
            resources={"cpu": 10, "gpu": 2}
            ),
        tune_config=tune.TuneConfig(
                time_budget_s=18000,
                num_samples=-1,
                metric="accuracy",
                mode="max",
                max_concurrent_trials=1
            ),
        run_config=tune.RunConfig(
            name=atlas_name,
            storage_path=local_dir,
            log_to_file=True,
            stop={"accuracy":0.90},
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
    print(f"Best hyperparameters found were: {best_result.config}")
    print(f"Best accuracy was: {best_result.metrics['accuracy']}")
    print()
    print()

    print(analysis.get_dataframe(filter_metric="k_avg_accuracy", filter_mode="max"))

    results = analysis.get_dataframe(filter_metric="k_avg_accuracy", filter_mode="max")
    results.to_csv(f"/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/analysis/tuning_results/ray_results_{atlas_name}.csv")