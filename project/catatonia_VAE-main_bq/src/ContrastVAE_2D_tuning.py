import sys
sys.path.append("/home/developer/.local/lib/python3.10/site-packages")
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime
import logging
from pathlib import Path
sys.path.append("../src")
import torch
import time
import ray
from ray import tune
from ray.air import config
from ray.tune.logger import CSVLogger
from torch.cuda.amp import GradScaler
import torchio as tio
import scanpy as sc
import seaborn as sns
import pandas as pd
import umap
# Set non-interactive plotting to avoid tmux crashes
matplotlib.use("Agg")

sys.path.append("../src")

from models.ContrastVAE_2D_dev import (
    NormativeVAE, 
    train_normative_model_plots,
    bootstrap_train_normative_models_plots
)
from utils.support_f import get_all_data, split_df, combine_dfs
from utils.config_utils_model import Config_2D

from module.data_processing_hc import (
    load_checkpoint_model, 
    load_mri_data_2D_all_atlases,
    load_mri_data_2D, 
    process_subjects, 
    train_val_split_annotations,
    train_val_split_subjects
)
    
from utils.logging_utils import (
    log_and_print,
    log_data_loading,
    log_model_ready,
    log_model_setup,
    setup_logging,
    log_atlas_mode
)

def extract_measurements(subjects):
    """Extract measurements from subjects as torch tensors."""
    all_measurements = []
    for subject in subjects:
        all_measurements.append(torch.tensor(subject["measurements"]).squeeze())
    return torch.stack(all_measurements)

def tune_normative_vae(config_dict):
    """
    Tune NormativeVAE with Ray Tune
    
    Args:
        config_dict (Dict): Configuration dictionary containing hyperparameters
    """
    # Unpack hyperparameters for tuning
    learning_rate = config_dict["learning_rate"]
    kldiv_weight = config_dict["kldiv_loss_weight"]
    latent_dim = config_dict["latent_dim"]
    hidden_dim_1 = config_dict["hidden_dim_1"]
    hidden_dim_2 = config_dict["hidden_dim_2"]
    batch_size = config_dict["batch_size"]

    # Set main paths
    path_original = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/metadata_20250110/full_data_train_valid_test.csv"
    path_to_dir = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training"
    TRAIN_CSV, TEST_CSV = split_df(path_original, path_to_dir)
    
    print("Überprüfe Pfade vor Konfigurationserstellung:")
    
    # Pfade vor der Konfigurationserstellung erstellen
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/analysis/tuning_results"
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Spezifischen Pfad erstellen
    specific_tuning_dir = os.path.join(analysis_dir, f"tuning_results_{timestamp}")
    os.makedirs(specific_tuning_dir, exist_ok=True)
    
    # Ausgabe der Pfade zur Überprüfung
    print(f"Analysis Verzeichnis: {analysis_dir}")
    print(f"Spezifisches Tuning-Verzeichnis: {specific_tuning_dir}")
    print(f"Analysis Verzeichnis existiert: {os.path.exists(analysis_dir)}")
    print(f"Spezifisches Tuning-Verzeichnis existiert: {os.path.exists(specific_tuning_dir)}")

    # Setup configuration
    config = Config_2D(
        # General Parameters
        RUN_NAME=f"NormativeVAE_Tuning_{timestamp}",
        # Input / Output Paths
        TRAIN_CSV=["/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training/training_metadata.csv"],
        TEST_CSV=["/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training/testing_metadata.csv"],
        MRI_DATA_PATH_TRAIN="/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/train_xml_data",
        MRI_DATA_PATH_TEST="/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/test_xml_data",
        ATLAS_NAME="all",  # Using all atlases
        PROC_DATA_PATH="/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training/proc_extracted_xml_data",
        OUTPUT_DIR=save_dir,
        # load_mri_data parameters
        VOLUME_TYPE= "Vgm",
        VALID_VOLUME_TYPES=["Vgm", "Vwm", "csf"],
        # Loading Model
        LOAD_MODEL=False,
        # Loss Parameters
        RECON_LOSS_WEIGHT=40.0,
        KLDIV_LOSS_WEIGHT=kldiv_weight,
        # Learning and Regularization
        TOTAL_EPOCHS=50,  # Fixed for tuning
        LEARNING_RATE=learning_rate,
        WEIGHT_DECAY=4e-3,
        # Data Parameters
        BATCH_SIZE=batch_size,
        DIAGNOSES=["HC"],
        # Misc.
        LATENT_DIM=latent_dim,
        SHUFFLE_DATA=True,
        SEED=42
    )

    # Set seed for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.DEVICE = device

    # Load healthy control data
    log_and_print("Loading healthy control data...")
    all_data_paths_train = get_all_data(directory=config.MRI_DATA_PATH_TRAIN, ext="h5")
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

    # Prepare data loaders
    train_data = extract_measurements(train_subjects_hc)
    valid_data = extract_measurements(valid_subjects_hc)

    # Initialize the normative VAE model
    normative_model = NormativeVAE(
        input_dim=len_atlas,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        latent_dim=latent_dim,
        learning_rate=learning_rate,
        kldiv_loss_weight=kldiv_weight,
        dropout_prob=0.1,
        device=device
    )

    # Train and evaluate model
    try:
        baseline_model, baseline_history = train_normative_model_plots(
            train_data=train_data,
            valid_data=valid_data,
            model=normative_model,
            epochs=config.TOTAL_EPOCHS,
            batch_size=batch_size,
            save_best=True,
            return_history=True
        )

        # Report validation loss to Ray Tune
        tune.report(
            val_loss=baseline_history['best_val_loss'], 
            final_val_loss=baseline_history['final_val_loss']
        )

    except Exception as e:
        # Report error to Ray Tune
        tune.report(val_loss=float('inf'), error=str(e))

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Normative VAE Hyperparameter Tuning')
    parser.add_argument('--atlas_name', help='Name of the atlas', default='all')
    return parser

if __name__ == "__main__":
    # Define hyperparameter search space
    search_space = {
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "kldiv_loss_weight": tune.loguniform(1, 10),
        "latent_dim": tune.choice([10, 20, 40, 60]),
        "hidden_dim_1": tune.choice([50, 100, 200]),
        "hidden_dim_2": tune.choice([50, 100, 200]),
        "batch_size": tune.choice([16, 32, 64, 128])
    }

    # Parse arguments
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args()
    atlas_name = parsed_args.atlas_name

    # Set the local_dir with the file:// scheme
    local_dir = f"file:///raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/normative_ray_results"
    os.makedirs(local_dir.replace("file://", ""), exist_ok=True)

    # Configure Ray Tune
    tuner = tune.Tuner(
        tune.with_resources(
            tune_normative_vae,
            resources={"cpu": 8, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            num_samples=-1,
            time_budget_s=3600,  # 1 hour timeout
            max_concurrent_trials=2
        ),
        run_config=tune.RunConfig(
            name=f"normative_vae_tuning_{atlas_name}",
            storage_path=local_dir,
            log_to_file=True,
            stop={"training_iteration": 50}  # Maximum 50 iterations
        ),
        param_space=search_space
    )

    # Run hyperparameter tuning
    analysis = tuner.fit()

    # Get and print best results
    best_result = analysis.get_best_result(metric="val_loss", mode="min")
    print("Best hyperparameters found were: ", best_result.config)
    print("Best validation loss: ", best_result.metrics["val_loss"])

    # Save results to CSV
    results_df = analysis.get_dataframe()
    results_df.to_csv(f"{local_dir.replace('file://', '')}/normative_vae_tuning_results.csv", index=False)