import sys
sys.path.append("/home/developer/.local/lib/python3.10/site-packages")
import matplotlib
import numpy as np
import pandas as pd
import argparse
import os
from datetime import datetime
import logging
from pathlib import Path
sys.path.append("../src")
import torch
import torchio as tio
from torch.cuda.amp import GradScaler

sys.path.append("../src")
from models.ContrastVAE_2D_f import ContrastVAE_2D
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

from models.ContrastVAE_2D_dev import (
    NormativeVAE, 
    train_normative_model,
    bootstrap_train_normative_models, 
    compute_deviation_scores,
    visualize_deviation_scores,
    plot_deviation_maps,
    run_normative_modeling_pipeline,
    run_regional_analysis_pipeline,
    get_atlas_regions, 
    integrate_regional_analysis,
    #compute_regional_deviations, 
    identify_top_deviant_regions,
    region_distribution_by_diagnosis, 
    create_region_heatmap,
    create_diagnosis_region_heatmap, 
    visualize_region_deviations,

)

# Use non-interactive plotting to avoid tmux crashes
matplotlib.use("Agg")

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Arguments for Normative Modeling')
    parser.add_argument('--atlas_name', help='Name of the desired atlas for training.')
    parser.add_argument('--num_epochs', help='Number of epochs to be trained for', type=int, default=20)
    parser.add_argument('--n_bootstraps', help='Number of bootstrap samples', type=int, default=50)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=32)
    parser.add_argument('--learning_rate', help='Learning rate', type=float, default=4e-5)
    parser.add_argument('--latent_dim', help='Dimension of latent space', type=int, default=20)
    parser.add_argument('--kldiv_weight', help='Weight for KL divergence loss', type=float, default=4.0)
    parser.add_argument('--save_models', help='Save all bootstrap models', action='store_true')
    parser.add_argument('--no_cuda', help='Disable CUDA (use CPU only)', action='store_true')
    parser.add_argument('--seed', help='Random seed for reproducibility', type=int, default=42)
    return parser

def main(atlas_name: str, num_epochs: int, n_bootstraps: int, batch_size: int, learning_rate: float, 
         latent_dim: int, kldiv_weight: float, save_models: bool, no_cuda: bool, seed: int):
    ## 0. Set Up ----------------------------------------------------------
    # Set main paths
    path_original = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/metadata_20250110/full_data_train_valid_test.csv"
    path_to_dir = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training"
    TRAIN_CSV, TEST_CSV = split_df(path_original, path_to_dir)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        save_dir = f"/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/normative_results_{timestamp}"
    else:
        save_dir = output_dir

    # Set up configuration for the normative modeling
    config = Config_2D(
        # General Parameters
        RUN_NAME=f"NormativeVAE_{atlas_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        # Input / Output Paths
        TRAIN_CSV=["/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training/training_metadata.csv"],
        TEST_CSV=["/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training/testing_metadata.csv"],
        MRI_DATA_PATH_TRAIN="/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/train_xml_data",
        MRI_DATA_PATH_TEST="/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/test_xml_data",
        #METADATA_WHOLE="/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/metadata_20250110/full_data_train_valid_test.csv",
        ATLAS_NAME=atlas_name,
        PROC_DATA_PATH="/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training/proc_extracted_xml_data",
        OUTPUT_DIR="/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/normative_results",
        #load_mri_data
        VOLUME_TYPE= "Vgm",
        VALID_VOLUME_TYPES=["Vgm", "Vwm", "csf"],
        # Loading Model
        LOAD_MODEL=False,
        PRETRAIN_MODEL_PATH=None,
        PRETRAIN_METRICS_PATH=None,
        CONTINUE_FROM_EPOCH=0,
        # Loss Parameters
        RECON_LOSS_WEIGHT=40.0,
        KLDIV_LOSS_WEIGHT=kldiv_weight,
        CONTR_LOSS_WEIGHT=0.0,  # No contrastive loss for normative model
        # Learning and Regularization
        TOTAL_EPOCHS=num_epochs,
        LEARNING_RATE=learning_rate,
        WEIGHT_DECAY=4e-3,
        EARLY_STOPPING=True,
        STOP_LEARNING_RATE=4e-8,
        SCHEDULE_ON_VALIDATION=True,
        SCHEDULER_PATIENCE=6,
        SCHEDULER_FACTOR=0.5,
        # Visualization
        CHECKPOINT_INTERVAL=5,
        DONT_PLOT_N_EPOCHS=0,
        UMAP_NEIGHBORS=30,
        UMAP_DOT_SIZE=20,
        METRICS_ROLLING_WINDOW=10,
        # Data Parameters
        BATCH_SIZE=batch_size,
        DIAGNOSES=["HC"],  # For normative modeling, we train on healthy controls
        # Misc.
        LATENT_DIM=latent_dim,
        SHUFFLE_DATA=True,
        SEED=seed
    )

    hidden_dim_1 = 100
    hidden_dim_2 = 100

    # Set up logging
    setup_logging(config)
    log_and_print(f"Starting normative modeling with atlas: {atlas_name}, epochs: {num_epochs}, bootstraps: {n_bootstraps}")

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{config.OUTPUT_DIR}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/models", exist_ok=True)
    os.makedirs(f"{save_dir}/figures", exist_ok=True)

    # Save configuration
    config_dict = vars(config)
    config_df = pd.DataFrame([config_dict])
    config_df.to_csv(f"{save_dir}/config.csv", index=False)
    log_and_print(f"Configuration saved to {save_dir}/config.csv")

    # Set seed for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available() and not no_cuda:
        torch.cuda.manual_seed_all(config.SEED)

    # Set device
    device = torch.device("cpu" if no_cuda or not torch.cuda.is_available() else "cuda")
    log_and_print(f"Using device: {device}")
    config.DEVICE = device

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
        
        
        # Load patient data (non-HC)
        log_and_print("Loading patient data...")
        subjects_patients, annotations_patients = load_mri_data_2D(
            csv_paths=config.TEST_CSV,
            data_path=config.MRI_DATA_PATH_TEST,
            atlas_name=config.ATLAS_NAME,
            diagnoses=["CTT", "SCHZ", "MDD"],  # Add all patient diagnoses you want to test
            hdf5=True,
            train_or_test="test",
            save=True,
            volume_type= config.VOLUME_TYPE,
            valid_volume_types= config.VALID_VOLUME_TYPES,
        )

    else:
        all_data_paths_train = get_all_data(directory=config.MRI_DATA_PATH_TRAIN, ext="h5")
        all_data_paths_test = get_all_data(directory=config.MRI_DATA_PATH_TEST, ext="h5")

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
        
        subjects_patients, annotations_patients = load_mri_data_2D_all_atlases(
            csv_paths=config.TEST_CSV,
            data_paths=all_data_paths_test,
            diagnoses=["CTT", "SCHZ", "MDD"],
            hdf5=True,
            train_or_test="test", 
            volume_type= config.VOLUME_TYPE,
            valid_volume_types= config.VALID_VOLUME_TYPES,
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
            #"Augmented": "bool",
            "Filename": "category",
            #"OG_Filename": "category",
        }
    )

    log_and_print(annotations)

    # Prepare data loaders
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
    
    patient_loader = process_subjects(
        subjects=subjects_patients,
        batch_size=config.BATCH_SIZE,
        shuffle_data=False,
    )

    # Log the used atlas and the number of ROIs
    log_atlas_mode(atlas_name=config.ATLAS_NAME, num_rois=len_atlas)

    # Log data setup
    log_data_loading(
        datasets={
            "Training Data (HC)": len(train_subjects_hc),
            "Validation Data (HC)": len(valid_subjects_hc),
            "Patient Data": len(subjects_patients),
        }
    )

    ## 2. Prepare and Run Normative Modeling Pipeline --------------------------------
    # Initialize Model
    log_model_setup()
    
    # Create a list to store all models for bootstrapping
    bootstrap_models = []
    
    # Extract features as torch tensors
    # Convert data to tensors
    def extract_measurements(subjects):
        all_measurements = []
        for subject in subjects:
            #all_measurements.append(subject["measurements"].squeeze())
            all_measurements.append(torch.tensor(subject["measurements"]).squeeze())
        return torch.stack(all_measurements)
    
    # Extract measurements
    train_data = extract_measurements(train_subjects_hc)
    valid_data = extract_measurements(valid_subjects_hc)
    patient_data = extract_measurements(subjects_patients)
    
    log_and_print(f"Training data shape: {train_data.shape}")
    log_and_print(f"Validation data shape: {valid_data.shape}")
    log_and_print(f"Patient data shape: {patient_data.shape}")
    
    # Initialize the normative VAE model
    normative_model = NormativeVAE(
        input_dim=len_atlas,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        latent_dim=config.LATENT_DIM,
        learning_rate=config.LEARNING_RATE,
        kldiv_loss_weight=config.KLDIV_LOSS_WEIGHT,
        dropout_prob=0.1,
        device=device
    )
    
    log_model_ready(normative_model)
    
    log_and_print("Running normative modeling pipeline...")
    # Run the normative modeling pipeline
    bootstrap_models, hc_global_z, patient_global_z = run_normative_modeling_pipeline(
        healthy_data=train_data,
        patient_data=patient_data,
        contrastvae=None,  # No pre-trained ContrastVAE
        n_bootstraps=n_bootstraps,
        train_epochs=num_epochs,
        batch_size=config.BATCH_SIZE,
        save_dir=save_dir,
        #device=device,
        save_models=save_models
    )
    
    # Create additional visualizations
    log_and_print("Creating additional visualizations...")
    
    # Save global deviation scores
    
    np.save(f"{save_dir}/hc_global_z_scores.npy", hc_global_z)
    np.save(f"{save_dir}/patient_global_z_scores.npy", patient_global_z)
    
    # Boxplot of Z-Scores for both groups
    visualize_deviation_scores(
        hc_scores=hc_global_z,
        patient_scores=patient_global_z,
        save_path=f"{save_dir}/figures/global_deviation_scores.png",
        title=f"Global Deviation Scores - {atlas_name}"
    )
    
  
    # Normative Modellierung ist fertig — jetzt regionale Analyse:
    regional_analysis_results = integrate_regional_analysis(
        bootstrap_models=bootstrap_models,
        train_data=train_data,
        patient_data=patient_data,
        annotations_patients=annotations_patients,
        atlas_name=atlas_name,
        save_dir=save_dir
    )

    log_and_print(f"Normative modeling pipeline completed successfully!\nResults saved to {save_dir}")
    


if __name__ == "__main__": 
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    
    atlas_name = str(parsed_args.atlas_name) if parsed_args.atlas_name else "AAL"
    num_epochs = int(parsed_args.num_epochs)
    n_bootstraps = int(parsed_args.n_bootstraps)
    batch_size = int(parsed_args.batch_size)
    learning_rate = float(parsed_args.learning_rate)
    latent_dim = int(parsed_args.latent_dim)
    kldiv_weight = float(parsed_args.kldiv_weight)
    save_models = parsed_args.save_models
    no_cuda = parsed_args.no_cuda
    seed = int(parsed_args.seed)
    
    main(
        atlas_name=atlas_name, 
        num_epochs=num_epochs, 
        n_bootstraps=n_bootstraps, 
        batch_size=batch_size, 
        learning_rate=learning_rate, 
        latent_dim=latent_dim, 
        kldiv_weight=kldiv_weight, 
        save_models=save_models, 
        no_cuda=no_cuda, 
        seed=seed
    )

