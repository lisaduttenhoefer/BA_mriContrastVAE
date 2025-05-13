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

sys.path.append("../src")
from models.ContrastVAE_2D_f import ContrastVAE_2D
from utils.support_f import get_all_data, split_df, combine_dfs
from utils.config_utils_model import Config_2D

from module.data_processing_hc import (
    load_checkpoint_model, 
    load_mri_data_2D_all_atlases,
    load_mri_data_2D, 
    process_subjects
)
    
from utils.logging_utils import (
    log_and_print,
    log_data_loading,
    setup_logging,
    log_atlas_mode
)

from models.ContrastVAE_2D_dev import (
    NormativeVAE,
    compute_deviation_scores,
    visualize_deviation_scores,
    plot_deviation_maps,
    run_regional_analysis_pipeline,
    get_atlas_regions,
    integrate_regional_analysis,
    identify_top_deviant_regions,
    region_distribution_by_diagnosis,
    create_region_heatmap,
    create_diagnosis_region_heatmap,
    visualize_region_deviations
)

# Use non-interactive plotting to avoid tmux crashes
matplotlib.use("Agg")

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Arguments for Normative Modeling Testing')
    parser.add_argument('--model_dir', help='Directory containing trained normative models', required=True)
    parser.add_argument('--atlas_name', help='Name of the atlas used for training')
    parser.add_argument('--batch_size', help='Batch size', type=int, default=32)
    parser.add_argument('--no_cuda', help='Disable CUDA (use CPU only)', action='store_true')
    parser.add_argument('--output_dir', help='Override default output directory', default=None)
    return parser

def extract_measurements(subjects):
    """Extract measurements from subjects as torch tensors."""
    all_measurements = []
    for subject in subjects:
        all_measurements.append(torch.tensor(subject["measurements"]).squeeze())
    return torch.stack(all_measurements)

def load_bootstrap_models(model_dir, device):
    """Load all saved bootstrap models from directory."""
    model_files = sorted([f for f in os.listdir(f"{model_dir}/models") if f.endswith('.pt')])
    models = []
    
    # Load the training metadata to get model parameters
    metadata = pd.read_csv(f"{model_dir}/training_metadata.csv")
    
    for model_file in model_files:
        # Initialize model with same parameters as training
        model = NormativeVAE(
            input_dim=metadata['input_dim'].iloc[0],
            hidden_dim_1=metadata['hidden_dim_1'].iloc[0],
            hidden_dim_2=metadata['hidden_dim_2'].iloc[0],
            latent_dim=metadata['latent_dim'].iloc[0],
            learning_rate=metadata['learning_rate'].iloc[0],
            kldiv_loss_weight=metadata['kldiv_weight'].iloc[0],
            dropout_prob=0.1,
            device=device
        )
        
        # Load the state dictionary
        model.load_state_dict(torch.load(f"{model_dir}/models/{model_file}", map_location=device))
        model.eval()
        models.append(model)
    
    return models

def main(model_dir: str, atlas_name: str = None, batch_size: int = 32, no_cuda: bool = False, output_dir: str = None):
    ## 0. Set Up ----------------------------------------------------------
    # Load training metadata
    training_metadata = pd.read_csv(f"{model_dir}/training_metadata.csv")
    
    # Use the atlas name from metadata if not provided
    if atlas_name is None:
        atlas_name = training_metadata['atlas_name'].iloc[0]
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        analysis_dir = f"{model_dir}/analysis_{timestamp}"
    else:
        analysis_dir = output_dir
        
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(f"{analysis_dir}/figures", exist_ok=True)
    
    # Set up configuration for analysis
    config = Config_2D(
        # General Parameters
        RUN_NAME=f"NormativeAnalysis_{atlas_name}_{timestamp}",
        # Input / Output Paths
        TRAIN_CSV=["/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training/training_metadata.csv"],
        TEST_CSV=["/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training/testing_metadata.csv"],
        MRI_DATA_PATH_TRAIN="/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/train_xml_data",
        MRI_DATA_PATH_TEST="/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/test_xml_data",
        ATLAS_NAME=atlas_name,
        PROC_DATA_PATH="/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training/proc_extracted_xml_data",
        OUTPUT_DIR=analysis_dir,
        # load_mri_data parameters
        VOLUME_TYPE="Vgm",
        VALID_VOLUME_TYPES=["Vgm", "Vwm", "csf"],
        # Data Parameters
        BATCH_SIZE=batch_size,
        DIAGNOSES=["HC", "CTT", "SCHZ", "MDD"],  # Include all diagnosis groups
    )

    # Set up logging
    setup_logging(config)
    log_and_print(f"Starting normative model analysis for models in: {model_dir}")
    log_and_print(f"Using atlas: {atlas_name}")

    # Set device
    device = torch.device("cpu" if no_cuda or not torch.cuda.is_available() else "cuda")
    log_and_print(f"Using device: {device}")
    config.DEVICE = device

    ## 1. Load Data --------------------------------
    # Load healthy control data for reference
    log_and_print("Loading healthy control data...")
    if config.ATLAS_NAME != "all":
        subjects_hc, annotations_hc = load_mri_data_2D(
            csv_paths=config.TRAIN_CSV,
            data_path=config.MRI_DATA_PATH_TRAIN,
            atlas_name=config.ATLAS_NAME,
            diagnoses=["HC"],
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
            volume_type=config.VOLUME_TYPE,
            valid_volume_types=config.VALID_VOLUME_TYPES,
        )
    else:
        all_data_paths_train = get_all_data(directory=config.MRI_DATA_PATH_TRAIN, ext="h5")
        all_data_paths_test = get_all_data(directory=config.MRI_DATA_PATH_TEST, ext="h5")
        
        subjects_hc, annotations_hc = load_mri_data_2D_all_atlases(
            csv_paths=config.TRAIN_CSV,
            data_paths=all_data_paths_train,
            diagnoses=["HC"],
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
            volume_type=config.VOLUME_TYPE,
            valid_volume_types=config.VALID_VOLUME_TYPES,
        )

    len_atlas = len(subjects_hc[0]["measurements"])
    log_and_print(f"Number of ROIs in atlas: {len_atlas}")

    # Extract features as torch tensors
    hc_data = extract_measurements(subjects_hc)
    patient_data = extract_measurements(subjects_patients)
    
    log_and_print(f"Healthy control data shape: {hc_data.shape}")
    log_and_print(f"Patient data shape: {patient_data.shape}")
    
    # Log the used atlas and the number of ROIs
    log_atlas_mode(atlas_name=config.ATLAS_NAME, num_rois=len_atlas)

    # Log data setup
    log_data_loading(
        datasets={
            "Healthy Controls": len(subjects_hc),
            "Patients": len(subjects_patients),
        }
    )
    
    ## 2. Load Bootstrap Models --------------------------------
    log_and_print("Loading bootstrap models...")
    bootstrap_models = load_bootstrap_models(model_dir, device)
    log_and_print(f"Loaded {len(bootstrap_models)} bootstrap models")
    
    ## 3. Compute Global Deviation Scores --------------------------------
    log_and_print("Computing global deviation scores...")
    
    # Calculate global deviation scores
    hc_global_z, patient_global_z = compute_deviation_scores(
        models=bootstrap_models,
        healthy_data=hc_data,
        patient_data=patient_data,
        device=device
    )
    
    # Save global deviation scores
    np.save(f"{analysis_dir}/hc_global_z_scores.npy", hc_global_z)
    np.save(f"{analysis_dir}/patient_global_z_scores.npy", patient_global_z)
    
    # Boxplot of Z-Scores for both groups
    visualize_deviation_scores(
        hc_scores=hc_global_z,
        patient_scores=patient_global_z,
        save_path=f"{analysis_dir}/figures/global_deviation_scores.png",
        title=f"Global Deviation Scores - {atlas_name}"
    )
    
    ## 4. Regional Analysis --------------------------------
    log_and_print("Performing regional analysis...")
    
    regional_analysis_results = integrate_regional_analysis(
        bootstrap_models=bootstrap_models,
        train_data=hc_data,
        patient_data=patient_data,
        annotations_patients=annotations_patients,
        atlas_name=atlas_name,
        save_dir=analysis_dir
    )
    
    log_and_print(f"Normative model analysis completed successfully!\nResults saved to {analysis_dir}")
    
    return analysis_dir

if __name__ == "__main__":
    # Parse command line arguments
    parser = create_arg_parser()
    args = parser.parse_args()
    
    # Run the main function with parsed arguments
    analysis_dir = main(
        model_dir=args.model_dir,
        atlas_name=args.atlas_name,
        batch_size=args.batch_size,
        no_cuda=args.no_cuda,
        output_dir=args.output_dir
    )
    
    # Final log message
    print(f"Normative model analysis complete. Results saved to {analysis_dir}")