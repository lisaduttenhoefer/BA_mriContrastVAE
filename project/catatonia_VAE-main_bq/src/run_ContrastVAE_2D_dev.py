import sys
sys.path.append("/home/developer/.local/lib/python3.10/site-packages")
import matplotlib
import numpy as np
import pandas as pd
import argparse
import os
from datetime import datetime
import logging

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

# Importiere das normative VAE-Modul - stellen Sie sicher, dass dieser Pfad korrekt ist
from models.ContrastVAE_2D_dev import (
    NormativeVAE, 
    train_normative_model,
    bootstrap_train_normative_models, 
    compute_deviation_scores,
    visualize_deviation_scores,
    plot_deviation_maps,
    run_normative_modeling_pipeline
)

# Use non-interactive plotting to avoid tmux crashes
matplotlib.use("Agg")

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Arguments for Normative Modeling')
    parser.add_argument('--atlas_name', help='Name of the desired atlas for training.')
    parser.add_argument('--num_epochs', help='Number of epochs to be trained for', type=int, default=50)
    parser.add_argument('--n_bootstraps', help='Number of bootstrap samples', type=int, default=100)
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
            save=True
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
            save=True
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
            train_or_test="train"
        )
        
        subjects_patients, annotations_patients = load_mri_data_2D_all_atlases(
            csv_paths=config.TEST_CSV,
            data_paths=all_data_paths_test,
            diagnoses=["CTT", "SCHZ", "MDD"],
            hdf5=True,
            train_or_test="test"
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
    
    # # If we have patient annotations with diagnosis information
    # if "Diagnosis" in annotations_patients.columns:
    #     diagnoses = annotations_patients["Diagnosis"].unique()
        
    #     # Create visualization per diagnosis
    #     diagnosis_scores = {}
    #     for diagnosis in diagnoses:
    #         diagnosis_idx = annotations_patients["Diagnosis"] == diagnosis
    #         diagnosis_patients = np.array(diagnosis_idx)
    #         if np.sum(diagnosis_patients) > 0:
    #             diagnosis_scores[diagnosis] = patient_global_z[diagnosis_patients].numpy()
        
    #     # Plot diagnosis-specific deviation scores
    #     if len(diagnosis_scores) > 1:
    #         plot_deviation_maps(
    #             diagnosis_scores=diagnosis_scores,
    #             hc_scores=hc_global_z,
    #             save_path=f"{save_dir}/figures/diagnosis_deviation_scores.png",
    #             title=f"Deviation Scores by Diagnosis - {atlas_name}"
    #         )

    # After running the normative modeling pipeline, add:
    regional_analysis_results = integrate_regional_analysis(
        bootstrap_models=bootstrap_models,
        train_data=train_data,
        patient_data=patient_data,
        annotations_patients=annotations_patients,
        atlas_name=atlas_name,
        save_dir=save_dir
    )

    log_and_print(f"Normative modeling pipeline completed successfully!\nResults saved to {save_dir}")
    
    
    return save_dir

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



    # Integration in main script
# Add this to the end of your main() function before the return statement

def get_atlas_regions(atlas_name):
    """
    Get region names for a given atlas.
    
    Parameters:
    -----------
    atlas_name : str
        Name of the atlas
        
    Returns:
    --------
    list: List of region names
    """
    # Define atlas region mappings
    atlas_regions = {
        "AAL": [
            "Precentral_L", "Precentral_R", "Frontal_Sup_L", "Frontal_Sup_R", 
            "Frontal_Sup_Orb_L", "Frontal_Sup_Orb_R", "Frontal_Mid_L", "Frontal_Mid_R", 
            # Add more AAL regions here...
        ],
        "Desikan": [
            "ctx-lh-bankssts", "ctx-lh-caudalanteriorcingulate", "ctx-lh-caudalmiddlefrontal",
            "ctx-lh-cuneus", "ctx-lh-entorhinal", "ctx-lh-fusiform", "ctx-lh-inferiorparietal",
            # Add more Desikan regions here...
        ],
        # Add other atlases as needed
    }
    
    # If atlas is not in our mapping, return None and let the function generate generic names
    return atlas_regions.get(atlas_name, None)

# Add this to your main() function before the return statement
def integrate_regional_analysis(bootstrap_models, train_data, patient_data, annotations_patients, 
                              atlas_name, save_dir):
    """
    Integrate the regional analysis into the main script.
    
    Parameters:
    -----------
    bootstrap_models : list
        List of trained normative VAE models from bootstrap samples
    train_data : torch.Tensor
        Tensor of healthy control ROI measurements
    patient_data : torch.Tensor
        Tensor of patient ROI measurements
    annotations_patients : pandas.DataFrame
        DataFrame containing patient metadata including diagnoses
    atlas_name : str
        Name of the brain atlas used
    save_dir : str or Path
        Directory to save results
        
    Returns:
    --------
    None
    """
    import logging
    import numpy as np
    import pandas as pd
    from pathlib import Path
    
    # Create a dictionary to map diagnosis codes to readable labels
    diagnosis_labels = {
        "CTT": "Catatonia",
        "SCHZ": "Schizophrenia",
        "MDD": "Major Depression",
        "HC": "Healthy Control"
    }
    
    # Extract diagnoses from annotations
    if "Diagnosis" in annotations_patients.columns:
        patient_diagnoses = annotations_patients["Diagnosis"].tolist()
    else:
        logging.warning("No 'Diagnosis' column found in patient annotations. Using generic labels.")
        patient_diagnoses = ["Unknown"] * len(patient_data)
    
    # Get region names for the atlas
    atlas_regions = get_atlas_regions(atlas_name)
    
    # Create a new directory for regional analysis
    regional_dir = Path(save_dir) / "regional_analysis"
    regional_dir.mkdir(exist_ok=True)
    
    # Run the regional analysis pipeline
    logging.info(f"Starting regional analysis for atlas: {atlas_name}")
    
    from regional_deviation_analysis import run_regional_analysis_pipeline
    
    analysis_results = run_regional_analysis_pipeline(
        bootstrap_models=bootstrap_models,
        healthy_data=train_data,
        patient_data=patient_data,
        patient_diagnoses=patient_diagnoses,
        atlas_regions=atlas_regions,
        diagnosis_labels=diagnosis_labels,
        save_dir=regional_dir
    )
    
    logging.info(f"Regional analysis completed. Results saved to {regional_dir}")
    
    return analysis_results


# Add the following to your main() function just before the return statement
if __name__ == "__main__":
    # ... (existing code)
    
    # After running the normative modeling pipeline, add:
    regional_analysis_results = integrate_regional_analysis(
        bootstrap_models=bootstrap_models,
        train_data=train_data,
        patient_data=patient_data,
        annotations_patients=annotations_patients,
        atlas_name=atlas_name,
        save_dir=save_dir
    )
    
    # Return statement (keep your existing return)
    return save_dir