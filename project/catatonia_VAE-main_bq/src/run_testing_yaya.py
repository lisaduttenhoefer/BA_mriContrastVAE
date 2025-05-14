"""
Calculate deviation scores for clinical groups (SCHZ, CTT, MDD) using trained normative VAE models -> testing set: only patients
"""
import argparse
import os
import sys
from pathlib import Path
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import umap

# Ensure we can import from parent directory
sys.path.append("../src")
from models.ContrastVAE_2D_dev import NormativeVAE
from utils.support_f import get_all_data
from utils.config_utils_model import Config_2D
from module.data_processing_hc import (
    load_mri_data_2D,
    load_mri_data_2D_all_atlases,
    process_subjects,
)
from utils.logging_utils import setup_logging_test, log_and_print_test
from utils.dev_scores_utils import (
    calculate_deviations, 
    plot_deviation_distributions, 
    analyze_score_auc, 
    visualize_embeddings, 
    calculate_cliffs_delta, 
    calculate_roi_deviation_scores, 
    plot_diagnosis_deviation_boxplots, 
    calculate_roi_contribution,
)

#--------------------------------------- NECESSARY ARGUMENTS -----------------------------------------------------
model_dir = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/normative_results_20250513_140323" #dir of trained models
clinical_data_path = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/test_xml_data" #dir to clinical data
clinical_csv = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training/testing_metadata.csv" #path to clinical csv with annotations
#-----------------------------------------------------------------------------------------------------------------

def extract_measurements(subjects):
    """Extract measurements from subjects as torch tensors."""
    all_measurements = []
    all_roi_names = []
    
    for subject in subjects:
        all_measurements.append(torch.tensor(subject["measurements"]).squeeze())
        
        # Extract ROI names if present, otherwise use indices
        if "roi_names" in subject and subject["roi_names"] is not None:
            all_roi_names = subject["roi_names"]
    
    # Create a mapping of indices to ROI names if available
    if not all_roi_names:
        all_roi_names = [f"region_{i}" for i in range(all_measurements[0].shape[0])]
    
    return torch.stack(all_measurements), all_roi_names

def main(args):
    # ---------------------- INITIAL SETUP (output dirs, device, seed) --------------------------------------------
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{args.output_dir}/clinical_deviations_{timestamp}" if args.output_dir else f"./clinical_deviations_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/figures", exist_ok=True)
    
    # Set up logging
    log_file = f"{save_dir}/deviation_analysis.log"
    logger = setup_logging_test(log_file=log_file)
    
    # Log start of analysis
    log_and_print_test("Starting deviation analysis for clinical groups")
    log_and_print_test(f"Model directory: {model_dir}")
    log_and_print_test(f"Output directory: {save_dir}")
    
    # Set device
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    log_and_print_test(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available() and not args.no_cuda:
        torch.cuda.manual_seed_all(args.seed)
    
    # ---------------------- LOAD MODEL CONFIG FROM IG TRAINING (cosistency)  --------------------------------------------
    try:
        config_path = os.path.join(model_dir, "config.csv")
        config_df = pd.read_csv(config_path)
        log_and_print_test(f"Loaded model configuration from {config_path}")
        
        # Extract relevant parameters
        atlas_name = config_df["ATLAS_NAME"].iloc[0]
        latent_dim = int(config_df["LATENT_DIM"].iloc[0])
        hidden_dim_1 = 100  # Default if not in config
        hidden_dim_2 = 100  # Default if not in config
        volume_type = config_df["VOLUME_TYPE"].iloc[0] if "VOLUME_TYPE" in config_df.columns else "Vgm"
        valid_volume_types = ["Vgm", "Vwm", "csf"]  # Default if not specified
        
    except (FileNotFoundError, KeyError) as e:
        log_and_print_test(f"Warning: Could not load config file. Using default parameters. Error: {e}")
        atlas_name = args.atlas_name
        latent_dim = args.latent_dim
        hidden_dim_1 = 100
        hidden_dim_2 = 100
        volume_type = "Vgm"
        valid_volume_types = ["Vgm", "Vwm", "csf"]
    
    # ------------------------------------------ LOADING CLINICAL DATA  --------------------------------------------
    log_and_print_test("Loading clinical data...")
    
    # Set paths for clinical data
    path_to_clinical_data = clinical_data_path
    
    # Load clinical data using the same function as for healthy controls
    if atlas_name != "all":
        subjects_clinical, annotations_clinical = load_mri_data_2D(
            csv_paths=[clinical_csv],
            data_path=path_to_clinical_data,
            atlas_name=atlas_name,
            diagnoses=["HC", "SCHZ", "CTT", "MDD"],  # Include all diagnoses
            hdf5=True,
            train_or_test="test",
            save=False,
            volume_type=volume_type,
            valid_volume_types=valid_volume_types,
        )
    else:
        all_data_paths = get_all_data(directory=path_to_clinical_data, ext="h5")
        subjects_clinical, annotations_clinical = load_mri_data_2D_all_atlases(
            csv_paths=[clinical_csv],
            data_paths=all_data_paths,
            diagnoses=["HC", "SCHZ", "CTT", "MDD"],
            hdf5=True,
            train_or_test="test",
            volume_type=volume_type,
            valid_volume_types=valid_volume_types,
        )
    
    # Extract measurements AND ROI names
    clinical_data, roi_names = extract_measurements(subjects_clinical)
    log_and_print_test(f"Clinical data shape: {clinical_data.shape}")
    log_and_print_test(f"Number of ROIs: {len(roi_names)}")
    
    # Get input dimension
    input_dim = clinical_data.shape[1]
    log_and_print_test(f"Input dimension: {input_dim}")
    
    # Count subjects by diagnosis
    diagnosis_counts = annotations_clinical["Diagnosis"].value_counts()
    log_and_print_test(f"Subject counts by diagnosis:\n{diagnosis_counts}")
    
    # Save ROI names for reference
    roi_df = pd.DataFrame({'ROI_Index': range(len(roi_names)), 'ROI_Name': roi_names})
    roi_df.to_csv(f"{save_dir}/roi_names.csv", index=False)
    log_and_print_test(f"Saved ROI names to {save_dir}/roi_names.csv")
    
    # ---------------------- LOAD BOOTSTRAP MODELS (increases robustness)  --------------------------------------------
    log_and_print_test("Loading normative bootstrap models...")
    bootstrap_models = []
    models_dir = os.path.join(model_dir, "models")
    model_files = [f for f in os.listdir(models_dir) if f.startswith("bootstrap_") and f.endswith(".pt")]
    
    if len(model_files) == 0:
        log_and_print_test("No bootstrap models found. Looking for baseline model...")
        if os.path.exists(os.path.join(models_dir, "baseline_model.pt")):
            model_files = ["baseline_model.pt"]
        else:
            raise FileNotFoundError("No models found in the specified directory.")
    
    # Load up to max_models if specified
    if args.max_models > 0:
        model_files = model_files[:args.max_models]
    
    for model_file in model_files:
        # Initialize model architecture
        model = NormativeVAE(
            input_dim=input_dim,
            hidden_dim_1=hidden_dim_1,
            hidden_dim_2=hidden_dim_2,
            latent_dim=latent_dim,
            learning_rate=1e-4,  # Not used for inference
            kldiv_loss_weight=1.0,  # Not used for inference
            dropout_prob=0.0,  # Set to 0 for inference
            device="cpu"  # Will move to appropriate device later
        )
        
        # Load model weights
        model_path = os.path.join(models_dir, model_file)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        bootstrap_models.append(model)
        log_and_print_test(f"Loaded model: {model_file}")
    
    log_and_print_test(f"Successfully loaded {len(bootstrap_models)} models")
    
    # ------------------------------- CALCULATION DEV_SCORES WITH ROI TRACKING --------------------------------------------
    log_and_print_test("Calculating deviation scores with ROI tracking...")
    
    # Use our modified function that maintains ROI names
    results_df = calculate_roi_deviation_scores(
        normative_models=bootstrap_models,
        data_tensor=clinical_data,
        annotations_df=annotations_clinical,
        device=device,
        roi_names=roi_names
    )
    
    # Save deviation scores
    results_df.to_csv(f"{save_dir}/deviation_scores.csv", index=False)
    log_and_print_test(f"Saved deviation scores to {save_dir}/deviation_scores.csv")
    
    # Generate visualizations like in the screenshot
    log_and_print_test("Generating visualizations...")
    
    # Plot deviation distributions for overall metrics
    for metric in ["deviation_score", "reconstruction_error", "kl_divergence"]:
        plot_diagnosis_deviation_boxplots(results_df, metric, save_dir)
    
    # Calculate ROI contributions and create visualizations
    log_and_print_test("Calculating ROI contributions to overall deviation...")
    roi_stats = calculate_roi_contribution(results_df, save_dir, roi_names)
    log_and_print_test(f"ROI contribution analysis complete. Results saved to {save_dir}/roi_contribution.csv")


    # Visualize embeddings in latent space
    log_and_print_test("Visualizing latent space embeddings...")
    embedding_fig, embedding_df = visualize_embeddings(
        normative_models=bootstrap_models,
        data_tensor=clinical_data,
        annotations_df=annotations_clinical,
        device=device
    )
    embedding_fig.savefig(f"{save_dir}/figures/latent_embeddings.png", dpi=300)
    embedding_df.to_csv(f"{save_dir}/latent_embeddings.csv", index=False)
    log_and_print_test("Saved latent space visualizations")
    
    # Create summary for top ROIs by diagnosis
    log_and_print_test("Creating ROI contribution summary by diagnosis...")
    for diagnosis in roi_stats['Diagnosis'].unique():
        if diagnosis == 'HC':  # Skip reference group
            continue
            
        # Get top contributing ROIs for this diagnosis
        diag_rois = roi_stats[roi_stats['Diagnosis'] == diagnosis].sort_values('Cliff_Delta', ascending=False)
        
        # Take top 20 ROIs if available
        top_n = min(20, len(diag_rois))
        top_rois = diag_rois.head(top_n)
        
        log_and_print_test(f"\nTop {top_n} contributing ROIs for {diagnosis}:")
        log_and_print_test(top_rois[['ROI', 'Cliff_Delta', 'ROI_Mean', 'Overall_Mean']])
    
    log_and_print_test(f"Deviation analysis complete! Results saved to {save_dir}")
    return save_dir 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate deviation scores for clinical groups.")
    parser.add_argument("--atlas_name", default=None, help="Atlas name (if not available in config)")
    parser.add_argument("--latent_dim", type=int, default=20, help="Latent dimension (if not available in config)")
    parser.add_argument("--max_models", type=int, default=0, help="Maximum number of models to use (0 = all)")
    parser.add_argument("--output_dir", default=None, help="Output directory for results")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    main(args)