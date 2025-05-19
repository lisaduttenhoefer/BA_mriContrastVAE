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
import umap

# Ensure we can import from parent directory
sys.path.append("../src")
from models.ContrastVAE_2D_dev import NormativeVAE
from utils.support_f import get_all_data
from utils.config_utils_model import Config_2D
from module.data_processing_hc import (
    load_mri_data_2D,
    load_mri_data_2D_all_atlases,
    #process_subjects,
)
from utils.logging_utils import setup_logging_test, log_and_print_test, end_logging
from utils.dev_scores_utils import (
    calculate_deviations, 
    plot_deviation_distributions, 
    visualize_embeddings, 
    analyze_regional_deviations,
    extract_roi_names,
)

#--------------------------------------- NECESSARY ARGUMENTS -----------------------------------------------------
model_dir = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/analysis/TRAINIG/normative_results_20250519_082729SCHZ" #dir of trained models (has to match NORM & RATIO)
clinical_data_path = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/all_xml_data" #dir to clinical data -> ALL
clinical_csv = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training/testing_metadata.csv" #path to metadata (has to match NORM & RATIO)
#-----------------------------------------------------------------------------------------------------------------
def extract_measurements(subjects):
    """Extract measurements from subjects as torch tensors."""
    all_measurements = []
    for subject in subjects:
        all_measurements.append(torch.tensor(subject["measurements"]).squeeze())
    return torch.stack(all_measurements)


def main(args):
    # ---------------------- INITIAL SETUP (output dirs, device, seed) --------------------------------------------
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{args.output_dir}/clinical_deviations_{timestamp}" if args.output_dir else f"/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/analysis/TESTING/clinical_deviations_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/figures", exist_ok=True)

    #NORM DIAGNOSIS
    norm_diagnosis = args.norm_diagnosis

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
    
    #######
    # Determine h5_file_path for ROI name extraction before loading subjects
    if atlas_name != "all":
        h5_file_path = os.path.join(path_to_clinical_data, f"{atlas_name}.h5")
    else:
        all_data_paths = get_all_data(directory=path_to_clinical_data, ext="h5")
        if all_data_paths:
            h5_file_path = all_data_paths[0]
        else:
            h5_file_path = None
            log_and_print_test("Warning: No HDF5 files found for ROI name extraction")
    

    ########

    # Load clinical data using the same function as for healthy controls
    if atlas_name != "all":
        subjects_clinical, annotations_clinical,roi_names = load_mri_data_2D(
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
        subjects_clinical, annotations_clinical, roi_names = load_mri_data_2D_all_atlases(
            csv_paths=[clinical_csv],
            data_paths=all_data_paths,
            diagnoses=["HC", "SCHZ", "CTT", "MDD"],
            hdf5=True,
            train_or_test="test",
            volume_type=volume_type,
            valid_volume_types=valid_volume_types,
        )
    
    # Extract measurements
    clinical_data = extract_measurements(subjects_clinical)
    log_and_print_test(f"Clinical data shape: {clinical_data.shape}")
    
    # Get input dimension
    input_dim = clinical_data.shape[1]
    log_and_print_test(f"Input dimension: {input_dim}")
    
    #######
    # Verify ROI names match the data dimension
    if roi_names is not None and len(roi_names) != input_dim:
        log_and_print_test(f"Warning: Number of ROI names ({len(roi_names)}) does not match input dimension ({input_dim})")
        log_and_print_test("Generating new ROI names to match input dimension")
        roi_names = [f"Region_{i+1}" for i in range(input_dim)]
    #######

    # Count subjects by diagnosis
    diagnosis_counts = annotations_clinical["Diagnosis"].value_counts()
    log_and_print_test(f"Subject counts by diagnosis:\n{diagnosis_counts}")
    
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
    
    # ------------------------------- CALCULATION DEV_SCORES  --------------------------------------------
    log_and_print_test("----------[INFO] Calculating deviation scores...")
    results_df = calculate_deviations(
        normative_models=bootstrap_models,
        data_tensor=clinical_data,
        annotations_df=annotations_clinical,
        device=device
    )
    
    # Map ROI names to region columns if we have them
    if roi_names is not None:
        region_cols = [col for col in results_df.columns if col.startswith("region_")]
        if len(region_cols) == len(roi_names):
            # Create a copy of results_df with renamed columns
            named_results_df = results_df.copy()
            roi_mapping = dict(zip(region_cols, roi_names))
            named_results_df.rename(columns=roi_mapping, inplace=True)
            
            # Save version with ROI names
            named_results_df.to_csv(f"{save_dir}/deviation_scores_with_roi_names.csv", index=False)
            log_and_print_test(f"Saved deviation scores with ROI names to {save_dir}/deviation_scores_with_roi_names.csv")
    
    # Save deviation scores
    results_df.to_csv(f"{save_dir}/deviation_scores.csv", index=False)
    log_and_print_test(f"Saved deviation scores to {save_dir}/deviation_scores.csv")
    
    # Generate visualizations
    log_and_print_test("Generating visualizations...")
    
    # Plot deviation distributions
    plot_deviation_distributions(results_df, save_dir)
    log_and_print_test("Plotted deviation distributions")
    
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
    
    # Run statistical tests between groups
    log_and_print_test("Running statistical tests between groups...")
    

    ##------------------------------------------------ROI-WISE DEVIATION SCORES --------------------------------------------------
     
    log_and_print_test("----------[INFO] Analyzing regional deviations...")
    
    # Create supplementary files for debugging and reference
    if h5_file_path and os.path.exists(h5_file_path):
        
        # Using our improved regional deviation analysis
        regional_results = analyze_regional_deviations(
            results_df=results_df,
            save_dir=save_dir,
            clinical_data_path=h5_file_path,
            volume_type=volume_type,
            atlas_name=atlas_name,
            roi_names=roi_names,
            norm_diagnosis=norm_diagnosis,
        )
        log_and_print_test(f"Regional deviation analysis complete. Results saved to {save_dir}/regional_effect_sizes.csv")
    else:
        log_and_print_test("Warning: Could not analyze regional deviations due to missing HDF5 file.")
    
        log_and_print_test(f"Regional deviation analysis complete. Results saved to {save_dir}/regional_effect_sizes.csv")
    
    log_and_print_test(f"Deviation analysis complete! Results saved to {save_dir}")

    #-----------------------------------------------------------------------------------------------------------------------------

    log_and_print_test(f"Deviation analysis complete! Results saved to {save_dir}")
    end_logging(Config_2D)

    return save_dir 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate deviation scores for clinical groups.")
    # parser.add_argument("--model_dir", required=True, help="Directory containing trained normative models")
    # parser.add_argument("--clinical_data_path", required=True, help="Path to clinical data")
    # parser.add_argument("--clinical_csv", required=True, help="Path to clinical CSV file")
    parser.add_argument("--norm_diagnosis", type=str, default="HC")
    parser.add_argument("--atlas_name", help="Atlas name (if not available in config)")
    parser.add_argument("--latent_dim", type=int, default=20, help="Latent dimension (if not available in config)")
    parser.add_argument("--max_models", type=int, default=0, help="Maximum number of models to use (0 = all)")
    parser.add_argument("--output_dir", default=None, help="Output directory for results")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    main(args)