import sys
sys.path.append("/home/developer/.local/lib/python3.10/site-packages")
import matplotlib
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
sys.path.append("../src")
import torch
import torchio as tio
from torch.cuda.amp import GradScaler

sys.path.append("../src")
from models.ContrastVAE_2D_f import ContrastVAE_2D, train_ContrastVAE_2D
from utils.support_f import get_all_data, split_df, combine_dfs
from utils.config_utils_model import Config_2D

from module.data_processing_hc import (load_checkpoint_model, load_mri_data_2D, process_subjects, train_val_split_annotations,train_val_split_subjects
   )
    
from utils.logging_utils import (
    log_and_print,
    log_data_loading,
    log_model_ready,
    log_model_setup,
    setup_logging,
    log_atlas_mode
)

# Use non-interactive plotting to avoid tmux crashes
matplotlib.use("Agg")

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Arguments for Normative Modeling Training')
    parser.add_argument('--atlas_name', help='Name of the desired atlas for training.')
    parser.add_argument('--num_epochs', help='Number of epochs to be trained for', type=int, default=20)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=32)
    parser.add_argument('--learning_rate', help='Learning rate', type=float, default=4e-5)
    parser.add_argument('--latent_dim', help='Dimension of latent space', type=int, default=20)
    parser.add_argument('--kldiv_weight', help='Weight for KL divergence loss', type=float, default=4.0)
    parser.add_argument('--seed', help='Random seed for reproducibility', type=int, default=42)
    parser.add_argument('--output_dir', help='Override default output directory', default=None)
    return parser

path_original = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/metadata_20250110/full_data_train_valid_test.csv"
path_to_dir = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training"
TRAIN_CSV, TEST_CSV = split_df(path_original, path_to_dir)

def main(atlas_name: str, num_epochs: int, batch_size: int, learning_rate: float, 
         latent_dim: int, kldiv_weight: float, seed: int, output_dir: str = None):
    ## 0. Set Up ----------------------------------------------------------
    # Set Parameters for model training
    # Config has default parameters you may want to check
    if output_dir is None:
        save_dir = f"/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/normative_results_{timestamp}"
    else:
        save_dir = output_dir
    # Split the original metadata csv file that contains all data into two separate ones: training metadata and testing metadata.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    config = Config_2D(
        # General Parameters
        RUN_NAME=f"NormativeVAE_{atlas_name}_{timestamp}",
        # Input / Output Paths
        TRAIN_CSV=["/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training/training_metadata.csv"],
        TEST_CSV=["/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training/testing_metadata.csv"],
        MRI_DATA_PATH_TRAIN="/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/train_xml_data",
        MRI_DATA_PATH_TEST="/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/test_xml_data",
        ATLAS_NAME=atlas_name,
        PROC_DATA_PATH="/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training/proc_extracted_xml_data",
        OUTPUT_DIR=save_dir,
        # load_mri_data 
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

    # Set up logging
    setup_logging(config)
    

    # Set seed for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)


    ## 1. Load Data and Initialize Model --------------------------------
     # Load all data for training (training + validation)
    
    if config.ATLAS_NAME != "all":
        subjects, annotations = load_mri_data_2D(
            csv_paths=config.TRAIN_CSV,
            data_path=config.MRI_DATA_PATH_TRAIN,
            atlas_name=config.ATLAS_NAME,
            # covars=config.COVARS,
            diagnoses=config.DIAGNOSES,
            hdf5=True,
            train_or_test="train",
            save=True
        )
    else:
        all_data_paths = get_all_data(directory=config.MRI_DATA_PATH_TRAIN, ext="h5")
        
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

    train_annotations.insert(1, "Data_Type", "train")
    valid_annotations.insert(1, "Data_Type", "valid")

    annotations = pd.concat([train_annotations, valid_annotations])
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



    train_loader = process_subjects(
        subjects=train_subjects,
        #transforms=transforms,
        batch_size=config.BATCH_SIZE,
        shuffle_data=config.SHUFFLE_DATA,
    )

    valid_loader = process_subjects(
        subjects=valid_subjects,
        #transforms=transforms,
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
    model = ContrastVAE_2D(
        #channels=1,
        num_classes=len(config.DIAGNOSES),
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        device=config.DEVICE,
        scaler=GradScaler(),
        kldiv_loss_weight=config.KLDIV_LOSS_WEIGHT,
        recon_loss_weight=config.RECON_LOSS_WEIGHT,
        contr_loss_weight=config.CONTR_LOSS_WEIGHT,
        # kernel_size=3,
        # stride=1,
        # padding=1,
        # dropout_prob=config.DROPOUT_PROB,
        # image_size=config.IMAGE_SIZE,
        latent_dim=config.LATENT_DIM,
        schedule_on_validation=config.SCHEDULE_ON_VALIDATION,
        scheduler_patience=config.SCHEDULER_PATIENCE,
        scheduler_factor=config.SCHEDULER_FACTOR,
        #use_ssim=config.USE_SSIM,
        input_dim=len_atlas
    )

    if config.LOAD_MODEL:
        model = load_checkpoint_model(model, config.PRETRAIN_MODEL_PATH)
        model_metrics = pd.read_csv(config.PRETRAIN_METRICS_PATH)
    else:
        model_metrics = pd.DataFrame(
            {
                "train_loss": [],
                "t_recon_loss": [],
                "t_kldiv_loss": [],
                "learning_rate": [],
                "valid_loss": [],
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

    train_ContrastVAE_2D(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        annotations=annotations,
        model_metrics=model_metrics,
        config=config,
        no_plotting=True,
        no_val_plotting=False,
        no_saving=False
    )
    return


if __name__ == "__main__": 
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    atlas_name = parsed_args.atlas_name
    atlas_name = str(atlas_name)
    num_epochs = parsed_args.num_epochs
    num_epochs = int(num_epochs)
    main(atlas_name=atlas_name, num_epochs=num_epochs)
