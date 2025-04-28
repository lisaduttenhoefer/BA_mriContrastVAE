import sys

import matplotlib
import numpy as np
import pandas as pd
import argparse

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
)

# Use non-interactive plotting to avoid tmux crashes
matplotlib.use("Agg")

def create_arg_parser():  # For training several different models with run_training.sh file
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--atlas_name', help='Name of the desired atlas for training.')
    parser.add_argument('--num_epochs', help='Number of epochs to be trained for')
    return parser

path_original = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/metadata_20250110/full_data_train_valid_test.csv"
path_to_dir = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training"
TRAIN_CSV, TEST_CSV = split_df(path_original, path_to_dir)

def main(atlas_name: str, num_epochs: int):
    ## 0. Set Up ----------------------------------------------------------
    # Set Parameters for model training
    # Config has default parameters you may want to check

    # Split the original metadata csv file that contains all data into two separate ones: training metadata and testing metadata.
    
    
    config = Config_2D(
        # General Parameters
        RUN_NAME="BasicVAE_2D_training01",
        # Input / Output Paths
        TRAIN_CSV= ["/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training/training_metadata.csv"], #"/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training/hc_metadata.csv", # TEST_CSV=["./data/relevant_metadata/testing_metadata.csv"],
        TEST_CSV= ["/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training/testing_metadata.csvs"], #"/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training/non_hc_metadata.csv",
        MRI_DATA_PATH="/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/train_xml_data_t", #"./data/raw_extracted_xml_data/train_xml_data", # This is the h5 file!
        ATLAS_NAME=atlas_name,
        PROC_DATA_PATH="/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data_training/proc_extracted_xml_data",
        OUTPUT_DIR="/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/analysis",
        # Loading Model
        LOAD_MODEL=False,
        PRETRAIN_MODEL_PATH=None,
        PRETRAIN_METRICS_PATH=None,
        CONTINUE_FROM_EPOCH=0,
        # Loss Parameters
        RECON_LOSS_WEIGHT=5.0,
        KLDIV_LOSS_WEIGHT=1.0,
        CONTR_LOSS_WEIGHT=0.0,  # if not zero, you're not running basicVAE
        #USE_SSIM=True,
        # Learning and Regularization
        TOTAL_EPOCHS=10,
        LEARNING_RATE=1e-5,
        WEIGHT_DECAY=1e-3,
        EARLY_STOPPING=True,
        STOP_LEARNING_RATE=1e-7,
        SCHEDULE_ON_VALIDATION=False,
        SCHEDULER_PATIENCE=10,
        SCHEDULER_FACTOR=0.5,
        #DROPOUT_PROB=0.1,
        # Visualization
        CHECKPOINT_INTERVAL=10,
        DONT_PLOT_N_EPOCHS=0,
        UMAP_NEIGHBORS=15,
        UMAP_DOT_SIZE=30,
        METRICS_ROLLING_WINDOW=20,
        # Data Parameters
        BATCH_SIZE=32,
        DIAGNOSES=["HC"], 
        #COVARS=["Dataset"],
        # Misc.
        LATENT_DIM=20,
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
     # Load all data for training (training + validation)
    
    if config.ATLAS_NAME != "all":
        subjects, annotations = load_mri_data_2D(
            csv_paths=config.TRAIN_CSV,
            data_path=config.MRI_DATA_PATH,
            atlas_name=config.ATLAS_NAME,
            # covars=config.COVARS,
            diagnoses=config.DIAGNOSES,
            hdf5=True,
            train_or_test="train",
            save=True
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
