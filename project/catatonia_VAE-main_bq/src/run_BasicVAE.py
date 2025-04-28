import sys

import matplotlib
import numpy as np
import pandas as pd
import torch
import torchio as tio
from torch.cuda.amp import GradScaler

sys.path.append("../src")
from models.ContrastVAE import ContrastVAE, train_ContrastVAE
from utils.config_utils import Config
from utils.data_processing import (load_checkpoint_model, load_mri_data, process_subjects, train_val_split_annotations,train_val_split_subjects
    train_val_split_subjects)
    
from utils.logging_utils import (
    log_and_print,
    log_data_loading,
    log_model_ready,
    log_model_setup,
    setup_logging,
)

# Use non-interactive plotting to avoid tmux crashes
matplotlib.use("Agg")


## 0. Set Up ----------------------------------------------------------
# Set Parameters for model training
# Config has default parameters you may want to check
config = Config_2D(
    # General Parameters
    RUN_NAME="BasicVAE_training",
    # Input / Output Paths
    TRAIN_CSV="data/aug_train.csv", # TEST_CSV=["./data/relevant_metadata/testing_metadata.csv"],
    VALID_CSV="data/aug_valid.csv", #["./data/relevant_metadata/training_metadata.csv"]
    MRI_DATA_PATH="data/augmented", #"./data/raw_extracted_xml_data/train_xml_data", # This is the h5 file!
    #PROC_DATA_PATH="./data/proc_extracted_xml_data",
    PROC_DATA_PATH="./data/proc_extracted_xml_data",
    OUTPUT_DIR="analysis",
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
    TOTAL_EPOCHS=500,
    LEARNING_RATE=1e-5,
    WEIGHT_DECAY=1e-3,
    EARLY_STOPPING=True,
    STOP_LEARNING_RATE=1e-7,
    SCHEDULE_ON_VALIDATION=False,
    SCHEDULER_PATIENCE=10,
    SCHEDULER_FACTOR=0.5,
    DROPOUT_PROB=0.1,
    # Visualization
    CHECKPOINT_INTERVAL=10,
    DONT_PLOT_N_EPOCHS=0,
    UMAP_NEIGHBORS=15,
    UMAP_DOT_SIZE=30,
    METRICS_ROLLING_WINDOW=20,
    # Data Parameters
    BATCH_SIZE=32,
    DIAGNOSES=["CTT", "HC", "MDD", "SCHZ"],
    #COVARS=["Dataset"],
    # Misc.
    LATENT_DIM=20,
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
train_subjects, train_annotations = load_mri_data(
    csv_path=config.TRAIN_CSV,
    data_path=config.MRI_DATA_PATH,
    diagnoses=config.DIAGNOSES,
    hdf5=True,
    train_or_test="train",
    save=True
    #covars=config.COVARS,
)

# Load validation data
valid_subjects, valid_annotations = load_mri_data(
    csv_path=config.VALID_CSV,
    data_path=config.MRI_DATA_PATH,
    covars=config.COVARS,
)

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

# Log data setup
log_data_loading(
    datasets={
        "Training Data": len(train_subjects),
        "Validation Data": len(valid_subjects),
    }
)


# Initialize Model
log_model_setup()
model = ContrastVAE(
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

train_ContrastVAE(
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
