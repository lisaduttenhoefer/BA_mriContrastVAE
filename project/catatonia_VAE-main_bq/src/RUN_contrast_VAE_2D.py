import sys
import matplotlib
import numpy as np
import pandas as pd
import torch
import torchio as tio #for medicinal image analysis
from torch.cuda.amp import GradScaler #to fasten the training

sys.path.append("../src") #/src wird zum python path hinzugefügt -> muss in einem Unterordner liegen
from models.ContrastVAE_2D import ContrastVAE_2D, train_ContrastVAE_2D    #the Model itself
from utils.config_utils_2D import Config       #Class to define the configuration
from utils.data_processing import load_checkpoint_model, load_mri_data, process_subjects
from utils.logging_utils import (log_and_print, log_data_loading, log_model_ready, log_model_setup, setup_logging,)

# Use non-interactive plotting to avoid tmux crashes -> matplotlib im Non-Interactive Mode
matplotlib.use("Agg")

## 0. Set Up ----------------------------------------------------------
# Set Parameters for model training
# Config has default parameters you may want to check -> Config definition in config_utils
config = Config(
    # General Parameters
    RUN_NAME="BasicVAE_1",  #run name for this training
    # Input / Output Paths
    TRAIN_CSV=["project/LISA_tryout_sessions/basicVAE_model/data/metadata_20250110/full_data_train_valid_test.csv"]     #training data paths (metadaten)
    VALID_CSV="data/aug_valid.csv",     #validation data paths (metadaten)
    MRI_DATA_PATH="data/augmented",     #all data
    OUTPUT_DIR="analysis",             #defining output path
    # Loading Model -> pretrained model parameter -> is there one? 
    LOAD_MODEL=False,
    PRETRAIN_MODEL_PATH=None,
    PRETRAIN_METRICS_PATH=None,
    CONTINUE_FROM_EPOCH=0,
    # Loss Parameters
    RECON_LOSS_WEIGHT=5.0,
    KLDIV_LOSS_WEIGHT=1.0,
    CONTR_LOSS_WEIGHT=0.0,  # if not zero, you're not running basicVAE !!
    USE_SSIM=False, #macht kein sinn bei numerischen 
    # Learning and Regularization
    TOTAL_EPOCHS=500, #how often does it work through the whole training data set -> a lot
    LEARNING_RATE=1e-5, #how big are the steps of the optimization algorithm towards a gradient
    #1e-5 small -> for pretrained models & complex models -> slow convergence
    WEIGHT_DECAY=1e-3, #L2-regularization, hurts big weights, more general
    EARLY_STOPPING=True,   #stopping early in case there is only a small learning rate 
    STOP_LEARNING_RATE=1e-7,
    SCHEDULE_ON_VALIDATION=False, #decides if the scheduler looks at the training or validation loss
    SCHEDULER_PATIENCE=10, #scheduler waits x epoches without improvement before reducing the learning rate
    SCHEDULER_FACTOR=0.5,  #after waiting the learning rate gets multiplied by x (0.5-> reduced by half)
    DROPOUT_PROB=0.1,
    # Visualization
    CHECKPOINT_INTERVAL=10,   # checkpoint after 10 epoches -> starting new oÄ / analyzing training
    DONT_PLOT_N_EPOCHS=0,    #starts with visualization right away 
    UMAP_NEIGHBORS=15,     #parameter for UMAP Visualisierung (local vs global structure)
    UMAP_DOT_SIZE=30,      #dots UMAP-plot
    METRICS_ROLLING_WINDOW=20,     #smoothes metrics for 20 epoches -> better plots
   
   #UMAP (Uniform Manifold Approximation and Projection) ist eine Dimensionalitätsreduktionstechnik, 
   #    -> visualize high-dimensional latent room (hier 128D) in 2D 
   #    -> gruppieren der Datenpunkte
    ROI_COUNT=90,  # Anzahl der ROIs
    BATCH_SIZE=32,
    DIAGNOSES=["ASD", "HC", "MDD", "SCHZ"], 
    COVARS=["Dataset"],
    # Misc.
    LATENT_DIM=20,
)

# Set up logging -> reproducable results (CPU and GPU)
setup_logging(config)

# Set seed for reproducibility
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)


## 1. Load Data and Initialize Model --------------------------------
# Load train data (ACTUAL MRI IMAGES! 3D)
train_subjects, train_annotations = load_mri_data(
    csv_path=config.TRAIN_CSV,
    data_path=config.MRI_DATA_PATH,
    covars=config.COVARS,
)

# Load validation data (ACTUAL MRI IMAGES! 3D)
valid_subjects, valid_annotations = load_mri_data(
    csv_path=config.VALID_CSV,
    data_path=config.MRI_DATA_PATH,
    covars=config.COVARS,
)
#annotations contains metadata to each image
train_annotations.insert(1, "Data_Type", "train")
valid_annotations.insert(1, "Data_Type", "valid")

annotations = pd.concat([train_annotations, valid_annotations])
annotations.sort_values(by=["Data_Type", "Filename"], inplace=True)
annotations.reset_index(drop=True, inplace=True)

#setting type of metadata (label and data type)
annotations = annotations.astype(
    {
        "Age": "float",
        "Dataset": "category",
        "Diagnosis": "category",
        "Sex": "category",
        "Data_Type": "category",
        "PatientID": "category",
    }
)

log_and_print(annotations)

# Transform data and put into train and validation loaders
    #image transformations  
    #CropOrPad: Bringt alle Bilder auf die gleiche Größe (128x128x128)
    #RescaleIntensity: Normalisiert die Pixelwerte auf den Bereich [0, 1]

#defining PyTorch DataLoader for training 
    #-> transformations from before
    #-> organize in batches (batch size)
    #-> mixing of the training set (NOT of the validation set)
train_loader = process_subjects(
    subjects=train_subjects,
    transforms=transforms,
    batch_size=config.BATCH_SIZE,
    shuffle_data=config.SHUFFLE_DATA,
)

valid_loader = process_subjects(
    subjects=valid_subjects,
    transforms=transforms,
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
    input_dim=config.ROI_COUNT
    num_classes=len(config.DIAGNOSES),              # how many defined diagnosis? 
    learning_rate=config.LEARNING_RATE,             # Learning rate
    weight_decay=config.WEIGHT_DECAY,               # L2-Regularisierung
    device=config.DEVICE,                           # CPU or GPU
    scaler=GradScaler(),                            # for Mixed Precision Training
    kldiv_loss_weight=config.KLDIV_LOSS_WEIGHT,     # weight for KL-Divergence-loss
    recon_loss_weight=config.RECON_LOSS_WEIGHT,     # weight for Rekonstruktionsverlust
    contr_loss_weight=config.CONTR_LOSS_WEIGHT,     # weight for Kontrastverlust (0 für BasicVAE) -> kein Kontrasterlust
    dropout_prob=config.DROPOUT_PROB,               # Dropout-Wahrscheinlichkeit
    latent_dim=config.LATENT_DIM,                   # size latent space
    schedule_on_validation=config.SCHEDULE_ON_VALIDATION,  # LR-Scheduling basierend auf Validierungsverlust
    scheduler_patience=config.SCHEDULER_PATIENCE,   # Geduld für LR-Scheduler
    scheduler_factor=config.SCHEDULER_FACTOR,       # Faktor für LR-Reduzierung
)

#loading a pretrained model in case that exists
if config.LOAD_MODEL:
    model = load_checkpoint_model(model, config.PRETRAIN_MODEL_PATH)
    model_metrics = pd.read_csv(config.PRETRAIN_METRICS_PATH)        #loads metrics of the pretrained model
else:                                                               #or adds new empy dfs for metrics of this model
    model_metrics = pd.DataFrame(
        {
            "train_loss": [],
            "t_recon_loss": [],
            "t_kldiv_loss": [],
            "learning_rate": [],
            "valid_loss": [],
            "v_recon_loss": [],
            "v_kldiv_loss": [],
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
)

#trains the model with the training dataset
#evaluates with the validation data
#saves checkpoints & metrics
#visualises depending on implementation
