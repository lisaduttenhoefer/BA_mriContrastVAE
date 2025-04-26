import os
import subprocess as sp
from datetime import datetime, timedelta
from typing import List

import torch

'''
Config class to set up all parameters for model preparation and training. Is used with the run_ModelXYZ.py scripts.
'''

# Get the GPU with the most memory capacity
def get_free_gpu() -> torch.device:

    # check if cuda is available
    if torch.cuda.is_available():
        # run nvidia-smi command to get data on free memory
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = (
            sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
        )
        # extract memory values
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]

        # return gpu with the most free memory
        gpu = f"cuda:{memory_free_values.index(max(memory_free_values))}"

    # if cuda isn't available, run on cpu
    else:
        gpu = "cpu"

    # return the device with the most free memory
    return torch.device(gpu)


# Set up all parameters for model preparation and training
class Config_2D:

    def __init__(
        self,
        # The learning rate for the model optimizer
        LEARNING_RATE: float,
        # The weight decay for the model optimizer
        WEIGHT_DECAY: float,
        # The batch size of the data loaders
        BATCH_SIZE: int,
        # The total number of epochs to train the model for
        # If loading a pre-trained model, this is the number of additional epochs to train for
        TOTAL_EPOCHS: int,
        # The weight of the reconstruction loss of the model
        RECON_LOSS_WEIGHT: float,
        # The weight of the KL divergence loss of the model
        KLDIV_LOSS_WEIGHT: float,
        # The latent space dimensionality of the model
        LATENT_DIM: int,
        # A string to identify your run (will be in name of the output directory)
        RUN_NAME: str,
        # The path to the csv files (contains metadata and filenames of MRI .nii files)
        TRAIN_CSV: List[str],
        # The paths to the csv files that contain metadata for the testing data
        TEST_CSV: List[str],
        # Name of atlas which should be used for training of the model
        ATLAS_NAME: str,
        # The path to the folder where intermediate normalitzed and scaled data should be saved
        PROC_DATA_PATH: str,
        # The path to the directory that contains the MRI .nii files
        MRI_DATA_PATH: str,
        # The folder in which a training specific output directory should be created
        OUTPUT_DIR: str,
        # The column names, from the CSVs, that contain the covariates that you want to be attached to the Subject objects
        # Covariates will be one-hot encoded. Diagnoses is always included. ClassCVAE requires Dataset to be included.
        # COVARS: List[str],
        # The list of diagnoses that you want to include in training. One-hot encoded Diagnosis information will always
        # be appended to the Subjects object.
        DIAGNOSES: List[str],
        # Whether to use the Structural Similarity Index (SSIM) as a loss function for reconstruction loss.
        # Not all models support this.
        USE_SSIM: bool = False,
        # Whether to use early stopping during training, based on the LR being too low.
        EARLY_STOPPING: bool = True,
        # The learning rate at which to stop training if early stopping is enabled.
        STOP_LEARNING_RATE: float = 1e-6,
        # The probability of dropout in each convolutional stack of the model. Not all models support this.
        DROPOUT_PROB: float = 0.1,
        # The weight of of the classifier loss. Not all models have classifiers.
        CLASS_LOSS_WEIGHT: float = None,
        # The weight of the contrastive loss. Only ContrastVAE uses this.
        CONTR_LOSS_WEIGHT: float = None,
        # The temperature of the contrastive loss. Temperature is a hyperparameter similar to a loss weight.
        # Only ContrastVAE uses this.
        CONTRAST_TEMPERATURE: float = None,
        # The path to the csv file containing the adversarial training data. Only AdverVAE uses this.
        ADVER_CSV: str = None,
        # The learning rate for the the adversarial training optimizer. Only AdverVAE uses this.
        ADVER_LR: float = None,
        # Whether to load a pre-trained model or not. If True, PRETRAIN_MODEL_PATH and PRETRAIN_METRICS_PATH must be set.
        LOAD_MODEL: bool = False,
        # The path to the pre-trained model to load.
        PRETRAIN_MODEL_PATH: str = None,
        # The path to the pre-trained model's performance metrics to load.
        PRETRAIN_METRICS_PATH: str = None,
        # The epoch to continue training from if loading a pre-trained model. If it is None, the model will start from the 0th epoch.
        CONTINUE_FROM_EPOCH: int = None,
        # The number of epochs between saving model checkpoints. Model checkpoints save the model's weights, latent representations, and latent plots.
        CHECKPOINT_INTERVAL: int = 10,
        # Wether to plot loss curves with a rolling window as well (can help clarify trends)
        METRICS_ROLLING_WINDOW: int = 10,
        # The size of the dots in the UMAP plot. Larger dots are easier to see, but may overlap.
        UMAP_DOT_SIZE: int = 30,
        # The number of neighbors to consider when plotting the UMAP plot. More neighbors can help clarify clusters, but hide local structure.
        UMAP_NEIGHBORS: int = 15,
        # Should training data be shuffled in the DataLoader?
        SHUFFLE_DATA: bool = True,
        # Seed for reproducibility
        SEED: int = 123,
        # The first training epochs can be volatile, and can mess with the scale of the plots. This parameter allows you to skip the first n epochs.
        DONT_PLOT_N_EPOCHS: int = 0,
        # Whether to schedule the learning rate based on the validation loss. If False, the learning rate will be scheduled based on the training loss.
        # Scheduling based on the validation loss can help prevent overfitting.
        SCHEDULE_ON_VALIDATION: bool = True,
        # The number of epochs to wait before reducing the learning rate if the loss does not improve.
        SCHEDULER_PATIENCE: int = 10,
        # The factor by which to reduce the learning rate if the loss does not improve.
        SCHEDULER_FACTOR: float = 0.01,
        # The device to use for training. If None, the device with the most free memory will be retrieved.
        DEVICE: torch.device = None,
        # A timestamp for the run. If None, the current time will be used. Timestamp will be in output directory name.
        TIMESTAMP: str = None,
    ):

        # set up training parameters ------------------------------------------------

        # set up mandatory training parameters
        self.LEARNING_RATE = LEARNING_RATE
        self.WEIGHT_DECAY = WEIGHT_DECAY
        self.BATCH_SIZE = BATCH_SIZE
        self.LATENT_DIM = LATENT_DIM
        self.CONTR_LOSS_WEIGHT = CONTR_LOSS_WEIGHT
        self.RECON_LOSS_WEIGHT = RECON_LOSS_WEIGHT
        self.KLDIV_LOSS_WEIGHT = KLDIV_LOSS_WEIGHT
        self.ADVER_LR = ADVER_LR
        self.CLASS_LOSS_WEIGHT = CLASS_LOSS_WEIGHT
        self.CONTRAST_TEMPERATURE = CONTRAST_TEMPERATURE
        self.TOTAL_EPOCHS = TOTAL_EPOCHS
        # self.COVARS = COVARS

        # set up optional training parameters
        self.CHECKPOINT_INTERVAL = CHECKPOINT_INTERVAL
        self.DIAGNOSES = DIAGNOSES
        self.UMAP_NEIGHBORS = UMAP_NEIGHBORS
        self.SHUFFLE_DATA = SHUFFLE_DATA
        self.SEED = SEED
        self.START_EPOCH = 0
        self.DROPOUT_PROB = DROPOUT_PROB
        self.USE_SSIM = USE_SSIM
        self.SCHEDULE_ON_VALIDATION = SCHEDULE_ON_VALIDATION
        self.SCHEDULER_PATIENCE = SCHEDULER_PATIENCE
        self.SCHEDULER_FACTOR = SCHEDULER_FACTOR

        # training stop parameters
        self.EARLY_STOPPING = EARLY_STOPPING
        self.STOP_LEARNING_RATE = STOP_LEARNING_RATE

        if DEVICE is not None:
            self.DEVICE = DEVICE
        else:
            self.DEVICE = get_free_gpu()

        if TIMESTAMP is not None:
            self.TIMESTAMP = TIMESTAMP
        else:
            # server is off german local time by 2 hrs
            current_time = datetime.now() + timedelta(hours=2)
            self.TIMESTAMP = current_time.strftime("%Y-%m-%d_%H-%M")

        # set up plotting parameters
        self.DONT_PLOT_N_EPOCHS = DONT_PLOT_N_EPOCHS
        self.METRICS_ROLLING_WINDOW = METRICS_ROLLING_WINDOW
        self.UMAP_DOT_SIZE = UMAP_DOT_SIZE

        # set up run parameters ------------------------------------------------

        # set up model loading parameters for pre-trained models
        self.LOAD_MODEL = LOAD_MODEL

        if self.LOAD_MODEL:
            for path in [PRETRAIN_MODEL_PATH, PRETRAIN_METRICS_PATH]:
                assert (
                    path is not None
                ), "PRETRAIN_MODEL_PATH and PRETRAIN_METRICS_PATH must be set if LOAD_MODEL is True."
                assert os.path.exists(path), f"Path {path} does not exist."
            self.START_EPOCH = CONTINUE_FROM_EPOCH

        self.PRETRAIN_MODEL_PATH = PRETRAIN_MODEL_PATH
        self.PRETRAIN_METRICS_PATH = PRETRAIN_METRICS_PATH
        self.FINAL_EPOCH = self.START_EPOCH + self.TOTAL_EPOCHS

        # check that all other paths exist
        for path in [ADVER_CSV, MRI_DATA_PATH, OUTPUT_DIR]:
            if path is not None:
                assert os.path.exists(path), f"Path {path} does not exist"

        # set up paths
        self.OUTPUT_DIR = OUTPUT_DIR
        self.TRAIN_CSV = TRAIN_CSV
        self.ADVER_CSV = ADVER_CSV
        self.MRI_DATA_PATH = MRI_DATA_PATH
        self.PROC_DATA_PATH = PROC_DATA_PATH
        self.ATLAS_NAME = ATLAS_NAME
        self.TEST_CSV = TEST_CSV

        # set up path dependent parameters
        if RUN_NAME is None:
            self.RUN_NAME = os.path.split(self.TRAIN_CSV)[-1].split(".")[0]
        else:
            self.RUN_NAME = RUN_NAME

        self.OUTPUT_DIR = os.path.join(
            self.OUTPUT_DIR, f"{self.TIMESTAMP}_{self.RUN_NAME}"
        )
        self.FIGURES_DIR = os.path.join(self.OUTPUT_DIR, "figures")
        self.LOGGING_DIR = os.path.join(self.OUTPUT_DIR, "logs")
        self.DATA_DIR = os.path.join(self.OUTPUT_DIR, "data")
        self.MODEL_DIR = os.path.join(self.OUTPUT_DIR, "models")

        # set up run specific output directories
        for path in [
            self.OUTPUT_DIR,
            self.FIGURES_DIR,
            self.LOGGING_DIR,
            self.DATA_DIR,
            self.MODEL_DIR,
        ]:
            # make sure it doesn't already exist
            assert not os.path.exists(
                path
            ), f"Path {path} already exists, and may be overwritten. Please rename or remove it."

            # create the directory
            os.makedirs(path)

    def __str__(self):
        return str(vars(self))