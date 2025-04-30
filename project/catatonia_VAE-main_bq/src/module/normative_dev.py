%tensorflow_version 2.x
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from google.colab import files
from tqdm import tqdm

from utils.config_utils_model import Config_2D
#DATA-CSV: Config_2D.MRI_DATA_PATH
#METADATA-CSV: Config_2D.METADATA_WHOLE

def main(
    config = Config_2D(
        MRI_DATA_PATH="/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/train_xml_data", #"./data/raw_extracted_xml_data/train_xml_data", # This is the h5 file!
        METADATA_WHOLE = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/metadata_20250110/full_data_train_valid_test.csv"
        
)

    ## !!! iterieren lassen durch verschiedene Atlanten oder einzeln 


    #Datasets mergen: data & metadata together
    dataset_df = pd.merge(config.MRI_DATA_PATH, config.METADATA_WHOLE, on='Participant_ID')
    # Set random seed
    random_seed = 42
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    
    #Next, we define the name of the brain regions in the variable COLUMNS_NAME
    COLUMNS_NAME = .columns.tolist()