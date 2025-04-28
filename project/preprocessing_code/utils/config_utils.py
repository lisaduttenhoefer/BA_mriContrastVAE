import os
import pathlib
from typing import List

def add_to_gitignore(path: str):
    with open(".gitignore", "a") as g:
        g.write(f"{path}")
        g.write("\n")
    return


'''
Config class to set up all parameters for model preparation and training. Is used with the run_ModelXYZ.py scripts.
'''

# Set up all parameters for model preparation and training
class Config:

    def __init__(
        self,
        # Path to directory where all the .xml files that should be processed are located.
        RAW_DATA_DIR: str = None,
        # Path to directory where all .csv data from the .xml files should be saved (rows=features, cols=patients).
        EXTRACTED_CSV_DIR: str = None,
        # Path to directory where all .csv data from the .xml files should be saved (rows=patients, cols=features).
        EXTRACTED_CSV_T_DIR: str = None,
        # Path to directory where all the log-normalized and z-scaled .csv files should be saved (rows=features, cols=patients).
        PROCESSED_CSV_DIR: str = None,
        # List of paths to metadata (.csv) files that should be used to filter the extracted per-patient data.
        METADATA_PATHS: list = None,  
        # Determines if for each patient all their atlases should be considered, or only one.
        ALL_ATLASES: bool = False,
        # Determines if training or testing data should be extracted.
        TRAIN_DATA: bool = True,
        # Names of all the folders in RAW_DATA_DIR that should be used for testing.
        TEST_DATA: list = None
    ):

        for path in [RAW_DATA_DIR, METADATA_PATHS]:
            if path is not None:
                if not isinstance(path, list):
                    assert os.path.exists(path), f"Path {path} does not exist"
                else: 
                    for p in path: 
                        assert os.path.exists(p), f"Path {p} does not exist"
        # set up paths
        self.RAW_DATA_DIR = RAW_DATA_DIR
        self.EXTRACTED_CSV_DIR = EXTRACTED_CSV_DIR
        self.EXTRACTED_CSV_T_DIR = EXTRACTED_CSV_T_DIR
        self.PROCESSED_CSV_DIR = PROCESSED_CSV_DIR
        self.METADATA_PATHS = METADATA_PATHS
        self.ALL_ATLASES = ALL_ATLASES
        self.TRAIN_DATA = TRAIN_DATA
        self.TEST_DATA = TEST_DATA

        metadata_parent = pathlib.Path(self.METADATA_PATHS[0]).parents[0]
        
        add_to_gitignore(str(metadata_parent))

        # set up run specific output directories
        for path in [
            self.EXTRACTED_CSV_DIR, 
            self.EXTRACTED_CSV_T_DIR, 
            self.PROCESSED_CSV_DIR,
        ]:
            # make sure it doesn't already exist
            assert not os.path.exists(
                path
            ), f"Path {path} already exists, and may be overwritten. Please rename or remove it."
            # create the directory
            os.makedirs(path)
        
        for path in [
            self.RAW_DATA_DIR,
            self.EXTRACTED_CSV_DIR, 
            self.EXTRACTED_CSV_T_DIR, 
            self.PROCESSED_CSV_DIR,
        ]:
            basename = os.path.basename(path)
            add_to_gitignore(basename)

    def __str__(self):
        return str(vars(self))