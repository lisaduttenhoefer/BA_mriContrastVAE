import os
from typing import List, Tuple
import pandas as pd
import regex as re
import anndata as ad
import pandas as pd
import scanpy as sc
from typing import List, Tuple
import numpy as np
import pathlib
import torch
import torchio as tio
from torch.utils.data import DataLoader

from utils.logging_utils import log_and_print, log_checkpoint


def flatten_array(df: pd.DataFrame) -> np.ndarray:
    # Converts data frame to flattened array. 
    array = df.to_numpy()
    flat_array = array.flatten()
    return flat_array


# def normalize_and_scale_df(df: pd.DataFrame) -> pd.DataFrame:
#     # Normalizes the columns (patient volumes) by Min-Max Scaling and scales the rows (ROIs) with Z-transformation.

#     df_copy = df.copy()
#     column_sums = df_copy.sum()
    
#     # Apply the formula: ln((10000*value)/sum_values + 1) "Log transformation"
#     # Alternatively for Min-Max Scaling: df_copy/df_copy.max() - Problem: Some rows have std = 0
#     transformed_df = np.log((10000 * df_copy) / column_sums + 1)
    
#     norm_copy = transformed_df.copy()

#     cols = norm_copy.columns.get_level_values(-1).tolist()
#     unique_cols = list(set(cols))

#     for col_type in unique_cols:
#         cols_to_scale = [col for col in norm_copy.columns if col[-1] == col_type]

#         # Scale the selected columns per row
#         scaled = norm_copy[cols_to_scale].apply(
#             lambda row: (row - row.mean()) / row.std() if row.std() > 0 else pd.Series(0.0, index=row.index),
#             axis=1
#         )
        
#         norm_copy.loc[:, cols_to_scale] = scaled
        
#     return norm_copy

def normalize_and_scale_df(df: pd.DataFrame, ticv_column=None) -> pd.DataFrame:
    """
    Normalizes brain region volumes using the described preprocessing approach:
    1. Calculate relative volumes by dividing by total intracranial volume (if provided)
    2. Perform robust normalization: (x - median) / IQR for each brain region
    """
    df_copy = df.copy()
    
    if ticv_column is not None:
        ticv_values = df_copy[ticv_column]
        if ticv_column in df_copy.columns:
            df_copy = df_copy.drop(columns=[ticv_column])
        for column in df_copy.columns:
            df_copy[column] = df_copy[column] / ticv_values

    # Use a dictionary to store normalized columns
    normalized_data = {}

    for column in df_copy.columns:
        median_value = df_copy[column].median()
        q1 = df_copy[column].quantile(0.25)
        q3 = df_copy[column].quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            normalized_data[column] = pd.Series(0, index=df_copy.index)
        else:
            normalized_data[column] = (df_copy[column] - median_value) / iqr

    # Create the final normalized DataFrame at once
    normalized_df = pd.DataFrame(normalized_data, index=df_copy.index)

    return normalized_df


def get_all_data(directory: str, ext: str = "h5") -> list:
    data_paths = list(pathlib.Path(directory).rglob(f"*.{ext}"))
    return data_paths


def get_atlas(path: pathlib.PosixPath) -> str:
    stem = path.stem
    match = re.search(r"_(.*)", stem)
    if match:
        atlas = match.group(1)
    return atlas

# def combine_dfs(paths: list):
#     # Combines any number of csv files to a single pandas DataFrame, keeping only shared column indices. 
#     for i in range(1,len(paths)):
#         if i == 1: 
#             joined_df = pd.read_csv(paths[i-1], header=[0], index_col=0)
#             next_df = pd.read_csv(paths[i], header=[0], index_col=0)
#             joined_df = pd.concat([joined_df, next_df], join="inner")  # Parameter "inner" keeps only the shared column indices.
#         else:
#             next_df = pd.read_csv(paths[i], header=[0], index_col=0)
#             joined_df = pd.concat([joined_df, next_df], join="inner")
#     return joined_df

def combine_dfs(paths: list):
    # Combines any number of csv files to a single pandas DataFrame, keeping only shared column indices. 
    
    if len(paths) > 1: 
        for i in range(1,len(paths)):
            if i == 1: 
                joined_df = pd.read_csv(paths[i-1], header=[0])
                next_df = pd.read_csv(paths[i], header=[0])
                joined_df = pd.concat([joined_df, next_df], join="inner")  # Parameter "inner" keeps only the shared column indices.
            else:
                next_df = pd.read_csv(paths[i], header=[0])
                joined_df = pd.concat([joined_df, next_df], join="inner")
    else: 
        joined_df = pd.read_csv(paths[0], header=[0])
    return joined_df


def read_hdf5_to_df(filepath: str):
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist")
        return None
    try:
        return pd.read_hdf(filepath, key='atlas_data')
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def read_hdf5_to_df_t(filepath: str):
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist")
        return None
    try:
        return pd.read_hdf(filepath, key='atlas_data_t')
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

#splitting the HC patients in training and validation subsets
def train_val_split_annotations(
    # The annotations dataframe that you want to split into a train and validation part
    annotations: pd.DataFrame,
    # The proportion of the data that should be in the training set (the rest is in the test set)
    train_proportion: float = 0.8,
    # The diagnoses you want to include in the split, defaults to all
    diagnoses: List[str] = None,
    # The datasets you want to include in the split, defaults to all
    datasets: List[str] = None,
    # The random seed for reproducibility
    seed: int = 123,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Initialize empty dataframes
    train = pd.DataFrame()
    valid = pd.DataFrame()

    # If no diagnoses are specified, use all diagnoses in the annotations
    if diagnoses is None:
        diagnoses = annotations["Diagnosis"].unique()

    # If no datasets are specified, use all datasets in the annotations
    if datasets is None:
        datasets = annotations["Dataset"].unique()

    # For each diagnosis, dataset and sex, take a random sample of the data and split it into train and test
    for diagnosis in diagnoses:
        for dataset in datasets:
            for sex in ["Female", "Male"]:
                # subset to the current cohort
                dataset_annotations = annotations[
                    (annotations["Diagnosis"] == diagnosis)
                    & (annotations["Dataset"] == dataset)
                    & (annotations["Sex"] == sex)
                ]
                # shuffle the data
                dataset_annotations = dataset_annotations.sample(
                    frac=1, random_state=seed
                )
                # split the data train_proportion of the way through (usually 80%)
                split = round(len(dataset_annotations) * train_proportion)

                # add the split data to the train and test dataframes
                train = pd.concat(
                    [train, dataset_annotations[:split]], ignore_index=True
                )
                valid = pd.concat([valid, dataset_annotations[split:]], ignore_index=True)

    # return the split annotations. They have no overlap.
    return train, valid

def train_val_split_subjects(
    # The list of patients that should be split according to the prior train_val_split
    subjects: List[dict], 
    # Training annotations
    train_ann: pd.DataFrame, 
    # Validation annotations
    val_ann: pd.DataFrame
) -> Tuple[List, List]:
    
    train_files = list(train_ann["Filename"])
    valid_files = list(val_ann["Filename"])

    train_subjects = []
    valid_subjects = []

    for subject in subjects:

        if subject["name"] in train_files:
            train_subjects.append(subject)

        elif subject["name"] in valid_files:
            valid_subjects.append(subject)
            
        else: 
            print(f"Filename {subject['name']} not found in annotations.")

    return train_subjects, valid_subjects


# This function loads MRI data (nii.gz or .nii file formats) and stores it in a list of tio.Subject objects. The filenames
# that are used to load the MRI data are stored in a CSV file, or provided as an annotations pandas DataFrame. The
# diagnoses and covariates are one-hot encoded and stored in the tio.Subject objects (covariates are other variables that
# are not diagnoses, that the user specifies, that are in the CSV or annotations). The diagnoses of the subjects can be
# filtered by providing a list of diagnoses. The function returns the list of tio.Subject objects and the annotations the
# filtered annotations pandas DataFrame.


class CustomDataset_2D():  
    # Create Datasets that can then be converted into DataLoader objects
    def __init__(self, subjects):
        self.subjects = subjects

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        measurements = self.subjects[idx]["measurements"]
        labels = self.subjects[idx]["labels"]

        labels_df = pd.DataFrame(labels)
        labels_arr = labels_df.values
        names = self.subjects[idx]["name"]

        measurements = torch.as_tensor(measurements, dtype=torch.float32)  # Ensure this is float32, as the weights of the model are initialized.
        labels = torch.as_tensor(labels_arr, dtype=torch.float32)  # float32 required for linear operations!

        return measurements, labels, names

   
def load_mri_data_2D(
    # The path to the directory where the MRI data is stored (.csv file formats)
    data_path: str,
    # Name of the atlas that should be used for training.
    atlas_name: str,
    # The path to the CSV file that contains the filenames of the MRI data and the diagnoses and covariates
    csv_paths: list = None,
    # The annotations DataFrame that contains the filenames of the MRI data and the diagnoses and covariates
    annotations: pd.DataFrame = None,
    # The diagnoses that you want to include in the data loading, defaults to all
    diagnoses: List[str] = None,
    covars: List[str] = [],
    # Are the files of raw extracted xml data in the .h5 (True) or .csv (False) format?
    hdf5: bool = True,
    # Intended usage for the currently loaded data.
    train_or_test: str = "train",
    # Should the normalized and scaled file be saved?
    save: bool = False
) -> Tuple:

    atlas_data_path = f"{data_path}/Aggregated_{atlas_name}.h5"
    #atlas_data_path = f"/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/train_xml_data/Aggregated_cobra.h5"

    # If the CSV path is provided, check if the file exists, make sure that the annotations are not provided
    if csv_paths is not None:
        for csv_path in csv_paths: 
            assert os.path.isfile(csv_path), f"CSV file '{csv_path}' not found"
            assert annotations is None, "Both CSV and annotations provided"

        # Initialize the data overview DataFrame
        data_overview = combine_dfs(csv_paths)

    # If the annotations are provided, make sure that they are a pandas DataFrame, and that the CSV path is not provided
    if annotations is not None:
        assert isinstance(
            annotations, pd.DataFrame
        ), "Annotations must be a pandas DataFrame"
        assert csv_paths is None, "Both CSV and annotations provided"

        # Initialize the data overview DataFrame
        data_overview = annotations

    # If no diagnoses are provided, use all diagnoses in the data overview
    if diagnoses is None:
        diagnoses = data_overview["Diagnosis"].unique().tolist()
    
    # if diagnosis != ["HC"]:
    #     diagnoses = data_overview["Diagnosis"].unique().tolist()
    #     diagnoses = [d for d in diagnoses if d != "HC"]
    # else:
    #     diagnoses = ["HC"]

    # If the covariates are not a list, make them a list
    if not isinstance(covars, list):
        covars = [covars]

    # If the diagnoses are not a list, make them a list
    if not isinstance(diagnoses, list):
        diagnoses = [diagnoses]
    
    # Set all the variables that will be one-hot encoded
    variables = ["Diagnosis"] + covars

    # Filter unwanted diagnoses
    data_overview = data_overview[data_overview["Diagnosis"].isin(diagnoses)]
    data_overview = data_overview.drop(columns=['Unnamed: 0'])
    data_overview = data_overview[["Filename", "Dataset", "Diagnosis" , "Age", "Sex", "Usage_original", "Sex_int"]]

    # produce one hot coded labels for each variable
    one_hot_labels = {}
    for var in variables:
        # check that the variables is in the data overview
        if var not in data_overview.columns:
            raise ValueError(f"Column '{var}' not found in CSV file or annotations")

        # one hot encode the variable
        one_hot_labels[var] = pd.get_dummies(data_overview[var], dtype=float)

    # For each subject, collect MRI data and variable data in the Subject object
    subjects = []

    if hdf5 == True: 
        data = read_hdf5_to_df(filepath=atlas_data_path)
    else:
        data = pd.read_csv(atlas_data_path, header=[0, 1], index_col=0)

    # if train_or_test == "train":
    #     data.to_csv(f"/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/train_processed_data/Proc_{atlas_name}.csv")
    #     all_file_names = data.columns
    # else:
    #     data.to_csv(f"/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/test_processed_data/Proc_{atlas_name}.csv")
    #     all_file_names = data.columns        
        
    # data.set_index("Filename", inplace=True)

    data = normalize_and_scale_df(data)

    if save == True:
        #atlas_name = data_path.stem
        data.to_csv(f"data/proc_extracted_xml_data/Proc_{atlas_name}_{train_or_test}.csv")
        
    all_file_names = data.columns

    for index, row in data_overview.iterrows():
        subject = {} 
        
        if not row["Filename"] in all_file_names:
            continue

        # Format correct filename
        if row["Filename"].endswith(".nii") or row["Filename"].endswith(".nii.gz"):
            pre_file_name = row["Filename"]
            match_no_ext = re.search(r"([^/\\]+)\.[^./\\]*$", pre_file_name)  # Extract file stem
            if match_no_ext:
                file_name = match_no_ext.group(1)
        else:
            file_name = row["Filename"]

        patient_data = data[file_name]
        flat_patient_data = flatten_array(patient_data).tolist()

        subject["name"] = file_name
        subject["measurements"] = flat_patient_data
        subject["labels"] = {}

        for var in variables:
            subject["labels"][var] = one_hot_labels[var].iloc[index].to_numpy().tolist()

        # Store Subject in our list
        subjects.append(subject)

    # Return the list of subjects and the filtered annotations
    return subjects, data_overview


def load_mri_data_2D_all_atlases(
    # The path to the directory where the MRI data is stored (.csv file formats)
    data_paths: list,
    # The path to the CSV file that contains the filenames of the MRI data and the diagnoses and covariates
    csv_paths: str = None,
    # The annotations DataFrame that contains the filenames of the MRI data and the diagnoses and covariates
    annotations = None,
    # The diagnoses that you want to include in the data loading, defaults to all
    diagnoses = None,
    covars = [],
    hdf5: bool = True,
    train_or_test: str = None
) -> Tuple:
    print("test 1")

    if csv_paths is not None:
        for csv_path in csv_paths: 
            assert os.path.isfile(csv_path), f"CSV file '{csv_path}' not found"
            assert annotations is None, "Both CSV and annotations provided"
        print("test 2")
        # Initialize the data overview DataFrame
        data_overview = combine_dfs(csv_paths)

    # If the annotations are provided, make sure that they are a pandas DataFrame, and that the CSV path is not provided
    if annotations is not None:
        assert isinstance(
            annotations, pd.DataFrame
        ), "Annotations must be a pandas DataFrame"
        assert csv_paths is None, "Both CSV and annotations provided"
        print("test 3")
        # Initialize the data overview DataFrame
        data_overview = annotations
    print("test 4")

    # If no diagnoses are provided, use all diagnoses in the data overview
    if diagnoses is None:
        diagnoses = data_overview["Diagnosis"].unique().tolist()
    print("test 5")

    # If the covariates are not a list, make them a list
    if not isinstance(covars, list):
        covars = [covars]
    print("test 6")

    # If the diagnoses are not a list, make them a list
    if not isinstance(diagnoses, list):
        diagnoses = [diagnoses]
    print("test 7")
    
    # Set all the variables that will be one-hot encoded
    variables = ["Diagnosis"] + covars
    print("test 8")

    # Filter unwanted diagnoses
    data_overview = data_overview[data_overview["Diagnosis"].isin(diagnoses)]
    data_overview = data_overview.drop(columns=['Unnamed: 0'])
    data_overview = data_overview[["Filename", "Dataset", "Diagnosis" , "Age", "Sex", "Usage_original", "Sex_int"]]
    print("test 9")
    print(data_overview.shape)
    # produce one hot coded labels for each variable
    one_hot_labels = {}
    for var in variables:
        # check that the variables is in the data overview
        if var not in data_overview.columns:
            print("test 10")
            raise ValueError(f"Column '{var}' not found in CSV file or annotations")

        # one hot encode the variable
        one_hot_labels[var] = pd.get_dummies(data_overview[var], dtype=float)
    print("test 11")
    # For each subject, collect MRI data and variable data in the Subject object
    subjects = {}
    
    for data_path in data_paths:
        current_atlas = get_atlas(data_path)

        if hdf5 == True: 
            data = read_hdf5_to_df(filepath=data_path)
        else:
            data = pd.read_csv(data_path, header=[0, 1], index_col=0)
        
        data = normalize_and_scale_df(data)
        atlas_name = data_path.stem

        if train_or_test == "train":
            data.to_csv(f"/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/normative_results/train_processed_data_norm/Proc_{atlas_name}.csv")
            all_file_names = data.columns
        else:
            data.to_csv(f"/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/normative_results/test_processed_data_norm/Proc_{atlas_name}.csv")
            all_file_names = data.columns
        #data.to_csv(f"data/proc_extracted_xml_data/Proc_{atlas_name}_{train_or_test}.csv")
        all_file_names = data.columns

        for index, row in data_overview.iterrows():
            
            if not row["Filename"] in all_file_names:
                continue

            # Format correct filename
            if row["Filename"].endswith(".nii") or row["Filename"].endswith(".nii.gz"):
                pre_file_name = row["Filename"]
                match_no_ext = re.search(r"([^/\\]+)\.[^./\\]*$", pre_file_name)  # Extract file stem
                if match_no_ext:
                    file_name = match_no_ext.group(1)
            else:
                file_name = row["Filename"]

            if file_name in subjects: 
                if data_path in subjects[file_name]["measurements"]:
                    continue

            patient_data = data[file_name]
            flat_patient_data = flatten_array(patient_data).tolist()

            if not file_name in subjects: 
                subjects[file_name] = {"name": file_name,
                                    "measurements": flat_patient_data,
                                    "labels": {}
                                    }
            else: 
                subjects[file_name]["measurements"] += flat_patient_data

            for var in variables:
                subjects[file_name]["labels"][var] = one_hot_labels[var].iloc[index].to_numpy().tolist()

    # Return the list of subjects and the filtered annotations
    return list(subjects.values()), data_overview
  

def process_subjects(
    subjects: List[tio.Subject],
   # transforms: tio.Compose,
    batch_size: int,
    shuffle_data: bool,
) -> DataLoader:

    # Apply transformations
    dataset = CustomDataset_2D(subjects=subjects)

    # Create data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data)

    # Return the DataLoader object
    return data_loader


#process latent space into visualization
def process_latent_space_2D(
    # The AnnData object that contains the latent space representation of the MRI data.
    adata: ad.AnnData,
    # The annotations DataFrame that contains the meta data annotations for the MRI data.
    annotations: pd.DataFrame,
    # The number of neighbors for the UMAP calculation.
    umap_neighbors: int,
    # The random seed for reproducibility of the UMAP calculation.
    seed: int,
    # Name of the atlas / model whose latent space should be processed. 
    atlas_name,
    # Whether the data should be saved or not. If True, save_path, timestamp, epoch and data_type must be provided.
    save_data: bool = False,
    # The path where the data should be saved.
    save_path: str = None,
    # The timestamp of the current run, used for the filename.
    timestamp: str = None,
    # The current epoch, used for the filename.
    epoch: int = None,
    # The data type of adata (typically "training" or "validation"), used for the filename.
    data_type: str = None,
) -> ad.AnnData:

    # align annotation metadata with anndata.obs
    aligned_ann = annotations.set_index("Filename").reindex(adata.obs_names)

    # add annotation metadata to anndata.obs
    for col in aligned_ann.columns:
        adata.obs[col] = aligned_ann[col]

    # perform PCA, UMAP and neighbors calculations
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, umap_neighbors, use_rep="X")
    sc.tl.umap(adata, random_state=seed)

    # save the data if specified
    if save_data:
        adata.write_h5ad(
            os.path.join(save_path, f"{timestamp}_e{epoch}_latent_{data_type}.h5ad")
        )

    # return the processed data
    return adata


def combine_latent_spaces(
    tdata: ad.AnnData,
    vdata: ad.AnnData,
    umap_neighbors: int,
    seed: int,
    save_data: bool = False,
    save_path: str = None,
    timestamp: str = None,
    epoch: int = None,
    data_type: str = None,
) -> ad.AnnData:

    # concatenate the training and validation data
    cdata = ad.concat({"train": tdata, "valid": vdata}, join="outer", label="Data_Type")

    # perform PCA, UMAP and neighbors calculations
    sc.pp.pca(cdata)
    sc.pp.neighbors(cdata, umap_neighbors, use_rep="X")
    sc.tl.umap(cdata, random_state=seed)

    # save the data if specified
    if save_data:
        cdata.write_h5ad(
            os.path.join(save_path, f"{timestamp}_e{epoch}_latent_{data_type}.h5ad")
        )

    # return the combined data
    return cdata


def load_checkpoint_model(model, model_filename: str):

    if os.path.isfile(model_filename):
        log_and_print(f"Loading checkpoint from '{model_filename}'")

        model_state = torch.load(model_filename)
        model.load_state_dict(model_state)

        log_and_print(f"Checkpoint loaded successfully")

    else:
        log_and_print(
            f"No checkpoint found at '{model_filename}', starting training from scratch"
        )

        raise FileNotFoundError(f"No checkpoint found at '{model_filename}'")

    return model


# This function saves the model to a file.
# This function needs an overhaul to:
# - save model weights
# - save optimizer state
# - save scheduler state
# - save model metrics
# - save model hyperparameters
# Ideally all of this would be saved in a single file.
# Also this function probably belongs in the base model file.
def save_model(model, save_path: str, timestamp: str, descriptor: str, epoch: int):
    model_save_path = os.path.join(
        save_path,
        f"{timestamp}_{descriptor}_e{epoch}_model.pth",
    )

    torch.save(model.state_dict(), model_save_path)

    log_checkpoint(model_path=model_save_path)


# This function saves the model metrics to a csv file.
# It should probably be combined with the save_model function.
def save_model_metrics(model_metrics, save_path: str, timestamp: str, descriptor: str):
    metrics_save_path = os.path.join(
        save_path,
        f"{timestamp}_{descriptor}_model_performance.csv",
    )

    pd.DataFrame(model_metrics).to_csv(metrics_save_path, index=False)

    log_checkpoint(
        metrics_path=metrics_save_path,
    )