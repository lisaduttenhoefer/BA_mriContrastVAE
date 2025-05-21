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
import h5py
import torch
import torchio as tio
from torch.utils.data import DataLoader

from utils.logging_utils import log_and_print, log_checkpoint


def flatten_array(df: pd.DataFrame) -> np.ndarray:
    # Converts data frame to flattened array. 
    array = df.to_numpy()
    flat_array = array.flatten()
    return flat_array


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

def read_hdf5_to_df(filepath: str) -> pd.DataFrame:
    """
    Read HDF5 file and convert to pandas DataFrame.
    
    Parameters:
    - filepath: Path to the HDF5 file
    
    Returns:
    - DataFrame containing the HDF5 data
    """
    try:
        with h5py.File(filepath, 'r') as f:
           
            # Assuming HDF5 file contains 'data' and 'index' datasets
            if 'data' in f and 'index' in f and 'columns' in f:
                data = f['data'][:]
                index = [idx.decode('utf-8') if isinstance(idx, bytes) else idx for idx in f['index'][:]]
                columns = [col.decode('utf-8') if isinstance(col, bytes) else col for col in f['columns'][:]]
                
                df = pd.DataFrame(data, index=index, columns=columns)
            else:
                # Try to load with pandas directly
                df = pd.read_hdf(filepath)
                
            return df
    except Exception as e:
        print(f"[ERROR] Failed to load HDF5 file {filepath}: {e}")
        # Try pandas direct method as fallback
        try:
            df = pd.read_hdf(filepath)
            return df
        except Exception as e2:
            print(f"[ERROR] Fallback also failed: {e2}")
            raise

def read_hdf5_to_df_OLD(filepath: str):
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
    #datasets: List[str] = None,
    # The random seed for reproducibility
    seed: int = 123,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Initialize empty dataframes
    train = pd.DataFrame()
    valid = pd.DataFrame()

    # If diagnoses is a string, convert to list
    if isinstance(diagnoses, str):
        diagnoses = [diagnoses]
    # If no diagnoses are specified, use all diagnoses in the annotations
    elif diagnoses is None:
        diagnoses = annotations["Diagnosis"].unique().tolist()
    
    # Get all datasets in the annotations
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
    subjects: List[dict],
    train_ann: pd.DataFrame,
    val_ann: pd.DataFrame
) -> Tuple[List, List]:
    """
    Split subject data based on previously split annotations.
    
    Parameters:
    - subjects: List of dictionaries containing subject data
    - train_ann: Training annotations DataFrame
    - val_ann: Validation annotations DataFrame
    
    Returns:
    - Tuple of training and validation subject lists
    """
    # Get filenames from annotations
    train_files = set(train_ann["Filename"].str.replace('.nii.gz', '').str.replace('.nii', ''))
    valid_files = set(val_ann["Filename"].str.replace('.nii.gz', '').str.replace('.nii', ''))
    
    # Initialize lists for subjects
    train_subjects = []
    valid_subjects = []
    unmatched_subjects = []
    
    # Assign subjects to train or validation set
    for subject in subjects:
        subject_name = subject["name"]
        # Remove file extensions if present
        subject_name_clean = subject_name.replace('.nii.gz', '').replace('.nii', '')
        
        if subject_name_clean in train_files:
            train_subjects.append(subject)
        elif subject_name_clean in valid_files:
            valid_subjects.append(subject)
        else:
            unmatched_subjects.append(subject_name)
    
    print(f"[INFO] {len(train_subjects)} subjects in training set")
    print(f"[INFO] {len(valid_subjects)} subjects in validation set")
    
    if unmatched_subjects:
        print(f"[WARNING] {len(unmatched_subjects)} subjects not found in annotations:")
        for i, name in enumerate(unmatched_subjects[:5]):
            print(f"  - {name}")
        if len(unmatched_subjects) > 5:
            print(f"  ... and {len(unmatched_subjects) - 5} more")
    
    return train_subjects, valid_subjects

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


def read_hdf5_to_df(filepath: str) -> pd.DataFrame:
    """
    Read HDF5 file and convert to pandas DataFrame.
    
    Parameters:
    - filepath: Path to the HDF5 file
    
    Returns:
    - DataFrame containing the HDF5 data
    """
    import h5py
    
    try:
        with h5py.File(filepath, 'r') as f:
    
            # Assuming HDF5 file contains 'data' and 'index' datasets
            if 'data' in f and 'index' in f and 'columns' in f:
                data = f['data'][:]
                index = [idx.decode('utf-8') if isinstance(idx, bytes) else idx for idx in f['index'][:]]
                columns = [col.decode('utf-8') if isinstance(col, bytes) else col for col in f['columns'][:]]
                df = pd.DataFrame(data, index=index, columns=columns)
            else:
                # Try to load with pandas directly
                df = pd.read_hdf(filepath)
            return df
        
    except Exception as e:
        print(f"[ERROR] Failed to load HDF5 file {filepath}: {e}")
        
        # Try pandas direct method as fallback
        try:
            df = pd.read_hdf(filepath)
            return df
        except Exception as e2:
            print(f"[ERROR] Fallback also failed: {e2}")
            raise

def combine_dfs(csv_paths: List[str]) -> pd.DataFrame:
    """Combine multiple CSV files into a single DataFrame."""
    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def flatten_array(arr):
    """Flatten an array or series to 1D."""
    if isinstance(arr, pd.Series):
        return arr.values
    return arr.flatten() if hasattr(arr, 'flatten') else arr

def load_mri_data_2D(
    data_path: str,
    atlas_name: List[str] = None,
    csv_paths: list[str] = None,
    annotations: pd.DataFrame = None,
    diagnoses: List[str] = None,
    covars: List[str] = [],
    hdf5: bool = True,
    train_or_test: str = "train",
    save: bool = False,
    volume_type: str = None,
    valid_volume_types: List[str] = ["Vgm", "Vwm", "csf"],
) -> Tuple:
    
    # Define all available atlases
    all_available_atlases = ["cobra", "lpba40", "neuromorphometrics", "suit", "thalamic_nuclei", "thalamus"]
    
    # Ensure atlas_name is a list
    if not isinstance(atlas_name, list):
        atlas_name = [atlas_name]
    
    # Check for the special case of ["all"]
    if len(atlas_name) == 1 and atlas_name[0] == "all":
        atlas_name = all_available_atlases

    print(f"[INFO] Processing atlases: {atlas_name}")
   
    csv_paths = [path.strip("[]'\"") for path in csv_paths]
    # Handle CSV paths
    if csv_paths is not None:
        for csv_path in csv_paths: 
            assert os.path.isfile(csv_path), f"[ERROR] CSV file '{csv_path}' not found"
            assert annotations is None, "[ERROR] Both CSV and annotations provided"
        data_overview = combine_dfs(csv_paths)
        
    # Handle annotations
    elif annotations is not None:
        assert isinstance(annotations, pd.DataFrame), "[ERROR] Annotations must be a pandas DataFrame"
        assert csv_paths is None, "[ERROR] Both CSV and annotations provided"
        data_overview = annotations
        print("[INFO] Annotations loaded successfully.")
        
    else:
        raise ValueError("[ERROR] No CSV path or annotations provided!")
    
    data_overview = data_overview[data_overview["Diagnosis"].isin(diagnoses)]
    # Handling diagnoses filtering
    if diagnoses is None:
        diagnoses = data_overview["Diagnosis"].unique().tolist()
    
    # Handling covariates and diagnoses lists
    covars = [covars] if not isinstance(covars, list) else covars
    diagnoses = [diagnoses] if not isinstance(diagnoses, list) else diagnoses
    
    # Prepare one-hot encoded labels
    one_hot_labels = {}
    variables = ["Diagnosis"] + covars
    
    for var in variables:
        if var not in data_overview.columns:
            raise ValueError(f"[ERROR] Column '{var}' not found in CSV file or annotations")
        one_hot_labels[var] = pd.get_dummies(data_overview[var], dtype=float)
    
    # Data structures to store combined information across atlases
    subjects_dict = {}
    all_roi_names = []
    
    # Process each atlas in the list
    for atlas in atlas_name:
        print(f"[INFO] Processing atlas: {atlas}")
        atlas_data_path = f"{data_path}/Aggregated_{atlas}.h5"
        
        # Check if atlas file exists before attempting to read it
        if not os.path.exists(atlas_data_path):
            print(f"[ERROR] Atlas file not found: {atlas_data_path} - skipping this atlas")
            continue
       
        try:
            if hdf5:
                data = read_hdf5_to_df(filepath=atlas_data_path)
            else:
                data = pd.read_csv(atlas_data_path, header=[0, 1], index_col=0)
         
        except Exception as e:
            print(f"[ERROR] Loading failed {atlas}; reason: {str(e)} -> skipping")
            continue
        
        # Extract unique patient IDs before flattening
        if isinstance(data.columns, pd.MultiIndex):
            all_file_names = data.columns.get_level_values(0).unique()
        else:
            all_file_names = data.columns
           
        # Extract ROI names for this atlas
        base_roi_names = data.index.tolist()
        
        # Validate volume type
        if volume_type != "all" and volume_type not in valid_volume_types:
            raise ValueError(f"[ERROR] Invalid volume_type: {volume_type}")
        
        atlas_roi_names = []
        if volume_type == "all":
            atlas_roi_names = [f"{atlas}_{roi}_{vt}" for roi in base_roi_names for vt in valid_volume_types]
        else:
            atlas_roi_names = [f"{atlas}_{roi}_{volume_type}" for roi in base_roi_names]
        
        # Add this atlas's ROI names to the overall list
        all_roi_names.extend(atlas_roi_names)
        print(f"[INFO] Added {len(atlas_roi_names)} ROI names from atlas {atlas}")
        
        # Normalize and scale data
        data = normalize_and_scale_df(data)
        
        # *** FLATTENING MULTIINDEX EARLIER IN THE PROCESS ***
        if isinstance(data.columns, pd.MultiIndex):
            # Filter by volume_type if specified
            if volume_type and volume_type != "all":
                # Get only columns matching the specified volume_type
                filtered_columns = [(patient, vol) for patient, vol in data.columns if vol == volume_type]
                if filtered_columns:
                    data = data[filtered_columns]
                else:
                    print(f"[ERROR] No columns found for volume_type {volume_type}")
            
            # Now flatten MultiIndex columns to patient_volumetype format
            flattened_columns = [f"{patient}_{volume}" for patient, volume in data.columns]
            data.columns = flattened_columns
        
        # Save processed data if needed
        if save:
            volume_suffix = "_all" if volume_type == "all" else f"_{volume_type}"
            save_path = f"data/proc_extracted_xml_data/Proc_{atlas}{volume_suffix}_{train_or_test}.csv"
            data.to_csv(save_path)
        
        # Process each subject for this atlas
        for index, row in data_overview.iterrows():
            file_name = re.sub(r"\.[^.]+$", "", row["Filename"])
            
            # Check if file_name is in all_file_names (might need to handle edge cases)
            if isinstance(all_file_names, pd.Index):
                file_found = any(file_name in fn for fn in all_file_names)
            else:
                file_found = file_name in all_file_names
                
            if not file_found:
                print(f"[ERROR] Filename {file_name} not found in MRI data for atlas {atlas}.")
                continue
            
            # Select patient data from the already flattened data frame
            patient_columns = [col for col in data.columns if col.startswith(f"{file_name}_")]
            
            if not patient_columns:
                print(f"[ERROR] No columns for patient {file_name} after flattening in atlas {atlas}.")
                continue
                
            patient_data = data[patient_columns]
            flat_patient_data = flatten_array(patient_data).to_numpy().tolist()
            
            # Create/update subject entry in the dictionary
            if file_name not in subjects_dict:
                subjects_dict[file_name] = {
                    "name": file_name,
                    "measurements": flat_patient_data,  # Start with this atlas's measurements
                    "labels": {var: one_hot_labels[var].iloc[index].to_numpy().tolist() for var in variables}
                }
            else:
                # Append this atlas's measurements to existing ones
                subjects_dict[file_name]["measurements"] += flat_patient_data
        
    # Make sure we found at least one valid atlas
    if not subjects_dict:
        raise ValueError("[ERROR] No valid data was processed from any atlas!")
    
    # Convert dictionary to list of subjects
    subjects = list(subjects_dict.values())
    print(f"[INFO] Total subjects processed across all atlases: {len(subjects)}")
    print("[INFO] Data loading complete!")
    
    return subjects, data_overview, all_roi_names

def load_mri_data_2D_all_atlases(
    # The base path to the directory where the MRI data is stored
    data_path: str,
    # Names of the atlases that should be used for training
    atlas_names: List[str],
    # The path to the CSV metadata files
    csv_paths: List[str] = None,
    # The annotations DataFrame that contains the filenames of the MRI data and the diagnoses and covariates
    annotations: pd.DataFrame = None,
    # The diagnoses that you want to include in the data loading, defaults to all
    diagnoses: List[str] = None,
    # Covariates to include
    covars: List[str] = [],
    # Are the files of raw extracted xml data in the .h5 (True) or .csv (False) format?
    hdf5: bool = True,
    # Intended usage for the currently loaded data.
    train_or_test: str = "train",
    # Should the normalized and scaled file be saved?
    save: bool = False,
    # Which volume type to use for analysis (Vgm, Vwm, csf, or all)
    volume_type: str = None,
    # List of valid volume types that can be specified
    valid_volume_types: List[str] = ["Vgm", "Vwm", "csf"],
) -> Tuple:
    """
    Load and process MRI data from multiple atlases.
    
    Returns:
        Tuple: A tuple containing the list of subject data, filtered annotations DataFrame, and list of ROI names
    """
    
    # Validate volume_type parameter
    if volume_type != "all" and volume_type not in valid_volume_types:
        raise ValueError(f"[ERROR] Invalid volume_type: {volume_type}. Must be one of: {valid_volume_types} or 'all'")
    
    # If the CSV path is provided, check if the file exists, make sure that the annotations are not provided
    if csv_paths is not None:
        for csv_path in csv_paths: 
            assert os.path.isfile(csv_path), f"[ERROR] CSV file '{csv_path}' not found"
            assert annotations is None, "[ERROR] Both CSV and annotations provided"

    # If the annotations are provided, make sure that they are a pandas DataFrame, and that the CSV path is not provided
    elif annotations is not None:
        assert isinstance(
            annotations, pd.DataFrame
        )
        assert csv_paths is None
        # Initialize the data overview DataFrame
        data_overview = annotations
        
    # If no diagnoses are provided, use all diagnoses in the data overview
    if diagnoses is None:
        diagnoses = data_overview["Diagnosis"].unique().tolist()
        
    # Filter unwanted diagnoses
    data_overview = data_overview[data_overview["Diagnosis"].isin(diagnoses)]
    print(data_overview["Diagnosis"].value_counts())
    
    # Handle data columns similar to load_mri_data_2D
    try:
        data_overview = data_overview[["Filename", "Dataset", "Diagnosis", "Age", "Sex", "Usage_original", "Sex_int"]]
    except KeyError as e:
        print(f"[ERROR] Could not select all columns: {e}. Using available columns.")
    
    # If the covariates are not a list, make them a list
    covars = [covars] if not isinstance(covars, list) else covars

    # If the diagnoses are not a list, make them a list
    diagnoses = [diagnoses] if not isinstance(diagnoses, list) else diagnoses
    
    # Set all the variables that will be one-hot encoded
    variables = ["Diagnosis"] + covars

    # Prepare one-hot encoded labels
    one_hot_labels = {}
    for var in variables:
        # check that the variables is in the data overview
        if var not in data_overview.columns:
            raise ValueError(f"[ERROR] Column '{var}' not found in CSV file or annotations")

        # one hot encode the variable
        one_hot_labels[var] = pd.get_dummies(data_overview[var], dtype=float)

    # For each subject, collect MRI data and variable data in the Subject object
    subjects_dict = {}
    all_roi_names = []

    # Process each atlas in the list
    for atlas in atlas_names:
        atlas_data_path = f"{data_path}/Aggregated_{atlas}.h5"
        
        if hdf5:
            data = read_hdf5_to_df(filepath=atlas_data_path)
        else:
            data = pd.read_csv(atlas_data_path, header=[0, 1], index_col=0)
        
        # Extract unique patient IDs before flattening
        if isinstance(data.columns, pd.MultiIndex):
            all_file_names = data.columns.get_level_values(0).unique()
        else:
            all_file_names = data.columns
            
        # Extract ROI names for this atlas
        base_roi_names = data.index.tolist()
        
        atlas_roi_names = []
        if volume_type == "all":
            atlas_roi_names = [f"{atlas}_{roi}_{vt}" for roi in base_roi_names for vt in valid_volume_types]
        else:
            atlas_roi_names = [f"{atlas}_{roi}_{volume_type}" for roi in base_roi_names]
        
        # Add this atlas's ROI names to the overall list
        all_roi_names.extend(atlas_roi_names)
        
        # Handle MultiIndex columns earlier in the process like in load_mri_data_2D
        if isinstance(data.columns, pd.MultiIndex):
            # Filter by volume_type if specified
            if volume_type and volume_type != "all":
                # Get only columns matching the specified volume_type
                filtered_columns = [(patient, vol) for patient, vol in data.columns if vol == volume_type]
                if filtered_columns:
                    data = data[filtered_columns]
                
            # Now flatten MultiIndex columns to patient_volumetype format
            flattened_columns = [f"{patient}_{volume}" for patient, volume in data.columns]
            data.columns = flattened_columns
            
        # Normalize and scale data
        data = normalize_and_scale_df(data)
        
        # Save processed data if needed
        if save:
            volume_suffix = "_all" if volume_type == "all" else f"_{volume_type}"
            save_path = f"data/proc_extracted_xml_data/Proc_{atlas}{volume_suffix}_{train_or_test}.csv"
            data.to_csv(save_path)
            
        # Process each subject for this atlas
        for index, row in data_overview.iterrows():
            file_name = re.sub(r"\.[^.]+$", "", row["Filename"])
            
            if file_name not in all_file_names:
            
                continue
            
            # Select patient data from the already flattened data frame
            patient_columns = [col for col in data.columns if col.startswith(f"{file_name}_")]
            
            if not patient_columns:

                continue
                
            patient_data = data[patient_columns]
            flat_patient_data = flatten_array(patient_data).to_numpy().tolist()
            
            # Create/update subject entry in the dictionary
            if file_name not in subjects_dict:
                subjects_dict[file_name] = {
                    "name": file_name,
                    "measurements": flat_patient_data,  # Start with this atlas's measurements
                    "labels": {var: one_hot_labels[var].iloc[index].to_numpy().tolist() for var in variables}
                }
            else:
                # Append this atlas's measurements to existing ones
                subjects_dict[file_name]["measurements"] += flat_patient_data
        
    
    # Convert dictionary to list of subjects
    subjects = list(subjects_dict.values())
    print("[INFO] Data loading complete!")
    
    return subjects, data_overview, all_roi_names

def process_subjects(
    subjects: List[tio.Subject],
   # transforms: tio.Compose,
    batch_size: int,
    shuffle_data: bool,
) -> DataLoader:

    # Apply transformations
    dataset = CustomDataset_2D(subjects=subjects)

    # Create data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data, num_workers=4,
        pin_memory=True)

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

def save_model(model, save_path: str, timestamp: str, descriptor: str, epoch: int):
    model_save_path = os.path.join(
        save_path,
        f"{timestamp}_{descriptor}_e{epoch}_model.pth",
    )

    torch.save(model.state_dict(), model_save_path)

    log_checkpoint(model_path=model_save_path)


def save_model_metrics(model_metrics, save_path: str, timestamp: str, descriptor: str):
    metrics_save_path = os.path.join(
        save_path,
        f"{timestamp}_{descriptor}_model_performance.csv",
    )

    pd.DataFrame(model_metrics).to_csv(metrics_save_path, index=False)

    log_checkpoint(
        metrics_path=metrics_save_path,
    )