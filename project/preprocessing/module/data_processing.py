import os
# import torch  # How to check nvidia-smi version???
import pandas as pd
import regex as re
from typing import List, Tuple
import numpy as np
import pathlib


def flatten_array(df: pd.DataFrame) -> np.ndarray:
    # Converts data frame to flattened array. 
    array = df.to_numpy()
    flat_array = array.flatten()
    return flat_array


def normalize_and_scale_df(df: pd.DataFrame) -> pd.DataFrame:
    # Normalizes the columns (patient volumes) by Min-Max Scaling and scales the rows (ROIs) with Z-transformation.

    df_copy = df.copy()
    column_sums = df_copy.sum()
    
    # Apply the formula: ln((10000*value)/sum_values + 1) "Log transformation"
    # Alternatively for Min-Max Scaling: df_copy/df_copy.max() - Problem: Some rows have std = 0
    transformed_df = np.log((10000 * df_copy) / column_sums + 1)
    
    norm_copy = transformed_df.copy()

    cols = norm_copy.columns.get_level_values(-1).tolist()
    unique_cols = list(set(cols))

    for col_type in unique_cols:
        cols_to_scale = [col for col in norm_copy.columns if col[-1] == col_type]

        # Scale the selected columns per row
        scaled = norm_copy[cols_to_scale].apply(
            lambda row: (row - row.mean()) / row.std() if row.std() > 0 else pd.Series(0.0, index=row.index),
            axis=1
        )
        
        norm_copy.loc[:, cols_to_scale] = scaled
        
    return norm_copy


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
    for i in range(1,len(paths)):
        if i == 1: 
            joined_df = pd.read_csv(paths[i-1], header=[0], index_col=0)
            next_df = pd.read_csv(paths[i], header=[0], index_col=0)
            joined_df = pd.concat([joined_df, next_df], join="inner")  # Parameter "inner" keeps only the shared column indices.
        else:
            next_df = pd.read_csv(paths[i], header=[0], index_col=0)
            joined_df = pd.concat([joined_df, next_df], join="inner")
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


# class CustomDataset(Dataset):  # Create Datasets that can then be converted into DataLoader objects
#     def __init__(self, subjects, transforms=None):
#         self.subjects = subjects
#         self.transforms = transforms

#     def __len__(self):
#         return len(self.subjects)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         measurements = self.subjects[idx]["measurements"]
#         labels = self.subjects[idx]["labels"]

#         if self.transform:
#             transformed = self.transforms(measurements=measurements)
#             measurements = transformed['measurements']

#         measurements = torch.as_tensor(measurements, dtype=torch.float64)
#         labels = torch.as_tensor(labels, dtype=torch.int64)

#         return measurements, labels


def load_mri_data_2D(
    # The path to the directory where the MRI data is stored (.csv file formats)
    data_path: str,
    # The path to the CSV file that contains the filenames of the MRI data and the diagnoses and covariates
    csv_paths: list = None,
    # The annotations DataFrame that contains the filenames of the MRI data and the diagnoses and covariates
    annotations: pd.DataFrame = None,
    # The diagnoses that you want to include in the data loading, defaults to all
    diagnoses: List[str] = None,
    covars: List[str] = [],
    hdf5: bool = True
) -> Tuple:

    data_path = pathlib.Path(data_path)

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
    #if diagnoses is None:
    #    diagnoses = data_overview["Diagnosis"].unique().tolist()


    if diagnosis != ["HC"]:
        diagnoses = data_overview["Diagnosis"].unique().tolist()
        diagnoses = [d for d in diagnoses if d != "HC"]
    else:
        diagnoses = ["HC"]

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
        data = read_hdf5_to_df(filepath=data_path)
    else:
        data = pd.read_csv(data_path, header=[0, 1], index_col=0)

    # if train:
    #         data.to_csv(f".data/train_processed_data/Proc_{atlas_name}.csv")
    #         all_file_names = data.columns
    # else:
    #     data.to_csv(f".data/test_processed_data/Proc_{atlas_name}.csv")
    #     all_file_names = data.columns
    
    # data.set_index("Filename", inplace=True)
    data = normalize_and_scale_df(data)

    atlas_name = data_path.stem
    data.to_csv(f"{config.PROCESSED_CSV_DIR}/Proc_{atlas_name}.csv")
        
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
    train: bool = True,
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
        
        # if train:
        #     data.to_csv(f".data/train_processed_data/Proc_{atlas_name}.csv")
        #     all_file_names = data.columns
        # else:
        #     data.to_csv(f".data/test_processed_data/Proc_{atlas_name}.csv")
        #     all_file_names = data.columns

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
                                       "measurements": {current_atlas: flat_patient_data},
                                       "labels": {}
                                      }
            else: 
                subjects[file_name]["measurements"][current_atlas] = flat_patient_data
            

            for var in variables:
                subjects[file_name]["labels"][var] = one_hot_labels[var].loc[index].to_numpy().tolist()



    # Return the list of subjects and the filtered annotations
    return list(subjects.values()), data_overview


def load_mri_data_3D_all_atlases(
    # The path to the directory where the MRI data is stored (.csv file formats)
    data_paths: list,
    # The path to the CSV file that contains the filenames of the MRI data and the diagnoses and covariates
    csv_paths: str = None,
    # The annotations DataFrame that contains the filenames of the MRI data and the diagnoses and covariates
    annotations = None,
    # The diagnoses that you want to include in the data loading, defaults to all
    diagnoses = None,
    covars = [],
    hdf5: bool = True
    ) -> Tuple:
    return print("testing")

# # This function processes a list of subjects by applying a series of transformations to them, and then loads
# # them into a DataLoader object.
# def process_subjects(
#     # The list of tio.Subject objects that you want to process
#     subjects: List,
#     # The transformations that you want to apply to the subjects
#     transforms: torch.compose,
#     # The batch size for the DataLoader (the larger, the more memory is needed)
#     batch_size: int,
#     # Whether the data should be shuffled or not. Shuffling is important for training, but not for validation.
#     shuffle_data: bool,
# ) -> DataLoader:

#     # Apply transformations
#     dataset = CustomDataset(subjects=subjects, transforms=transform)

#     # Create data loader
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data)

#     # Return the DataLoader object
#     return data_loader


if __name__ == "__main__":
    
    if not os.path.exists("./processed_data"):
        os.makedirs("./processed_data")
    
    # subjects, overview = load_mri_data_2D(data_path=pathlib.Path("./xml_data/Aggregated_suit.csv"),
    #                                       csv_path="./metadata_20250110/full_data_train_valid_test.csv")
    #print("\nSubjects for one atlas:\n")
    #print(subjects)

    data_paths = get_all_data(directory="/net/data.isilon/ag-cherrmann/nschmidt/project/parse_xml_for_VAE/xml_data", ext="h5")
    #print(data_paths)
    subjects_all, data_overview = load_mri_data_2D_all_atlases(data_paths=data_paths,
                                                               csv_paths=["./metadata_20250110/full_data_train_valid_test.csv",
                                                                          "./metadata_20250110/meta_data_NSS_all_variables.csv",
                                                                          "./metadata_20250110/meta_data_whiteCAT_all_variables.csv"],
                                                               hdf5=True)
    #print("\nSubjects with all atlases:\n")
    #print(subjects_all)
    
    