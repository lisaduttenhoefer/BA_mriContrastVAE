import pathlib
import pandas as pd
import regex as re
import os
import numpy as np


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

    cols = norm_copy.columns.get_level_values(-1).tolist() # Select lowest level of Multiindex (Measurements: Vgm, Vwm, Vcsf)
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
        df = pd.read_hdf(filepath, key='atlas_data')
        return df 
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
    
# def split_df(path_original: str, path_to_dir: str):

#     df = pd.read_csv(path_original, header=[0])
#     df = df.drop(columns=["Unnamed: 0"])

#     for option in ["training", "testing"]:
#         if option == "training":
#             path = f"{path_to_dir}/{option}_metadata.csv"
#             subset_df = df[df["Usage_original"] != "testing"]
#             subset_df.to_csv(path)
#         elif option == "testing":
#             path = f"{path_to_dir}/{option}_metadata.csv"
#             subset_df = df[df["Usage_original"] == option]
#             subset_df.to_csv(path)
#changed to hc/others split

import pandas as pd
import os

def split_df(path_original: str, path_to_dir: str):
    # Lade die ursprüngliche CSV-Datei
    df = pd.read_csv(path_original, header=[0])
    df = df.drop(columns=["Unnamed: 0"])  # Entferne die unnötige Spalte

    # Erstelle das Trainings-CSV für HC
    path_hc = f"{path_to_dir}/hc_metadata.csv"
    subset_hc = df[df["Diagnosis"] == "HC"]
    subset_hc.to_csv(path_hc, index=False)  # Speichern ohne Index
    
    # Erstelle das Test-CSV für non-HC
    path_non_hc = f"{path_to_dir}/non_hc_metadata.csv"
    subset_non_hc = df[df["Diagnosis"] != "HC"]
    subset_non_hc.to_csv(path_non_hc, index=False)  # Speichern ohne Index
    
    # Rückgabe der Pfade der erzeugten CSV-Dateien
    return path_hc, path_non_hc




    