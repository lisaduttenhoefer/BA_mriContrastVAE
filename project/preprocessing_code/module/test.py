import pandas as pd 
import os
import pathlib
import numpy as np

def read_hdf5_to_df(filepath: str):
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist")
        return None
    try:
        return pd.read_hdf(filepath, key='atlas_data')
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

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


path = "./xml_data/Aggregated_cobra.h5"
df = read_hdf5_to_df(path)
df_norm = normalize_and_scale_df(df)
df_norm.to_csv(pathlib.PosixPath("./processed_data/test.csv"))

# csv_paths = ["./metadata_20250110/full_data_train_valid_test.csv",
#                       "./metadata_20250110/meta_data_NSS_all_variables.csv",
#                       "./metadata_20250110/meta_data_whiteCAT_all_variables.csv"]


# data_overview = combine_dfs(csv_paths)

# variables = ["Diagnosis"]

# one_hot_labels = {}
# for var in variables:
#     # check that the variables is in the data overview
#     if var not in data_overview.columns:
#         raise ValueError(f"Column '{var}' not found in CSV file or annotations")

#     # one hot encode the variable
#     one_hot_labels[var] = pd.get_dummies(data_overview[var], dtype=float)

# print(one_hot_labels["Diagnosis"])

# print(one_hot_labels["Diagnosis"].iloc[3170])
        