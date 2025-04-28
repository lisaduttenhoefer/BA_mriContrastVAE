import numpy as np 
import pytest
import xml.etree.ElementTree as ET
import pandas as pd
import os
import pathlib
import regex as re
import time


def add_to_gitignore(path: str):
    with open(".gitignore", "a") as g:
        g.write("\n")
        g.write(f"{path}")
    return


def remove_paths_containing(list_of_paths: list, keys: list):
    filtered_paths = []
    for path in list_of_paths:
        should_include = True
        for folder in keys:
            if folder + os.sep in path or folder in os.path.basename(path):
                should_include = False
                break  # If one excluded folder is found, no need to check others for this path
        if should_include:
            filtered_paths.append(path)
    return filtered_paths


def keep_paths_containing(list_of_paths: list, keys: list):
    filtered_paths = []
    for path in list_of_paths:
        should_include = False
        for folder in keys:
            if folder + os.sep in path or folder in os.path.basename(path):
                should_include = True
                break  # If one desired folder is found, no need to check others for this path
        if should_include:
            filtered_paths.append(path)
    return filtered_paths


def xml_parser(path_to_xml_file: str) -> dict:
    # Parses .xml file to extract from each atlas the ROI names ("names") and the corresponding measurements ("data")
    tree = ET.parse(path_to_xml_file)
    root = tree.getroot()
    results = {}

    for section in root: # Each atlas represents one section
        section_name = section.tag
        results[section_name] = {} # Save data for each atlas separately

        names_element = section.find("names")
        if names_element is not None: 
            names = []
            for item in names_element.findall("item"):
                names.append(item.text)
            results[section_name]["names"] = names
        
        data_element = section.find("data")
        if data_element is not None:
            for data_type in data_element: 
                data_tag = data_type.tag
                data = [float(val) for val in data_type.text.strip("[]").split(";")] # Convert string into a list of floats
                results[section_name][data_tag] = data

    return results


def dict_to_df(data_dict: dict, patient: str, ext: str, train: bool = True, config=None):
    # Converts the dict of atlases into separate pandas DataFrames and saves these each to
    # a csv file (once with rows as features and once with columns as features).
    
    for k, v in data_dict.items():  # k is the atlas, v is the data in the atlas
        
        if config:
            # Use config paths if provided
            if train == True: 
                filepath = f"{config.EXTRACTED_CSV_DIR}/Aggregated_{k}.{ext}"
                filepath_t = f"{config.EXTRACTED_CSV_T_DIR}/Aggregated_{k}_t.{ext}"
            else:
                filepath = f"{config.EXTRACTED_CSV_DIR}/Aggregated_{k}.{ext}"
                filepath_t = f"{config.EXTRACTED_CSV_T_DIR}/Aggregated_{k}_t.{ext}"
        else:
            # Fallback to original behavior
            if train == True: 
                filepath = f"./train_xml_data/Aggregated_{k}.{ext}"
                filepath_t = f"./train_xml_data_t/Aggregated_{k}_t.{ext}"
            else:
                filepath = f"./test_xml_data/Aggregated_{k}.{ext}"
                filepath_t = f"./test_xml_data_t/Aggregated_{k}_t.{ext}"

        volumes = [vs for vs in v.keys() if vs != "names"]  # Measurements are volumes of white and gray matter

        arrays = [[patient]*len(volumes), volumes]

        tuples = list(zip(*arrays))

        index = pd.MultiIndex.from_tuples(tuples, names = ["Filename", "Volume"])

        if "names" not in v:
            print(f"No names found in section {k}, skipping.")
            continue
       
        data = {volume: v[volume] for volume in volumes}
        df_new = pd.DataFrame(data, index=v["names"])
        df_new.columns = index

        # Check if file exists and has content
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            # If file doesn't exist or is empty, just save the new dataframe
            df_new.to_csv(filepath)  # feature columns
            df_new_t = df_new.T
            df_new_t.to_csv(filepath_t)  # feature rows
        else:
            try:
                # Read existing file with proper MultiIndex handling
                df_existing = pd.read_csv(filepath, header=[0, 1], index_col=0)
                
                # Concatenate horizontally while preserving MultiIndex. To understand concatenation see process all paths function.
                result = pd.concat([df_existing, df_new], axis=1)
                # result.set_index(keys="Filename", inplace=True)
                # Save with MultiIndex preserved
                result.to_csv(filepath)  # feature columns
                result_t = result.T
                result_t.to_csv(filepath_t)  # feature rows
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                # Fallback - just write the new data
                df_new.to_csv(filepath)  # feature columns
                df_new_t = df_new.T
                df_new_t.to_csv(filepath_t)  # feature rows
    return


def dict_to_hdf5(data_dict: dict, patient: str, ext: str, train: bool = True, config=None):
    # Converts the dict of atlases into separate pandas DataFrames and saves these each to
    # a h5 file (once with rows as features and once with columns as features). h5 format allows computationally more efficient processing.
    for k, v in data_dict.items():  # k is the atlas, v is the data in the atlas
        
        if config:
            # Use config paths if provided
            if train == True: 
                filepath = f"{config.EXTRACTED_CSV_DIR}/Aggregated_{k}.{ext}"
                filepath_t = f"{config.EXTRACTED_CSV_T_DIR}/Aggregated_{k}_t.{ext}"
            else:
                filepath = f"{config.EXTRACTED_CSV_DIR}/Aggregated_{k}.{ext}"
                filepath_t = f"{config.EXTRACTED_CSV_T_DIR}/Aggregated_{k}_t.{ext}"
        else:
            # Fallback to original behavior
            if train == True: 
                filepath = f"./train_xml_data/Aggregated_{k}.{ext}"
                filepath_t = f"./train_xml_data_t/Aggregated_{k}_t.{ext}"
            else:
                filepath = f"./test_xml_data/Aggregated_{k}.{ext}"
                filepath_t = f"./test_xml_data_t/Aggregated_{k}_t.{ext}"

# Sicherstellen, dass beide Ordner existieren
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        os.makedirs(os.path.dirname(filepath_t), exist_ok=True)
        
        volumes = [vs for vs in v.keys() if vs != "names"]  # Measurements are volumes of white and gray matter

        if "names" not in v:
            print(f"No names found in section {k}, skipping.")
            continue
            
        # Create MultiIndex for columns
        arrays = [[patient]*len(volumes), volumes]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=["Filename", "Volume"])
        
        # Create DataFrame with the new data
        data = {volume: v[volume] for volume in volumes}
        df_new = pd.DataFrame(data, index=v["names"])
        df_new.columns = index
        
        # Check if file exists
        df_new_t = df_new.T  # Transposed version
        
        # Process regular version
        if not os.path.exists(filepath):
            # If file doesn't exist, create it and save the dataframe
            df_new.to_hdf(filepath, key='atlas_data', mode='w')
        else:
            try:
                # Read the existing dataframe
                df_existing = pd.read_hdf(filepath, key='atlas_data')
                
                # Check if patient already exists
                patient_cols = [col for col in df_existing.columns if col[0] == patient]
                if patient_cols:
                    # Drop existing patient data
                    df_existing = df_existing.drop(columns=patient_cols)
                    
                # Combine the existing data with new data
                result = pd.concat([df_existing, df_new], axis=1)
                
                # Save updated dataframe
                result.to_hdf(filepath, key='atlas_data', mode='w')
                
            except Exception as e:
                print(f"Error processing regular file {filepath}: {e}")
                # Fallback - just write the new data
                df_new.to_hdf(filepath, key='atlas_data', mode='w')
        
        # Process transposed version
        if not os.path.exists(filepath_t):
            # If file doesn't exist, create it and save the dataframe
            df_new_t.to_hdf(filepath_t, key='atlas_data_t', mode='w')
        else:
            try:
                # Read the existing dataframe
                df_existing_t = pd.read_hdf(filepath_t, key='atlas_data_t')
                
                # Remove existing patient data if it exists
                if patient in df_existing_t.index.get_level_values(0):
                    df_existing_t = df_existing_t.drop(index=patient, level=0, errors='ignore')
                
                # Combine the existing data with new data
                result_t = pd.concat([df_existing_t, df_new_t], axis=0)
                
                # Save updated dataframe
                result_t.to_hdf(filepath_t, key='atlas_data_t', mode='w')
                
            except Exception as e:
                print(f"Error processing transposed file {filepath_t}: {e}")
                # Fallback - just write the new data
                df_new_t.to_hdf(filepath_t, key='atlas_data_t', mode='w')


# def get_all_xml_paths(directory: str, valid_patients: list, train: bool = True, test_data: list = None) -> list:
#     # Finds all xml paths in the directory for which there is also a marker in the metadata.
#     if train == True: 
#         assert test_data is not None, "Provide names of folders with desired testing data to exclude!"
        
#         xml_paths = pathlib.Path(directory).rglob("label/*.xml")
#         xml_paths = list(xml_paths)
#         xml_paths = [str(i) for i in xml_paths]

#         xml_paths = remove_paths_containing(list_of_paths=xml_paths, keys=test_data)

#     if train == False: 
#         assert test_data is not None, "Provide names of folders with desired testing data to include!"

#         xml_paths = pathlib.Path(directory).rglob("label/*.xml")  # rglob searches in all subdirectories
#         xml_paths = list(xml_paths)
#         xml_paths = [str(i) for i in xml_paths]  # Convert PosixPath to string to allow iteration

#         xml_paths = keep_paths_containing(list_of_paths=xml_paths, keys=test_data)

#     partial_set = set(valid_patients)
#     filtered_paths = [
#         xml_path for xml_path in xml_paths
#         if any(partial_path in xml_path for partial_path in partial_set)
#     ]
    
#     return filtered_paths

def get_all_xml_paths(directory: str, valid_patients: list, metadata_paths: list, train: bool = True) -> list:
    """
    Finds all xml paths in the directory based on diagnosis criteria
    
    Args:
        directory: Directory to search for XML files
        valid_patients: List of valid patient IDs
        metadata_paths: Paths to metadata CSV files with diagnosis information
        train: If True, get non-HC patients for training; if False, get HC patients for testing
        
    Returns:
        List of filtered XML paths
    """
    # First, get all xml paths
    xml_paths = pathlib.Path(directory).rglob("label/*.xml")
    xml_paths = [str(i) for i in list(xml_paths)]
    
    # Load metadata to get diagnosis information
    diagnosis_data = {}
    for path in metadata_paths:
        df = pd.read_csv(path)
        if 'Filename' in df.columns and 'Diagnosis' in df.columns:
            for _, row in df.iterrows():
                diagnosis_data[row['Filename']] = row['Diagnosis']
    
    # Filter by valid patients
    partial_set = set(valid_patients)
    filtered_paths = [
        xml_path for xml_path in xml_paths
        if any(partial_path in xml_path for partial_path in partial_set)
    ]
    
    # Further filter by diagnosis
    diagnosis_filtered = []
    for path in filtered_paths:
        # Extract patient ID from path
        match_no_ext = re.search(r"([^/\\]+)\.[^./\\]*$", path)
        if match_no_ext:
            patient_id = match_no_ext.group(1)
        
        new_match = re.search(r"catROI_(.+)", patient_id)
        if new_match:
            patient_id = new_match.group(1)
            
        # Check if patient is in diagnosis data
        if patient_id in diagnosis_data:
            is_hc = diagnosis_data[patient_id] == "HC"
            
            if (train and is_hc) or (not train and not is_hc):
                diagnosis_filtered.append(path)
    
    return diagnosis_filtered


def process_all_paths(directory: str, valid_patients: list, metadata_paths: list, 
                      batch_size: int = 10, hdf5: bool = True, train: bool = True, config=None):
    # Convert xml files to csv/h5 files filtered by diagnosis
    paths = get_all_xml_paths(directory, valid_patients, metadata_paths, train)
    print(f"Found a total of {len(paths)} valid {'HC' if train else 'non-HC'} patient .xml files.")

    # First, identify all section types that will be processed in next step
    section_types = set()
    for path in paths:
        parsed_dict = xml_parser(path)
        section_types.update(parsed_dict.keys())
    
    # # Clear existing files at the beginning
    # for section in section_types:
    #     filepath = f"./xml_data/Aggregated_{section}.csv"
    #     if os.path.exists(filepath):
    #         # Clear the file by opening in write mode
    #         with open(filepath, "w+") as f:
    #             f.close()
    
    # Each xml file is handled and saved separately, before moving to next. 
    # Importantly, the results of every new patient is concatenated to the existing aggregated atlas. 
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(paths)-1)//batch_size + 1} ({len(batch_paths)} files)")
        
        start = time.perf_counter()

        for idx, path in enumerate(batch_paths):  
            # print(f"Processing file {idx+1}/{len(paths)}: {path}")
            parsed_dict = xml_parser(path)

            match_no_ext = re.search(r"([^/\\]+)\.[^./\\]*$", path)  # Extract file stem
            if match_no_ext:
                patient_id = match_no_ext.group(1)
            
            new_match = re.search(r"catROI_(.+)", patient_id)  # Extract file ID
            if new_match:
                patient_id = new_match.group(1)

            if hdf5 == True:
                dict_to_hdf5(parsed_dict, patient=patient_id, train=train, ext="h5", config=config)
            else: 
                dict_to_df(parsed_dict, patient=patient_id, train=train, ext="csv", config=config)
        
        stop = time.perf_counter()
        print(f"Elapsed time for batch: {stop-start}")
    return



def valid_patients(paths: list) -> list:
    # From the metadata, get all file identifiers to filter xml files. 

    valid_list = []
    for path in paths: 
        df = pd.read_csv(path) 
        list_of_patients = df['Filename'].tolist()
        valid_list += list_of_patients
    return valid_list


if __name__ == "__main__":
    directory = "./testing_files"
    add_to_gitignore(path="xml_data")
    add_to_gitignore(path="xml_data_t")

    if not os.path.exists("./xml_data"):
        os.makedirs("./xml_data")

    if not os.path.exists("./xml_data_t"):
        os.makedirs("./xml_data_t")

    # Determine which metadata files should be used to filter xml files. 
    paths_to_consider = ["./metadata_20250110/full_data_train_valid_test.csv",
                        "./metadata_20250110/meta_data_NSS_all_variables.csv",
                        "./metadata_20250110/meta_data_whiteCAT_all_variables.csv"]

    valid_patients = valid_patients(paths_to_consider)
    
    process_all_paths(directory=directory, valid_patients=valid_patients)
