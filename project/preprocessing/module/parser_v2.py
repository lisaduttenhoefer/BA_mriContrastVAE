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


def dict_to_df(parsed_dict, patient=None, ext="csv", config=None):
    """
    Convert parsed dictionary to CSV files
    No longer differentiates between HC/non-HC
    """
    output_dir = config.EXTRACTED_CSV_DIR if config else "./xml_data"
    
    for atlas_name, df in parsed_dict.items():
        # Add patient ID as column if provided
        if patient is not None:
            df["patient_id"] = patient
        
        # Define file path
        filepath = os.path.join(output_dir, f"Aggregated_{atlas_name}.{ext}")
        
        try:
            # Check if file exists to decide between creating or appending
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                # Read existing data
                existing_df = pd.read_csv(filepath, index_col=0)
                # Append new data
                combined_df = pd.concat([existing_df, df], axis=0)
                # Save combined data
                combined_df.to_csv(filepath)
                
                # Also save transposed version if needed
                if config and hasattr(config, 'EXTRACTED_CSV_T_DIR'):
                    t_output_dir = config.EXTRACTED_CSV_T_DIR
                    os.makedirs(t_output_dir, exist_ok=True)
                    t_filepath = os.path.join(t_output_dir, f"Aggregated_{atlas_name}_t.{ext}")
                    df_t = combined_df.T
                    df_t.to_csv(t_filepath)
            else:
                # Create new file
                df.to_csv(filepath)
                
                # Also save transposed version if needed
                if config and hasattr(config, 'EXTRACTED_CSV_T_DIR'):
                    t_output_dir = config.EXTRACTED_CSV_T_DIR
                    os.makedirs(t_output_dir, exist_ok=True)
                    t_filepath = os.path.join(t_output_dir, f"Aggregated_{atlas_name}_t.{ext}")
                    df_t = df.T
                    df_t.to_csv(t_filepath)
        except Exception as e:
            print(f"Error saving CSV {filepath}: {e}")

def dict_to_hdf5(parsed_dict, patient=None, ext="h5", config=None):
    """
    Convert parsed dictionary to HDF5 files
    No longer differentiates between HC/non-HC
    """
    output_dir = config.EXTRACTED_CSV_DIR if config else "./xml_data"
    
    for atlas_name, df in parsed_dict.items():
        # Add patient ID as column if provided
        if patient is not None:
            df["patient_id"] = patient
        
        # Define file path
        filepath = os.path.join(output_dir, f"Aggregated_{atlas_name}.{ext}")
        
        try:
            # Check if file exists to decide between creating or appending
            if os.path.exists(filepath):
                # Read existing data
                try:
                    existing_df = pd.read_hdf(filepath, key='atlas_data')
                    # Append new data
                    combined_df = pd.concat([existing_df, df], axis=0)
                    # Save combined data
                    combined_df.to_hdf(filepath, key='atlas_data', mode='w', format='table')
                    
                    # Also save transposed version if needed
                    if config and hasattr(config, 'EXTRACTED_CSV_T_DIR'):
                        t_output_dir = config.EXTRACTED_CSV_T_DIR
                        os.makedirs(t_output_dir, exist_ok=True)
                        t_filepath = os.path.join(t_output_dir, f"Aggregated_{atlas_name}.{ext}")
                        df_t = combined_df.T
                        df_t.to_hdf(t_filepath, key='atlas_data_t', mode='w', format='table')
                except Exception as e:
                    print(f"Error appending to HDF5 {filepath}: {e}")
            else:
                # Create new file
                df.to_hdf(filepath, key='atlas_data', mode='w', format='table')
                
                # Also save transposed version if needed
                if config and hasattr(config, 'EXTRACTED_CSV_T_DIR'):
                    t_output_dir = config.EXTRACTED_CSV_T_DIR
                    os.makedirs(t_output_dir, exist_ok=True)
                    t_filepath = os.path.join(t_output_dir, f"Aggregated_{atlas_name}.{ext}")
                    df_t = df.T
                    df_t.to_hdf(t_filepath, key='atlas_data_t', mode='w', format='table')
        except Exception as e:
            print(f"Error saving HDF5 {filepath}: {e}")

def get_all_xml_paths(directory: str, valid_patients: list, metadata_paths: list):
    """
    Get all xml files from directory that match valid_patients
    without filtering by HC/non-HC status
    """
    all_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.xml'):
                full_path = os.path.join(root, file)
                
                # Extract patient ID from filename
                match_no_ext = re.search(r"([^/\\]+)\.[^./\\]*$", full_path)
                if match_no_ext:
                    patient_id = match_no_ext.group(1)
                    new_match = re.search(r"catROI_(.+)", patient_id)
                    if new_match:
                        patient_id = new_match.group(1)
                        
                    # Check if patient is in valid_patients list
                    if patient_id in valid_patients:
                        all_paths.append(full_path)
    
    print(f"Found {len(all_paths)} valid patient XML files in total")
    return all_paths


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
    
    # Clear existing files at the beginning
    for section in section_types:
        filepath = f"./xml_data/Aggregated_{section}.csv"
        if os.path.exists(filepath):
            # Clear the file by opening in write mode
            with open(filepath, "w+") as f:
                f.close()
    
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

def process_all_paths_no_diagnosis_filter(directory: str, valid_patients: list, batch_size: int = 10, 
                                          hdf5: bool = True, config=None):
    """
    Process all XML files without filtering by diagnosis (HC/non-HC)
    
    Parameters:
    -----------
    directory : str
        Directory containing XML files
    valid_patients : list
        List of valid patient IDs
    batch_size : int, optional
        Number of files to process in each batch
    hdf5 : bool, optional
        Whether to save as HDF5 (True) or CSV (False)
    config : Config object, optional
        Configuration object with paths
    """
    # Get all valid XML paths without filtering by diagnosis
    paths = get_all_xml_paths(directory, valid_patients, config.METADATA_PATHS if config else None)
    print(f"Found a total of {len(paths)} valid patient XML files.")
    
    # First, identify all section types that will be processed in next step
    section_types = set()
    for path in paths[:min(100, len(paths))]:  # Sample a subset to identify section types
        try:
            parsed_dict = xml_parser(path)
            section_types.update(parsed_dict.keys())
        except Exception as e:
            print(f"Error parsing {path}: {e}")
    
    print(f"Found {len(section_types)} section types: {section_types}")
    
    # Determine output directory from config
    output_dir = config.EXTRACTED_CSV_DIR if config else "./xml_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear existing files at the beginning
    for section in section_types:
        if hdf5:
            filepath = os.path.join(output_dir, f"Aggregated_{section}.h5")
        else:
            filepath = os.path.join(output_dir, f"Aggregated_{section}.csv")
            
        if os.path.exists(filepath):
            try:
                if hdf5:
                    # Create empty HDF5 file
                    with h5py.File(filepath, "w") as f:
                        pass
                else:
                    # Clear CSV file
                    with open(filepath, "w+") as f:
                        f.close()
                print(f"Cleared existing file: {filepath}")
            except Exception as e:
                print(f"Error clearing file {filepath}: {e}")
    
    # Process files in batches
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(paths)-1)//batch_size + 1} ({len(batch_paths)} files)")
        start = time.perf_counter()
        
        for idx, path in enumerate(batch_paths):
            try:
                parsed_dict = xml_parser(path)
                
                # Extract patient ID
                match_no_ext = re.search(r"([^/\\]+)\.[^./\\]*$", path)
                if match_no_ext:
                    patient_id = match_no_ext.group(1)
                    new_match = re.search(r"catROI_(.+)", patient_id)
                    if new_match:
                        patient_id = new_match.group(1)
                    
                    # Save data without differentiating between HC/non-HC
                    if hdf5:
                        dict_to_hdf5(parsed_dict, patient=patient_id, ext="h5", config=config)
                    else:
                        dict_to_df(parsed_dict, patient=patient_id, ext="csv", config=config)
            except Exception as e:
                print(f"Error processing file {idx+1}/{len(batch_paths)}: {path}\nError: {e}")
        
        stop = time.perf_counter()
        print(f"Elapsed time for batch: {stop-start:.2f} seconds")
    
    print("Finished processing all files!")
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
