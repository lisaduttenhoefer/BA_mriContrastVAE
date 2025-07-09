import numpy as np 
import pytest
import xml.etree.ElementTree as ET
import pandas as pd
import os
import pathlib
import regex as re


def add_to_gitignore(path: str):
    with open(".gitignore", "a") as g:
        g.write("\n")
        g.write(f"{path}")
    return


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


def dict_to_df(data_dict: dict, patient: str):
    # Converts the dict of atlases into separate pandas DataFrames and saves these each to
    # a csv file (once with rows as features and once with columns as features).
    for k, v in data_dict.items():  # k is the atlas, v is the data in the atlas
        filepath = f"./xml_data/Aggregated_{k}.csv"  
        filepath_t = f"./xml_data_t/Aggregated_{k}_t.csv"

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
    

def get_all_xml_paths(directory: str, valid_patients: list) -> list:
    # Finds all xml paths in the directory for which there is also a marker in the metadata.
    xml_paths = pathlib.Path(directory).rglob("label/*.xml")  # rglob searches in all subdirectories
    xml_paths = list(xml_paths)
    xml_paths = [str(i) for i in xml_paths]  # Convert PosixPath to string to allow iteration

    partial_set = set(valid_patients)
    filtered_paths = [
        xml_path for xml_path in xml_paths
        if any(partial_path in xml_path for partial_path in partial_set)
    ]
    
    return filtered_paths


def process_all_paths(directory: str, valid_patients: list): 
    # Convert a number of xml files to csv files with aggregated results for each brain atlas.
    paths = get_all_xml_paths(directory, valid_patients)
    print(f"Found a total of {len(paths)} valid patient .xml files.")

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
    for idx, path in enumerate(paths):  
        print(f"Processing file {idx+1}/{len(paths)}: {path}")
        parsed_dict = xml_parser(path)

        match_no_ext = re.search(r"([^/\\]+)\.[^./\\]*$", path)  # Extract file stem
        if match_no_ext:
            patient_id = match_no_ext.group(1)
        
        new_match = re.search(r"catROI_(.+)", patient_id)  # Extract file ID
        if new_match:
            patient_id = new_match.group(1)

        dict_to_df(parsed_dict, patient=patient_id)
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
