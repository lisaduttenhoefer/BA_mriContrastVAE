import pandas as pd
import numpy as np
import re

def extract_icd10_category(icd10_code):
    """
    Extract the main category from ICD-10 code.
    F2x.x -> F2 (Schizophrenia spectrum)
    F3x.x -> F3 (Mood disorders/Depression)
    """
    if pd.isna(icd10_code) or not isinstance(icd10_code, str):
        return None
    
    # Extract F followed by digit(s)
    match = re.match(r'F([0-9]+)', icd10_code.upper())
    if match:
        category = int(match.group(1))
        if 20 <= category <= 29:  # F20-F29: Schizophrenia spectrum
            return 'SCHZ'
        elif 30 <= category <= 39:  # F30-F39: Mood disorders
            return 'MDD'
    
    return None

def merge_metadata_with_codiagnosis(original_metadata_path, whitecat_metadata_path, nss_metadata_path):
    """
    Merge original metadata with additional WhiteCAT and NSS metadata to create
    new diagnosis categories including co-diagnosis information and clinical scores.
    
    Parameters:
    - original_metadata_path: Path to original metadata CSV
    - whitecat_metadata_path: Path to WhiteCAT metadata CSV  
    - nss_metadata_path: Path to NSS metadata CSV
    
    Returns:
    - merged_df: DataFrame with new diagnosis categories and clinical scores
    """
    
    # Load original metadata
    print("Loading original metadata...")
    original_df = pd.read_csv(original_metadata_path)
    print(f"Original metadata shape: {original_df.shape}")
    print(f"Original diagnoses: {original_df['Diagnosis'].value_counts()}")
    print(f"Datasets: {original_df['Dataset'].value_counts()}")
    
    # Load additional metadata files
    print("\nLoading WhiteCAT metadata...")
    whitecat_df = pd.read_csv(whitecat_metadata_path)
    print(f"WhiteCAT metadata shape: {whitecat_df.shape}")
    print(f"WhiteCAT columns: {list(whitecat_df.columns)}")
    
    print("Loading NSS metadata...")
    nss_df = pd.read_csv(nss_metadata_path)
    print(f"NSS metadata shape: {nss_df.shape}")
    print(f"NSS columns: {list(nss_df.columns)}")
    
    # Create a copy of original dataframe for merging
    merged_df = original_df.copy()
    merged_df['New_Diagnosis'] = merged_df['Diagnosis'].copy()
    merged_df['Co_Diagnosis'] = None
    merged_df['ICD10_Code'] = None
    
    # Initialize clinical score columns
    clinical_columns = [
        'GAF_Score', 'PANSS_Positive', 'PANSS_Negative', 'PANSS_General', 'PANSS_Total',
        'BPRS_Total', 'NCRS_Motor', 'NCRS_Affective', 'NCRS_Behavioral', 'NCRS_Total',
        'NSS_Motor', 'NSS_Total'
    ]
    
    for col in clinical_columns:
        merged_df[col] = None
    
    # Process WhiteCAT patients
    print("\nProcessing WhiteCAT patients...")
    whitecat_patients = merged_df[merged_df['Dataset'].str.contains('whiteCAT', na=False)]
    print(f"Found {len(whitecat_patients)} WhiteCAT patients")
    
    for idx, row in whitecat_patients.iterrows():
        filename = row['Filename']
        
        # Find matching record in WhiteCAT metadata
        whitecat_match = None
        
        # Strategy 1: Direct filename match
        if 'Filename' in whitecat_df.columns:
            whitecat_match = whitecat_df[whitecat_df['Filename'] == filename]
        
        # Strategy 2: Try matching without file extension
        if (whitecat_match is None or len(whitecat_match) == 0) and 'Filename' in whitecat_df.columns:
            filename_no_ext = re.sub(r'\.[^.]+$', '', filename)
            whitecat_match = whitecat_df[whitecat_df['Filename'].str.contains(filename_no_ext, na=False)]
        
        # Strategy 3: Try other potential ID columns
        if (whitecat_match is None or len(whitecat_match) == 0):
            for col in whitecat_df.columns:
                if 'id' in col.lower() or 'subject' in col.lower():
                    filename_no_ext = re.sub(r'\.[^.]+$', '', filename)
                    whitecat_match = whitecat_df[whitecat_df[col].astype(str).str.contains(filename_no_ext, na=False)]
                    if len(whitecat_match) > 0:
                        break
        
        if whitecat_match is not None and len(whitecat_match) > 0:
            match_row = whitecat_match.iloc[0]
            
            # Get ICD-10 code from ana_icd10 column (corrected column name)
            icd10_code = match_row.get('ana_icd10', None)
            merged_df.at[idx, 'ICD10_Code'] = icd10_code
            
            # Extract co-diagnosis category
            co_diagnosis = extract_icd10_category(icd10_code)
            merged_df.at[idx, 'Co_Diagnosis'] = co_diagnosis
            
            # Extract clinical scores for WhiteCAT
            merged_df.at[idx, 'GAF_Score'] = match_row.get('gaf_score_v1', None)
            merged_df.at[idx, 'PANSS_Positive'] = match_row.get('panss_p_v1', None)
            merged_df.at[idx, 'PANSS_Negative'] = match_row.get('panss_n_v1', None)
            merged_df.at[idx, 'PANSS_General'] = match_row.get('panss_g_v1', None)
            merged_df.at[idx, 'BPRS_Total'] = match_row.get('bprs_total_v1', None)
            merged_df.at[idx, 'NCRS_Motor'] = match_row.get('ncrs_motor_v1', None)
            merged_df.at[idx, 'NCRS_Affective'] = match_row.get('ncrs_affective_v1', None)
            merged_df.at[idx, 'NCRS_Behavioral'] = match_row.get('ncrs_behavior_v1', None)
            merged_df.at[idx, 'NCRS_Total'] = match_row.get('ncrs_total_v1', None)
            merged_df.at[idx, 'NSS_Motor'] = match_row.get('nssc_motor_v1', None)
            merged_df.at[idx, 'NSS_Total'] = match_row.get('nssc_total_v1', None)
            
            # Calculate PANSS Total if individual scores are available
            panss_p = match_row.get('panss_p_v1', None)
            panss_n = match_row.get('panss_n_v1', None)
            panss_g = match_row.get('panss_g_v1', None)
            if all(pd.notna([panss_p, panss_n, panss_g])):
                merged_df.at[idx, 'PANSS_Total'] = panss_p + panss_n + panss_g
            
            # Update diagnosis if CTT and has co-diagnosis
            if row['Diagnosis'] == 'CTT' and co_diagnosis is not None:
                merged_df.at[idx, 'New_Diagnosis'] = f'CTT-{co_diagnosis}'
                print(f"  Updated {filename}: CTT -> CTT-{co_diagnosis} (ICD10: {icd10_code})")
    
    # Process NSS patients
    print("\nProcessing NSS patients...")
    nss_patients = merged_df[merged_df['Dataset'].str.contains('NSS', na=False)]
    print(f"Found {len(nss_patients)} NSS patients")
    
    for idx, row in nss_patients.iterrows():
        filename = row['Filename']
        
        # Find matching record in NSS metadata
        nss_match = None
        
        # Similar matching strategies as for WhiteCAT
        if 'Filename' in nss_df.columns:
            nss_match = nss_df[nss_df['Filename'] == filename]
        
        if (nss_match is None or len(nss_match) == 0) and 'Filename' in nss_df.columns:
            filename_no_ext = re.sub(r'\.[^.]+$', '', filename)
            nss_match = nss_df[nss_df['Filename'].str.contains(filename_no_ext, na=False)]
        
        if (nss_match is None or len(nss_match) == 0):
            for col in nss_df.columns:
                if 'id' in col.lower() or 'subject' in col.lower():
                    filename_no_ext = re.sub(r'\.[^.]+$', '', filename)
                    nss_match = nss_df[nss_df[col].astype(str).str.contains(filename_no_ext, na=False)]
                    if len(nss_match) > 0:
                        break
        
        if nss_match is not None and len(nss_match) > 0:
            match_row = nss_match.iloc[0]
            
            # Get ICD-10 code from Diagnosis_y column
            icd10_code = match_row.get('Diagnosis_y', None)
            merged_df.at[idx, 'ICD10_Code'] = icd10_code
            
            # Extract co-diagnosis category
            co_diagnosis = extract_icd10_category(icd10_code)
            merged_df.at[idx, 'Co_Diagnosis'] = co_diagnosis
            
            # Extract clinical scores for NSS
            merged_df.at[idx, 'GAF_Score'] = match_row.get('GAF_currently', None)
            merged_df.at[idx, 'PANSS_Positive'] = match_row.get('PANSS_p', None)
            merged_df.at[idx, 'PANSS_Negative'] = match_row.get('PANSS_n', None)
            merged_df.at[idx, 'PANSS_General'] = match_row.get('PANSS_g', None)
            merged_df.at[idx, 'BPRS_Total'] = match_row.get('BPRS_total', None)
            merged_df.at[idx, 'NCRS_Motor'] = match_row.get('NCRS_motor', None)
            merged_df.at[idx, 'NCRS_Affective'] = match_row.get('NCRS_affective', None)
            merged_df.at[idx, 'NCRS_Behavioral'] = match_row.get('NCRS_behavior', None)
            merged_df.at[idx, 'NCRS_Total'] = match_row.get('NCRS_total', None)
            merged_df.at[idx, 'NSS_Motor'] = match_row.get('NSS_motor', None)
            merged_df.at[idx, 'NSS_Total'] = match_row.get('NSS_total', None)
            
            # Calculate PANSS Total if individual scores are available
            panss_p = match_row.get('PANSS_p', None)
            panss_n = match_row.get('PANSS_n', None)
            panss_g = match_row.get('PANSS_g', None)
            if all(pd.notna([panss_p, panss_n, panss_g])):
                merged_df.at[idx, 'PANSS_Total'] = panss_p + panss_n + panss_g
            
            # Update diagnosis if CTT and has co-diagnosis
            if row['Diagnosis'] == 'CTT' and co_diagnosis is not None:
                merged_df.at[idx, 'New_Diagnosis'] = f'CTT-{co_diagnosis}'
                print(f"  Updated {filename}: CTT -> CTT-{co_diagnosis} (ICD10: {icd10_code})")
    
    # Print summary of changes
    print("\n" + "="*50)
    print("SUMMARY OF DIAGNOSIS CHANGES")
    print("="*50)
    print("Original diagnosis counts:")
    print(merged_df['Diagnosis'].value_counts())
    print("\nNew diagnosis counts:")
    print(merged_df['New_Diagnosis'].value_counts())
    
    # Show patients with co-diagnosis
    ctt_with_codiag = merged_df[merged_df['New_Diagnosis'].str.contains('CTT-', na=False)]
    print(f"\nPatients with CTT co-diagnosis: {len(ctt_with_codiag)}")
    if len(ctt_with_codiag) > 0:
        print(ctt_with_codiag[['Filename', 'Dataset', 'Diagnosis', 'New_Diagnosis', 'Co_Diagnosis', 'ICD10_Code']].head(10))
    
    # Check for unmatched CTT patients
    unmatched_ctt = merged_df[(merged_df['Diagnosis'] == 'CTT') & 
                             (merged_df['New_Diagnosis'] == 'CTT')]
    print(f"\nCTT patients without co-diagnosis match: {len(unmatched_ctt)}")
    if len(unmatched_ctt) > 0:
        print("These patients remain as 'CTT':")
        print(unmatched_ctt[['Filename', 'Dataset']].head(10))
    
    # Print clinical scores summary
    print("\n" + "="*50)
    print("CLINICAL SCORES SUMMARY")
    print("="*50)
    
    for col in clinical_columns:
        non_null_count = merged_df[col].notna().sum()
        if non_null_count > 0:
            print(f"{col}: {non_null_count} patients have data")
            if merged_df[col].dtype in ['int64', 'float64']:
                print(f"  Mean: {merged_df[col].mean():.2f}, Std: {merged_df[col].std():.2f}")
    
    return merged_df

def update_load_mri_data_for_new_diagnoses(merged_metadata_df, save_path=None):
    """
    Prepare the merged metadata for use in load_mri_data_2D function.
    
    Parameters:
    - merged_metadata_df: DataFrame from merge_metadata_with_codiagnosis
    - save_path: Optional path to save the updated metadata
    
    Returns:
    - updated_df: DataFrame ready for use in load_mri_data_2D
    """
    
    # Create a copy with the new diagnosis column as the main diagnosis
    updated_df = merged_metadata_df.copy()
    updated_df['Diagnosis'] = updated_df['New_Diagnosis']
    
    # Keep relevant columns
    cols_to_keep = ['Filename', 'Dataset', 'Diagnosis', 'Age', 'Sex', 'Usage_original', 'Sex_int']
    if 'Unnamed: 0' in updated_df.columns:
        cols_to_keep.insert(0, 'Unnamed: 0')
    
    # Add the new co-diagnosis information columns
    cols_to_keep.extend(['Co_Diagnosis', 'ICD10_Code'])
    
    # Add clinical score columns
    clinical_columns = [
        'GAF_Score', 'PANSS_Positive', 'PANSS_Negative', 'PANSS_General', 'PANSS_Total',
        'BPRS_Total', 'NCRS_Motor', 'NCRS_Affective', 'NCRS_Behavioral', 'NCRS_Total',
        'NSS_Motor', 'NSS_Total'
    ]
    cols_to_keep.extend(clinical_columns)
    
    updated_df = updated_df[cols_to_keep]
    
    print("Updated metadata ready for load_mri_data_2D:")
    print(f"Shape: {updated_df.shape}")
    print("Diagnosis counts:")
    print(updated_df['Diagnosis'].value_counts())
    
    # Print availability of clinical scores
    print("\nClinical scores availability:")
    for col in clinical_columns:
        non_null_count = updated_df[col].notna().sum()
        print(f"  {col}: {non_null_count}/{len(updated_df)} patients ({non_null_count/len(updated_df)*100:.1f}%)")
    
    if save_path:
        updated_df.to_csv(save_path, index=False)
        print(f"Updated metadata saved to: {save_path}")
    
    return updated_df

# Example usage:
def main_example():
    """
    Example of how to use the functions above.
    """
    
    # Define your file paths
    original_metadata_path = "/workspace/project/catatonia_VAE-main_bq/metadata_20250110/full_data_train_valid_test.csv"
    whitecat_metadata_path = "/workspace/project/catatonia_VAE-main_bq/metadata_20250110/meta_data_whiteCAT_all_variables.csv"
    nss_metadata_path = "/workspace/project/catatonia_VAE-main_bq/metadata_20250110/meta_data_NSS_all_variables.csv"
    
    # Merge metadata with co-diagnosis information and clinical scores
    merged_df = merge_metadata_with_codiagnosis(
        original_metadata_path=original_metadata_path,
        whitecat_metadata_path=whitecat_metadata_path,
        nss_metadata_path=nss_metadata_path
    )
    
    # Prepare for use in load_mri_data_2D
    updated_metadata = update_load_mri_data_for_new_diagnoses(
        merged_metadata_df=merged_df,
        save_path="/workspace/project/catatonia_VAE-main_bq/metadata_20250110/full_data_with_codiagnosis_and_scores.csv"
    )
    
    return updated_metadata

if __name__ == "__main__":
    # Run the example
    updated_metadata = main_example()