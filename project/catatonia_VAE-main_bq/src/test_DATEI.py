def analyze_regional_deviations(results_df, save_dir, clinical_data_path, volume_type, atlas_name, roi_names, norm_diagnosis,
                               add_catatonia_subgroups=False, metadata_path=None, subgroup_columns=None, 
                               high_low_thresholds=None, catatonia_diagnosis='CTT'):
    """
    Analyze regional deviations using ROI names with optional Catatonia subgroups
    
    Parameters:
    -----------
    add_catatonia_subgroups : bool
        Whether to add Catatonia subgroups to the analysis
    metadata_path : str
        Path to metadata CSV file containing additional patient information
    subgroup_columns : list
        List of column names to create subgroups for (e.g., ['GAF_Score', 'PANSS_Total'])
    high_low_thresholds : dict
        Dictionary mapping column names to threshold values for high/low split
        e.g., {'GAF_Score': 50, 'PANSS_Total': 75}
    catatonia_diagnosis : str
        The diagnosis code for Catatonia patients (default: 'CTT')
    """
    
    # Original analysis code remains the same until heatmap creation
    region_cols = [col for col in results_df.columns if col.startswith("region_")]
    
    if len(roi_names) != len(region_cols):
        print(f"Warning: Number of ROI names ({len(roi_names)}) does not match number of region columns ({len(region_cols)}). Using generic names.")
        roi_names = [f"Region_{i+1}" for i in range(len(region_cols))]
    
    roi_mapping = dict(zip(region_cols, roi_names))
    named_results_df = results_df.copy()
    named_results_df.rename(columns=roi_mapping, inplace=True)
    diagnoses = results_df["Diagnosis"].unique()

    norm_data = results_df[results_df["Diagnosis"] == norm_diagnosis]
    
    if len(norm_data) == 0:
        print(f"Warning: No data found for normative diagnosis '{norm_diagnosis}'. Cannot calculate comparisons.")
        return pd.DataFrame()  
    
    # Function to calculate Cliff's Delta
    def calculate_cliffs_delta(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
    
        dominance = np.zeros((len(x), len(y)))
        for i, x_i in enumerate(x):
            dominance[i] = np.sign(x_i - y)
        
        delta = np.mean(dominance)
        return delta

    effect_sizes = []
    
    # Function to create subgroups for Catatonia patients
    def create_catatonia_subgroups(results_df, metadata_df, subgroup_columns, high_low_thresholds, catatonia_diagnosis):
        """Create subgroups of Catatonia patients based on metadata"""
        subgroups = {}
        
        # Get Catatonia patients
        ctt_patients = results_df[results_df["Diagnosis"] == catatonia_diagnosis].copy()
        
        if len(ctt_patients) == 0:
            print(f"No {catatonia_diagnosis} patients found for subgroup analysis")
            return subgroups
        
        # Merge with metadata
        if 'Subject_ID' in ctt_patients.columns and 'Subject_ID' in metadata_df.columns:
            ctt_with_metadata = ctt_patients.merge(metadata_df, on='Subject_ID', how='left')
        elif 'ID' in ctt_patients.columns and 'Subject_ID' in metadata_df.columns:
            ctt_with_metadata = ctt_patients.merge(metadata_df, left_on='ID', right_on='Subject_ID', how='left')
        else:
            print("Warning: Could not merge metadata. Check ID column names.")
            return subgroups
        
        # Create subgroups for each specified column
        for col in subgroup_columns:
            if col not in ctt_with_metadata.columns:
                print(f"Warning: Column '{col}' not found in metadata")
                continue
            
            # Remove rows with missing values for this column
            valid_data = ctt_with_metadata.dropna(subset=[col])
            
            if len(valid_data) == 0:
                print(f"Warning: No valid data for column '{col}'")
                continue
            
            # Determine threshold
            if col in high_low_thresholds:
                threshold = high_low_thresholds[col]
            else:
                # Use median as default threshold
                threshold = valid_data[col].median()
                print(f"Using median threshold for {col}: {threshold}")
            
            # Create high and low subgroups
            high_group = valid_data[valid_data[col] >= threshold]
            low_group = valid_data[valid_data[col] < threshold]
            
            if len(high_group) > 0:
                subgroups[f"{catatonia_diagnosis}-high_{col}"] = high_group
                print(f"Created {catatonia_diagnosis}-high_{col} subgroup: n={len(high_group)}")
            
            if len(low_group) > 0:
                subgroups[f"{catatonia_diagnosis}-low_{col}"] = low_group
                print(f"Created {catatonia_diagnosis}-low_{col} subgroup: n={len(low_group)}")
        
        return subgroups
    
    # Create Catatonia subgroups if requested
    catatonia_subgroups = {}
    if add_catatonia_subgroups and metadata_path and subgroup_columns:
        try:
            metadata_df = pd.read_csv(metadata_path)
            catatonia_subgroups = create_catatonia_subgroups(
                results_df, metadata_df, subgroup_columns, 
                high_low_thresholds or {}, catatonia_diagnosis
            )
        except Exception as e:
            print(f"Error loading metadata or creating subgroups: {e}")
    
    # Calculate effect sizes for original diagnoses
    for diagnosis in diagnoses:
        if diagnosis == norm_diagnosis:
            continue  
        
        dx_data = results_df[results_df["Diagnosis"] == diagnosis]
        
        if len(dx_data) == 0:
            print(f"No data found for diagnosis: {diagnosis}")
            continue
        
        print(f"Analyzing diagnosis: {diagnosis} (n={len(dx_data)}) vs {norm_diagnosis} (n={len(norm_data)})")
        
        # Analyze each region
        for i, region_col in enumerate(region_cols):
            roi_name = roi_names[i] if i < len(roi_names) else f"Region_{i+1}"
            
            dx_region_values = dx_data[region_col].values
            norm_region_values = norm_data[region_col].values
            
            if len(dx_region_values) == 0 or len(norm_region_values) == 0:
                continue
                
            # Calculate statistics
            dx_mean = np.mean(dx_region_values)
            dx_std = np.std(dx_region_values)
            norm_mean = np.mean(norm_region_values)
            norm_std = np.std(norm_region_values)
            
            mean_diff = dx_mean - norm_mean
            cliff_delta = calculate_cliffs_delta(dx_region_values, norm_region_values)
            
            # Calculate Cohen's d
            pooled_std = np.sqrt(((len(dx_region_values) - 1) * dx_std**2 + 
                                  (len(norm_region_values) - 1) * norm_std**2) / 
                                 (len(dx_region_values) + len(norm_region_values) - 2))
            
            cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0
            
            effect_sizes.append({
                "Diagnosis": diagnosis,
                "Vs_Norm_Diagnosis": norm_diagnosis,
                "Region_Column": region_col,
                "ROI_Name": roi_name,
                "Diagnosis_Mean": dx_mean,
                "Diagnosis_Std": dx_std,
                "Norm_Mean": norm_mean,
                "Norm_Std": norm_std,
                "Mean_Difference": mean_diff,
                "Cliffs_Delta": cliff_delta,
                "Cohens_d": cohens_d
            })
    
    # Calculate effect sizes for Catatonia subgroups
    for subgroup_name, subgroup_data in catatonia_subgroups.items():
        print(f"Analyzing subgroup: {subgroup_name} (n={len(subgroup_data)}) vs {norm_diagnosis} (n={len(norm_data)})")
        
        for i, region_col in enumerate(region_cols):
            roi_name = roi_names[i] if i < len(roi_names) else f"Region_{i+1}"
            
            subgroup_region_values = subgroup_data[region_col].values
            norm_region_values = norm_data[region_col].values
            
            if len(subgroup_region_values) == 0 or len(norm_region_values) == 0:
                continue
            
            # Calculate statistics
            subgroup_mean = np.mean(subgroup_region_values)
            subgroup_std = np.std(subgroup_region_values)
            norm_mean = np.mean(norm_region_values)
            norm_std = np.std(norm_region_values)
            
            mean_diff = subgroup_mean - norm_mean
            cliff_delta = calculate_cliffs_delta(subgroup_region_values, norm_region_values)
            
            # Calculate Cohen's d
            pooled_std = np.sqrt(((len(subgroup_region_values) - 1) * subgroup_std**2 + 
                                  (len(norm_region_values) - 1) * norm_std**2) / 
                                 (len(subgroup_region_values) + len(norm_region_values) - 2))
            
            cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0
            
            effect_sizes.append({
                "Diagnosis": subgroup_name,
                "Vs_Norm_Diagnosis": norm_diagnosis,
                "Region_Column": region_col,
                "ROI_Name": roi_name,
                "Diagnosis_Mean": subgroup_mean,
                "Diagnosis_Std": subgroup_std,
                "Norm_Mean": norm_mean,
                "Norm_Std": norm_std,
                "Mean_Difference": mean_diff,
                "Cliffs_Delta": cliff_delta,
                "Cohens_d": cohens_d
            })
    
    effect_sizes_df = pd.DataFrame(effect_sizes)
    
    if effect_sizes_df.empty:
        print("No effect sizes calculated. Returning empty DataFrame.")
        return effect_sizes_df
    
    effect_sizes_df["Abs_Cliffs_Delta"] = effect_sizes_df["Cliffs_Delta"].abs()
    effect_sizes_df["Abs_Cohens_d"] = effect_sizes_df["Cohens_d"].abs()
    os.makedirs(f"{save_dir}/figures", exist_ok=True)
    
    # Create visualization of top affected regions for each diagnosis (existing code)
    all_diagnoses = effect_sizes_df["Diagnosis"].unique()
    
    for diagnosis in all_diagnoses:
        dx_effect_sizes = effect_sizes_df[effect_sizes_df["Diagnosis"] == diagnosis].copy()
        
        if dx_effect_sizes.empty:
            continue
            
        # Sort by absolute effect size (Cliff's Delta)
        dx_effect_sizes_sorted = dx_effect_sizes.sort_values("Abs_Cliffs_Delta", ascending=False)
        top_regions = dx_effect_sizes_sorted.head(20)
        
        # Create bar plot for Cliff's Delta
        plt.figure(figsize=(14, 10))
        bars = plt.barh(range(len(top_regions)), top_regions["Cliffs_Delta"])
        
        for i, bar in enumerate(bars):
            if top_regions.iloc[i]["Cliffs_Delta"] < 0:
                bar.set_color("blue")
            else:
                bar.set_color("red")
        
        plt.yticks(range(len(top_regions)), top_regions["ROI_Name"])
        plt.axvline(x=0, color="black", linestyle="--", alpha=0.7)
        plt.title(f"Top 20 Regions with Largest Effect Sizes - {diagnosis} vs {norm_diagnosis}")
        plt.xlabel("Cliff's Delta")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/figures/top_regions_cliffs_delta_{diagnosis}_vs_{norm_diagnosis}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create bar plot for Cohen's d
        plt.figure(figsize=(14, 10))
        bars = plt.barh(range(len(top_regions)), top_regions["Cohens_d"])
        
        for i, bar in enumerate(bars):
            if top_regions.iloc[i]["Cohens_d"] < 0:
                bar.set_color("blue")
            else:
                bar.set_color("red")
                
        plt.yticks(range(len(top_regions)), top_regions["ROI_Name"])
        plt.axvline(x=0, color="black", linestyle="--", alpha=0.7)
        plt.title(f"Top 20 Regions with Largest Effect Sizes - {diagnosis} vs {norm_diagnosis}")
        plt.xlabel("Cohen's d")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/figures/top_regions_cohens_d_{diagnosis}_vs_{norm_diagnosis}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Distribution plot
    plt.figure(figsize=(10, 6))
    for diagnosis in all_diagnoses:
        dx_effect_sizes = effect_sizes_df[effect_sizes_df["Diagnosis"] == diagnosis]
        if not dx_effect_sizes.empty:
            sns.kdeplot(dx_effect_sizes["Cliffs_Delta"], label=diagnosis)
    
    plt.axvline(x=0, color="black", linestyle="--", alpha=0.7)
    plt.title(f"Distribution of Regional Effect Sizes vs {norm_diagnosis}")
    plt.xlabel("Cliff's Delta")
    plt.legend()
    plt.tight_layout()  
    plt.savefig(f"{save_dir}/figures/effect_size_distributions_vs_{norm_diagnosis}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Enhanced heatmap with subgroups
    region_avg_effects = effect_sizes_df.groupby("ROI_Name")["Abs_Cliffs_Delta"].mean().reset_index()
    top_regions_overall = region_avg_effects.sort_values("Abs_Cliffs_Delta", ascending=False).head(30)["ROI_Name"].values
    
    # Create matrix of effect sizes for these regions including subgroups
    heatmap_data = []
    for region in top_regions_overall:
        row = {"ROI_Name": region}
        for diagnosis in all_diagnoses:
            region_data = effect_sizes_df[(effect_sizes_df["ROI_Name"] == region) & 
                                         (effect_sizes_df["Diagnosis"] == diagnosis)]
            if not region_data.empty:
                row[diagnosis] = region_data.iloc[0]["Cliffs_Delta"]
            else:
                row[diagnosis] = np.nan
        
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df.set_index("ROI_Name", inplace=True)
    
    if len(heatmap_df.columns) > 0:
        # Adjust figure size based on number of columns
        fig_width = max(12, len(heatmap_df.columns) * 1.5)
        plt.figure(figsize=(fig_width, 14))
        
        sns.heatmap(heatmap_df, cmap="RdBu_r", center=0, annot=True, fmt=".2f", 
                   cbar_kws={"label": "Cliff's Delta"})
        plt.title(f"Top 30 Regions Effect Sizes vs {norm_diagnosis} (Including Subgroups)")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/figures/region_effect_heatmap_with_subgroups_vs_{norm_diagnosis}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save heatmap data
        heatmap_df.to_csv(f"{save_dir}/top_regions_heatmap_data_with_subgroups_vs_{norm_diagnosis}.csv")
    
    print(f"\nRegional analysis completed. Results saved to {save_dir}")
    print(f"Total effect sizes calculated: {len(effect_sizes_df)}")
    print(f"Average absolute Cliff's Delta: {effect_sizes_df['Abs_Cliffs_Delta'].mean():.3f}")
    print(f"Max absolute Cliff's Delta: {effect_sizes_df['Abs_Cliffs_Delta'].max():.3f}")
    
    if catatonia_subgroups:
        print(f"\nCatatonia subgroups created: {list(catatonia_subgroups.keys())}")
    
    return effect_sizes_df


# Example usage:
"""
effect_sizes_df = analyze_regional_deviations(
    results_df=your_results_df,
    save_dir=your_save_dir,
    clinical_data_path=your_clinical_data_path,
    volume_type=your_volume_type,
    atlas_name=your_atlas_name,
    roi_names=your_roi_names,
    norm_diagnosis='HC',
    add_catatonia_subgroups=True,
    metadata_path='/workspace/project/catatonia_VAE-main_bq/metadata_20250110/full_data_with_codiagnosis_and_scores.csv',
    subgroup_columns=['GAF_Score', 'PANSS_Total', 'NCRS_Total'],
    high_low_thresholds={'GAF_Score': 50, 'PANSS_Total': 75, 'NCRS_Total': 25},
    catatonia_diagnosis='CTT'
)
"""

def analyze_regional_deviations(results_df, save_dir, clinical_data_path, volume_type, atlas_name, roi_names, norm_diagnosis,
                               add_catatonia_subgroups=False, metadata_path=None, subgroup_columns=None, 
                               high_low_thresholds=None, catatonia_diagnosis='CTT'):
    """
    Analyze regional deviations using ROI names with optional Catatonia subgroups
    
    Parameters:
    -----------
    add_catatonia_subgroups : bool
        Whether to add Catatonia subgroups to the analysis
    metadata_path : str
        Path to metadata CSV file containing additional patient information
    subgroup_columns : list
        List of column names to create subgroups for (e.g., ['GAF_Score', 'PANSS_Total'])
    high_low_thresholds : dict
        Dictionary mapping column names to threshold values for high/low split
        e.g., {'GAF_Score': 50, 'PANSS_Total': 75}
    catatonia_diagnosis : str
        The diagnosis code for Catatonia patients (default: 'CTT')
    """
    
    # Original analysis code remains the same until heatmap creation
    region_cols = [col for col in results_df.columns if col.startswith("region_")]
    
    if len(roi_names) != len(region_cols):
        print(f"Warning: Number of ROI names ({len(roi_names)}) does not match number of region columns ({len(region_cols)}). Using generic names.")
        roi_names = [f"Region_{i+1}" for i in range(len(region_cols))]
    
    roi_mapping = dict(zip(region_cols, roi_names))
    named_results_df = results_df.copy()
    named_results_df.rename(columns=roi_mapping, inplace=True)
    diagnoses = results_df["Diagnosis"].unique()

    norm_data = results_df[results_df["Diagnosis"] == norm_diagnosis]
    
    if len(norm_data) == 0:
        print(f"Warning: No data found for normative diagnosis '{norm_diagnosis}'. Cannot calculate comparisons.")
        return pd.DataFrame()  
    
    # Function to calculate Cliff's Delta
    def calculate_cliffs_delta(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
    
        dominance = np.zeros((len(x), len(y)))
        for i, x_i in enumerate(x):
            dominance[i] = np.sign(x_i - y)
        
        delta = np.mean(dominance)
        return delta

    effect_sizes = []
    
    # Function to create subgroups for Catatonia patients
    def create_catatonia_subgroups(results_df, metadata_df, subgroup_columns, high_low_thresholds, catatonia_diagnosis):
        """Create subgroups of Catatonia patients based on metadata"""
        subgroups = {}
        
        # Get ALL Catatonia patients (CTT, CTT-SCHZ, CTT-MDD, etc.)
        if catatonia_diagnosis == 'CTT':
            # Look for all diagnoses that start with 'CTT'
            ctt_patients = results_df[results_df["Diagnosis"].str.startswith("CTT")].copy()
            print(f"Found Catatonia diagnoses: {ctt_patients['Diagnosis'].unique()}")
        else:
            # Look for specific diagnosis
            ctt_patients = results_df[results_df["Diagnosis"] == catatonia_diagnosis].copy()
        
        if len(ctt_patients) == 0:
            print(f"No Catatonia patients found for subgroup analysis")
            return subgroups
        
        # Merge with metadata
        if 'Subject_ID' in ctt_patients.columns and 'Subject_ID' in metadata_df.columns:
            ctt_with_metadata = ctt_patients.merge(metadata_df, on='Subject_ID', how='left')
        elif 'ID' in ctt_patients.columns and 'Subject_ID' in metadata_df.columns:
            ctt_with_metadata = ctt_patients.merge(metadata_df, left_on='ID', right_on='Subject_ID', how='left')
        else:
            print("Warning: Could not merge metadata. Check ID column names.")
            return subgroups
        
        # Create subgroups for each specified column
        for col in subgroup_columns:
            if col not in ctt_with_metadata.columns:
                print(f"Warning: Column '{col}' not found in metadata")
                continue
            
            # Remove rows with missing values for this column
            valid_data = ctt_with_metadata.dropna(subset=[col])
            
            if len(valid_data) == 0:
                print(f"Warning: No valid data for column '{col}'")
                continue
            
            # Determine threshold
            if col in high_low_thresholds:
                threshold = high_low_thresholds[col]
            else:
                # Use median as default threshold
                threshold = valid_data[col].median()
                print(f"Using median threshold for {col}: {threshold}")
            
            # Create high and low subgroups
            high_group = valid_data[valid_data[col] >= threshold]
            low_group = valid_data[valid_data[col] < threshold]
            
            if len(high_group) > 0:
                subgroups[f"CTT-high_{col}"] = high_group
                print(f"Created CTT-high_{col} subgroup: n={len(high_group)} (from {high_group['Diagnosis'].value_counts().to_dict()})")
            
            if len(low_group) > 0:
                subgroups[f"CTT-low_{col}"] = low_group
                print(f"Created CTT-low_{col} subgroup: n={len(low_group)} (from {low_group['Diagnosis'].value_counts().to_dict()})")
        
        return subgroups
    
    # Create Catatonia subgroups if requested
    catatonia_subgroups = {}
    if add_catatonia_subgroups and metadata_path and subgroup_columns:
        try:
            metadata_df = pd.read_csv(metadata_path)
            catatonia_subgroups = create_catatonia_subgroups(
                results_df, metadata_df, subgroup_columns, 
                high_low_thresholds or {}, catatonia_diagnosis
            )
        except Exception as e:
            print(f"Error loading metadata or creating subgroups: {e}")
    
    # Calculate effect sizes for original diagnoses
    for diagnosis in diagnoses:
        if diagnosis == norm_diagnosis:
            continue  
        
        dx_data = results_df[results_df["Diagnosis"] == diagnosis]
        
        if len(dx_data) == 0:
            print(f"No data found for diagnosis: {diagnosis}")
            continue
        
        print(f"Analyzing diagnosis: {diagnosis} (n={len(dx_data)}) vs {norm_diagnosis} (n={len(norm_data)})")
        
        # Analyze each region
        for i, region_col in enumerate(region_cols):
            roi_name = roi_names[i] if i < len(roi_names) else f"Region_{i+1}"
            
            dx_region_values = dx_data[region_col].values
            norm_region_values = norm_data[region_col].values
            
            if len(dx_region_values) == 0 or len(norm_region_values) == 0:
                continue
                
            # Calculate statistics
            dx_mean = np.mean(dx_region_values)
            dx_std = np.std(dx_region_values)
            norm_mean = np.mean(norm_region_values)
            norm_std = np.std(norm_region_values)
            
            mean_diff = dx_mean - norm_mean
            cliff_delta = calculate_cliffs_delta(dx_region_values, norm_region_values)
            
            # Calculate Cohen's d
            pooled_std = np.sqrt(((len(dx_region_values) - 1) * dx_std**2 + 
                                  (len(norm_region_values) - 1) * norm_std**2) / 
                                 (len(dx_region_values) + len(norm_region_values) - 2))
            
            cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0
            
            effect_sizes.append({
                "Diagnosis": diagnosis,
                "Vs_Norm_Diagnosis": norm_diagnosis,
                "Region_Column": region_col,
                "ROI_Name": roi_name,
                "Diagnosis_Mean": dx_mean,
                "Diagnosis_Std": dx_std,
                "Norm_Mean": norm_mean,
                "Norm_Std": norm_std,
                "Mean_Difference": mean_diff,
                "Cliffs_Delta": cliff_delta,
                "Cohens_d": cohens_d
            })
    
    # Calculate effect sizes for Catatonia subgroups
    for subgroup_name, subgroup_data in catatonia_subgroups.items():
        print(f"Analyzing subgroup: {subgroup_name} (n={len(subgroup_data)}) vs {norm_diagnosis} (n={len(norm_data)})")
        
        for i, region_col in enumerate(region_cols):
            roi_name = roi_names[i] if i < len(roi_names) else f"Region_{i+1}"
            
            subgroup_region_values = subgroup_data[region_col].values
            norm_region_values = norm_data[region_col].values
            
            if len(subgroup_region_values) == 0 or len(norm_region_values) == 0:
                continue
            
            # Calculate statistics
            subgroup_mean = np.mean(subgroup_region_values)
            subgroup_std = np.std(subgroup_region_values)
            norm_mean = np.mean(norm_region_values)
            norm_std = np.std(norm_region_values)
            
            mean_diff = subgroup_mean - norm_mean
            cliff_delta = calculate_cliffs_delta(subgroup_region_values, norm_region_values)
            
            # Calculate Cohen's d
            pooled_std = np.sqrt(((len(subgroup_region_values) - 1) * subgroup_std**2 + 
                                  (len(norm_region_values) - 1) * norm_std**2) / 
                                 (len(subgroup_region_values) + len(norm_region_values) - 2))
            
            cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0
            
            effect_sizes.append({
                "Diagnosis": subgroup_name,
                "Vs_Norm_Diagnosis": norm_diagnosis,
                "Region_Column": region_col,
                "ROI_Name": roi_name,
                "Diagnosis_Mean": subgroup_mean,
                "Diagnosis_Std": subgroup_std,
                "Norm_Mean": norm_mean,
                "Norm_Std": norm_std,
                "Mean_Difference": mean_diff,
                "Cliffs_Delta": cliff_delta,
                "Cohens_d": cohens_d
            })
    
    effect_sizes_df = pd.DataFrame(effect_sizes)
    
    if effect_sizes_df.empty:
        print("No effect sizes calculated. Returning empty DataFrame.")
        return effect_sizes_df
    
    effect_sizes_df["Abs_Cliffs_Delta"] = effect_sizes_df["Cliffs_Delta"].abs()
    effect_sizes_df["Abs_Cohens_d"] = effect_sizes_df["Cohens_d"].abs()
    os.makedirs(f"{save_dir}/figures", exist_ok=True)
    
    # Create visualization of top affected regions for each diagnosis (existing code)
    all_diagnoses = effect_sizes_df["Diagnosis"].unique()
    
    for diagnosis in all_diagnoses:
        dx_effect_sizes = effect_sizes_df[effect_sizes_df["Diagnosis"] == diagnosis].copy()
        
        if dx_effect_sizes.empty:
            continue
            
        # Sort by absolute effect size (Cliff's Delta)
        dx_effect_sizes_sorted = dx_effect_sizes.sort_values("Abs_Cliffs_Delta", ascending=False)
        top_regions = dx_effect_sizes_sorted.head(20)
        
        # Create bar plot for Cliff's Delta
        plt.figure(figsize=(14, 10))
        bars = plt.barh(range(len(top_regions)), top_regions["Cliffs_Delta"])
        
        for i, bar in enumerate(bars):
            if top_regions.iloc[i]["Cliffs_Delta"] < 0:
                bar.set_color("blue")
            else:
                bar.set_color("red")
        
        plt.yticks(range(len(top_regions)), top_regions["ROI_Name"])
        plt.axvline(x=0, color="black", linestyle="--", alpha=0.7)
        plt.title(f"Top 20 Regions with Largest Effect Sizes - {diagnosis} vs {norm_diagnosis}")
        plt.xlabel("Cliff's Delta")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/figures/top_regions_cliffs_delta_{diagnosis}_vs_{norm_diagnosis}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create bar plot for Cohen's d
        plt.figure(figsize=(14, 10))
        bars = plt.barh(range(len(top_regions)), top_regions["Cohens_d"])
        
        for i, bar in enumerate(bars):
            if top_regions.iloc[i]["Cohens_d"] < 0:
                bar.set_color("blue")
            else:
                bar.set_color("red")
                
        plt.yticks(range(len(top_regions)), top_regions["ROI_Name"])
        plt.axvline(x=0, color="black", linestyle="--", alpha=0.7)
        plt.title(f"Top 20 Regions with Largest Effect Sizes - {diagnosis} vs {norm_diagnosis}")
        plt.xlabel("Cohen's d")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/figures/top_regions_cohens_d_{diagnosis}_vs_{norm_diagnosis}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Distribution plot
    plt.figure(figsize=(10, 6))
    for diagnosis in all_diagnoses:
        dx_effect_sizes = effect_sizes_df[effect_sizes_df["Diagnosis"] == diagnosis]
        if not dx_effect_sizes.empty:
            sns.kdeplot(dx_effect_sizes["Cliffs_Delta"], label=diagnosis)
    
    plt.axvline(x=0, color="black", linestyle="--", alpha=0.7)
    plt.title(f"Distribution of Regional Effect Sizes vs {norm_diagnosis}")
    plt.xlabel("Cliff's Delta")
    plt.legend()
    plt.tight_layout()  
    plt.savefig(f"{save_dir}/figures/effect_size_distributions_vs_{norm_diagnosis}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Enhanced heatmap with subgroups
    region_avg_effects = effect_sizes_df.groupby("ROI_Name")["Abs_Cliffs_Delta"].mean().reset_index()
    top_regions_overall = region_avg_effects.sort_values("Abs_Cliffs_Delta", ascending=False).head(30)["ROI_Name"].values
    
    # Create matrix of effect sizes for these regions including subgroups
    heatmap_data = []
    for region in top_regions_overall:
        row = {"ROI_Name": region}
        for diagnosis in all_diagnoses:
            region_data = effect_sizes_df[(effect_sizes_df["ROI_Name"] == region) & 
                                         (effect_sizes_df["Diagnosis"] == diagnosis)]
            if not region_data.empty:
                row[diagnosis] = region_data.iloc[0]["Cliffs_Delta"]
            else:
                row[diagnosis] = np.nan
        
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df.set_index("ROI_Name", inplace=True)
    
    if len(heatmap_df.columns) > 0:
        # Adjust figure size based on number of columns
        fig_width = max(12, len(heatmap_df.columns) * 1.5)
        plt.figure(figsize=(fig_width, 14))
        
        sns.heatmap(heatmap_df, cmap="RdBu_r", center=0, annot=True, fmt=".2f", 
                   cbar_kws={"label": "Cliff's Delta"})
        plt.title(f"Top 30 Regions Effect Sizes vs {norm_diagnosis} (Including Subgroups)")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/figures/region_effect_heatmap_with_subgroups_vs_{norm_diagnosis}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save heatmap data
        heatmap_df.to_csv(f"{save_dir}/top_regions_heatmap_data_with_subgroups_vs_{norm_diagnosis}.csv")
    
    print(f"\nRegional analysis completed. Results saved to {save_dir}")
    print(f"Total effect sizes calculated: {len(effect_sizes_df)}")
    print(f"Average absolute Cliff's Delta: {effect_sizes_df['Abs_Cliffs_Delta'].mean():.3f}")
    print(f"Max absolute Cliff's Delta: {effect_sizes_df['Abs_Cliffs_Delta'].max():.3f}")
    
    if catatonia_subgroups:
        print(f"\nCatatonia subgroups created: {list(catatonia_subgroups.keys())}")
    
    return effect_sizes_df


# Example usage:
"""
effect_sizes_df = analyze_regional_deviations(
    results_df=your_results_df,
    save_dir=your_save_dir,
    clinical_data_path=your_clinical_data_path,
    volume_type=your_volume_type,
    atlas_name=your_atlas_name,
    roi_names=your_roi_names,
    norm_diagnosis='HC',
    add_catatonia_subgroups=True,
    metadata_path='/workspace/project/catatonia_VAE-main_bq/metadata_20250110/full_data_with_codiagnosis_and_scores.csv',
    subgroup_columns=['GAF_Score', 'PANSS_Total', 'NCRS_Total'],
    high_low_thresholds={'GAF_Score': 50, 'PANSS_Total': 75, 'NCRS_Total': 25},
    catatonia_diagnosis='CTT'
)
"""