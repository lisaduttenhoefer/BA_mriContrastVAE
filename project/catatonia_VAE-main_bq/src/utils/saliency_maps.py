import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import plotting, datasets, image
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_neuromorphometrics_atlas(atlas_path=None):
    """
    Load neuromorphometrics atlas. If no path provided, try to download from nilearn.
    
    Parameters:
    -----------
    atlas_path : str, optional
        Path to neuromorphometrics atlas file
        
    Returns:
    --------
    atlas_img : nibabel image
        Loaded atlas image
    labels_df : pd.DataFrame
        DataFrame with ROI labels and indices
    """
    
    if atlas_path is None:
        try:
            # Try to fetch neuromorphometrics from nilearn
            atlas_data = datasets.fetch_atlas_neuromorphometrics()
            atlas_img = atlas_data['maps']
            # Create labels dataframe
            labels_df = pd.DataFrame({
                'roi_index': range(len(atlas_data['labels'])),
                'roi_name': atlas_data['labels']
            })
        except:
            print("Could not download neuromorphometrics atlas. Please provide atlas_path.")
            return None, None
    else:
        # Load from provided path
        atlas_img = nib.load(atlas_path)
        # You'll need to provide ROI labels separately or extract from filename conventions
        labels_df = None
        
    return atlas_img, labels_df

def map_effect_sizes_to_brain(effect_sizes_df, diagnosis, atlas_dir,
                             effect_type='Cliffs_Delta', roi_name_col='ROI_Name',
                             atlas_name_prefix=None, volume_suffix=None):
    """
    Map effect sizes from your analysis back to brain space using neuromorphometrics atlas
    
    Parameters:
    -----------
    effect_sizes_df : pd.DataFrame
        Your effect sizes DataFrame from analyze_regional_deviations()
    diagnosis : str
        Which diagnosis to map (filters the DataFrame)
    effect_type : str
        Which effect size column to use ('Cliffs_Delta' or 'Cohens_d')
    roi_name_col : str
        Column name containing ROI names
    atlas_name_prefix : str, optional
        Prefix to remove from ROI names (e.g., "neuromorphometrics_")
    volume_suffix : str, optional
        Suffix to remove from ROI names (e.g., "_Vgm")
        
    Returns:
    --------
    effect_map_img : nibabel image
        3D brain image with effect sizes mapped to voxels
    """
    atlas_img = nib.load(f"/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/atlases_niis/neuromorphometrics.nii")

    # Filter data for specific diagnosis
    dx_data = effect_sizes_df[effect_sizes_df['Diagnosis'] == diagnosis].copy()
    
    if dx_data.empty:
        print(f"No data found for diagnosis: {diagnosis}")
        return None
    
    # Load atlas data
    atlas_data = atlas_img.get_fdata()
    
    # Create empty effect map
    effect_map = np.zeros_like(atlas_data)
    
    # Clean ROI names by removing prefix and suffix
    def clean_roi_name(roi_name):
        cleaned = roi_name
        if atlas_name_prefix and cleaned.startswith(atlas_name_prefix):
            cleaned = cleaned[len(atlas_name_prefix):]
        if volume_suffix and cleaned.endswith(volume_suffix):
            cleaned = cleaned[:-len(volume_suffix)]
        return cleaned
    
    # Create mapping dictionary from cleaned ROI names to effect values
    roi_effects = {}
    for _, row in dx_data.iterrows():
        original_name = row[roi_name_col]
        cleaned_name = clean_roi_name(original_name)
        roi_effects[cleaned_name] = row[effect_type]
    
    print(f"Mapping {len(roi_effects)} ROIs for {diagnosis}")
    print(f"Example cleaned ROI names: {list(roi_effects.keys())[:5]}")
    
    # Load neuromorphometrics labels if available
    try:
        neuromorpho_data = datasets.fetch_atlas_neuromorphometrics()
        neuromorpho_labels = neuromorpho_data['labels']
        print(f"Using {len(neuromorpho_labels)} neuromorphometrics labels")
    except:
        neuromorpho_labels = None
        print("Could not load neuromorphometrics labels, using index matching")
    
    # Map effect sizes to brain voxels
    unique_labels = np.unique(atlas_data)
    unique_labels = unique_labels[unique_labels != 0]  # Remove background
    
    mapped_count = 0
    for label_idx in unique_labels:
        # Create mask for this ROI
        roi_mask = (atlas_data == label_idx)
        
        # Try different matching strategies
        effect_value = None
        
        # Strategy 1: Direct label matching if we have neuromorphometrics labels
        if neuromorpho_labels is not None and int(label_idx) <= len(neuromorpho_labels):
            atlas_roi_name = neuromorpho_labels[int(label_idx) - 1]  # Labels are 1-indexed
            
            # Try exact match
            if atlas_roi_name in roi_effects:
                effect_value = roi_effects[atlas_roi_name]
            else:
                # Try partial matching
                for cleaned_roi, eff_val in roi_effects.items():
                    if (cleaned_roi.lower() in atlas_roi_name.lower() or 
                        atlas_roi_name.lower() in cleaned_roi.lower()):
                        effect_value = eff_val
                        break
        
        # Strategy 2: Index-based matching
        if effect_value is None:
            # Try to match by label index
            for cleaned_roi, eff_val in roi_effects.items():
                if str(int(label_idx)) in cleaned_roi:
                    effect_value = eff_val
                    break
        
        # Apply effect value if found
        if effect_value is not None:
            effect_map[roi_mask] = effect_value
            mapped_count += 1
    
    print(f"Successfully mapped {mapped_count}/{len(unique_labels)} ROIs to brain space")
    
    # Create new nibabel image
    effect_map_img = nib.Nifti1Image(effect_map, atlas_img.affine, atlas_img.header)
    
    return effect_map_img

def create_roi_name_mapping(roi_names, atlas_labels=None):
    """
    Create mapping between your ROI names and atlas labels
    
    Parameters:
    -----------
    roi_names : list
        List of ROI names from your analysis
    atlas_labels : list, optional
        List of atlas labels from neuromorphometrics
        
    Returns:
    --------
    mapping : dict
        Dictionary mapping ROI names to atlas indices
    """
    
    # This function helps create explicit mapping between your ROI names 
    # and the neuromorphometrics atlas indices
    # You'll need to customize this based on your specific setup
    
    mapping = {}
    
    if atlas_labels is not None:
        # Try automatic matching based on string similarity
        for i, roi_name in enumerate(roi_names):
            # Simple approach - you may want to use more sophisticated matching
            best_match_idx = None
            best_score = 0
            
            for j, atlas_label in enumerate(atlas_labels):
                # Calculate simple similarity score
                similarity = len(set(roi_name.lower().split()) & set(atlas_label.lower().split()))
                if similarity > best_score:
                    best_score = similarity
                    best_match_idx = j + 1  # Atlas indices usually start from 1
            
            if best_match_idx is not None:
                mapping[roi_name] = best_match_idx
    else:
        # Fallback: assume ROI names contain region numbers
        for roi_name in roi_names:
            # Extract numbers from ROI name
            import re
            numbers = re.findall(r'\d+', roi_name)
            if numbers:
                mapping[roi_name] = int(numbers[0])
    
    return mapping

def plot_effect_sizes_on_brain(effect_sizes_df, atlas_name, diagnosis, 
                              effect_type='Cliffs_Delta', save_dir=None,
                              atlas_dir=None, volume_suffix="_Vgm"):
    """
    Complete pipeline to plot your effect sizes on brain - automatically matches atlas and data
    
    Parameters:
    -----------
    effect_sizes_df : pd.DataFrame
        Your effect sizes DataFrame
    atlas_name : str
        Name of the atlas (e.g., "neuromorphometrics", "aal", "harvard_oxford")
        Will look for atlas_name.nii and ROI names with atlas_name_ prefix
    diagnosis : str
        Diagnosis to visualize
    effect_type : str
        'Cliffs_Delta' or 'Cohens_d'
    save_dir : str, optional
        Directory to save plots
    atlas_dir : str, optional
        Directory containing atlas files (if None, uses current directory)
    volume_suffix : str
        ROI name suffix to remove (e.g., "_Vgm", "_Vwm", "_csf")
    """
    
    # Construct atlas file path
    if atlas_dir is None:
        atlas_dir = "."
    
    # Try different file extensions
    possible_extensions = ['.nii', '.nii.gz']
    atlas_path = None
    import os
    for ext in possible_extensions:
        potential_path = f"{atlas_dir}/{atlas_name}{ext}"
        if os.path.exists(potential_path):
            atlas_path = potential_path
            break
    
    if atlas_path is None:
        print(f"Error: Could not find atlas file for '{atlas_name}' in directory '{atlas_dir}'")
        print(f"Looked for: {[f'{atlas_dir}/{atlas_name}{ext}' for ext in possible_extensions]}")
        return
    
    # Load atlas
    atlas_img = nib.load(atlas_path)
    print(f"Loaded atlas: {atlas_path}")
    print(f"Atlas shape: {atlas_img.shape}")
    
    # Filter effect sizes data to only include ROIs matching this atlas
    atlas_prefix = f"{atlas_name}_"
    atlas_data = effect_sizes_df[
        effect_sizes_df['ROI_Name'].str.startswith(atlas_prefix) & 
        effect_sizes_df['ROI_Name'].str.endswith(volume_suffix)
    ].copy()
    
    if atlas_data.empty:
        print(f"Error: No ROI data found for atlas '{atlas_name}' with suffix '{volume_suffix}'")
        print(f"Available ROI prefixes in data: {set([name.split('_')[0] for name in effect_sizes_df['ROI_Name'] if '_' in name])}")
        print(f"Available volume suffixes: {set([name.split('_')[-1] for name in effect_sizes_df['ROI_Name'] if '_' in name])}")
        return
    
    print(f"Found {len(atlas_data)} ROIs matching atlas '{atlas_name}' with suffix '{volume_suffix}'")
    
    # Map effect sizes to brain
    effect_map_img = map_effect_sizes_to_brain(
        atlas_data, atlas_dir, atlas_img, diagnosis, effect_type,
        atlas_name_prefix=atlas_prefix, volume_suffix=volume_suffix
    )
    
    if effect_map_img is None:
        return
    
    # Determine threshold and color scheme based on effect type
    if effect_type == 'Cliffs_Delta':
        threshold = 0.1  # Small effect size threshold
        vmax = 0.8  # Large effect size
        title_suffix = "Cliff's Delta"
    else:  # Cohen's d
        threshold = 0.2  # Small effect size threshold  
        vmax = 1.5  # Large effect size
        title_suffix = "Cohen's d"
    
    # Create comprehensive brain visualization
    fig = plt.figure(figsize=(20, 16))
    
    # Glass brain view
    ax1 = plt.subplot(3, 3, 1)
    plotting.plot_glass_brain(effect_map_img, threshold=threshold, 
                             colorbar=True, cmap='RdBu_r', 
                             vmax=vmax, axes=ax1,
                             title=f"Glass Brain - {diagnosis}")
    
    # Axial slices
    ax2 = plt.subplot(3, 3, 2)
    plotting.plot_stat_map(effect_map_img, threshold=threshold,
                          cmap='RdBu_r', vmax=vmax, axes=ax2,
                          title="Axial View", display_mode='z',
                          cut_coords=7)
    
    # Sagittal slices
    ax3 = plt.subplot(3, 3, 3)
    plotting.plot_stat_map(effect_map_img, threshold=threshold,
                          cmap='RdBu_r', vmax=vmax, axes=ax3,
                          title="Sagittal View", display_mode='x',
                          cut_coords=7)
    
    # Coronal slices
    ax4 = plt.subplot(3, 3, 4)
    plotting.plot_stat_map(effect_map_img, threshold=threshold,
                          cmap='RdBu_r', vmax=vmax, axes=ax4,
                          title="Coronal View", display_mode='y',
                          cut_coords=7)
    
    # Add bar plot of top regions for context
    dx_data = atlas_data[atlas_data['Diagnosis'] == diagnosis].copy()
    
    if dx_data.empty:
        print(f"Warning: No data found for diagnosis '{diagnosis}' in atlas '{atlas_name}'")
        return
    
    dx_data['Abs_Effect'] = dx_data[effect_type].abs()
    top_regions = dx_data.nlargest(15, 'Abs_Effect')
    
    ax5 = plt.subplot(3, 3, (5, 9))
    bars = ax5.barh(range(len(top_regions)), top_regions[effect_type])
    
    # Color bars by effect direction
    for i, bar in enumerate(bars):
        if top_regions.iloc[i][effect_type] < 0:
            bar.set_color('blue')
        else:
            bar.set_color('red')
    
    ax5.set_yticks(range(len(top_regions)))
    # Clean ROI names for display
    clean_names = []
    for name in top_regions['ROI_Name']:
        clean_name = name
        if atlas_prefix and clean_name.startswith(atlas_prefix):
            clean_name = clean_name[len(atlas_prefix):]
        if volume_suffix and clean_name.endswith(volume_suffix):
            clean_name = clean_name[:-len(volume_suffix)]
        clean_names.append(clean_name)
    
    ax5.set_yticklabels(clean_names, fontsize=8)
    ax5.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax5.set_xlabel(title_suffix)
    ax5.set_title(f'Top 15 Regions - {title_suffix}')
    
    plt.suptitle(f'{diagnosis} - {title_suffix} Effect Sizes on {atlas_name.title()} Atlas', 
                 fontsize=16, y=0.95)
    plt.tight_layout()
    
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/brain_map_{diagnosis}_{effect_type}_{atlas_name}.png", 
                   dpi=300, bbox_inches='tight')
        print(f"Saved brain map to: {save_dir}/brain_map_{diagnosis}_{effect_type}_{atlas_name}.png")
    
    plt.show()
    
    # Also save the brain map as NIfTI for further analysis
    if save_dir:
        nib.save(effect_map_img, f"{save_dir}/{diagnosis}_{effect_type}_{atlas_name}_brain_map.nii.gz")
        print(f"Saved brain map data to: {save_dir}/{diagnosis}_{effect_type}_{atlas_name}_brain_map.nii.gz")

# Example usage function
def visualize_all_diagnoses_for_atlas(effect_sizes_df, atlas_name, save_dir, 
                                      effect_type='Cliffs_Delta', norm_diagnosis='HC',
                                      atlas_dir=None, volume_suffix="_Vgm"):
    """
    Create brain maps for all diagnoses for a specific atlas
    
    Parameters:
    -----------
    effect_sizes_df : pd.DataFrame
        Your complete effect sizes DataFrame
    atlas_name : str
        Name of the atlas (e.g., "neuromorphometrics", "aal")
    save_dir : str
        Directory to save results
    effect_type : str
        'Cliffs_Delta' or 'Cohens_d'
    norm_diagnosis : str
        Normative diagnosis to exclude from visualization
    atlas_dir : str, optional
        Directory containing atlas files
    volume_suffix : str
        Volume type suffix (e.g., "_Vgm", "_Vwm", "_csf")
    """
    
    # Filter data for this specific atlas
    atlas_prefix = f"{atlas_name}_"
    atlas_data = effect_sizes_df[
        effect_sizes_df['ROI_Name'].str.startswith(atlas_prefix) & 
        effect_sizes_df['ROI_Name'].str.endswith(volume_suffix)
    ].copy()
    
    if atlas_data.empty:
        print(f"No data found for atlas '{atlas_name}' with suffix '{volume_suffix}'")
        return
    
    diagnoses = atlas_data['Diagnosis'].unique()
    diagnoses = [d for d in diagnoses if d != norm_diagnosis]  # Exclude normative group
    
    print(f"Creating brain maps for {len(diagnoses)} diagnoses using {atlas_name} atlas:")
    print(f"Diagnoses: {diagnoses}")
    
    for diagnosis in diagnoses:
        print(f"\nProcessing {diagnosis}...")
        plot_effect_sizes_on_brain(
            effect_sizes_df, atlas_name, diagnosis, 
            effect_type=effect_type, save_dir=save_dir,
            atlas_dir=atlas_dir, volume_suffix=volume_suffix
        )

def compare_atlases_for_diagnosis(effect_sizes_df, diagnosis, atlas_names, 
                                 save_dir, effect_type='Cliffs_Delta',
                                 atlas_dir=None, volume_suffix="_Vgm"):
    """
    Compare effect sizes across different atlases for the same diagnosis
    
    Parameters:
    -----------
    effect_sizes_df : pd.DataFrame
        Your complete effect sizes DataFrame
    diagnosis : str
        Diagnosis to compare across atlases
    atlas_names : list
        List of atlas names to compare
    save_dir : str
        Directory to save results
    effect_type : str
        'Cliffs_Delta' or 'Cohens_d'
    atlas_dir : str, optional
        Directory containing atlas files
    volume_suffix : str
        Volume type suffix
    """
    
    print(f"Comparing atlases for diagnosis: {diagnosis}")
    
    for atlas_name in atlas_names:
        print(f"\nProcessing atlas: {atlas_name}")
        
        # Check if data exists for this atlas
        atlas_prefix = f"{atlas_name}_"
        atlas_data = effect_sizes_df[
            effect_sizes_df['ROI_Name'].str.startswith(atlas_prefix) & 
            effect_sizes_df['ROI_Name'].str.endswith(volume_suffix) &
            (effect_sizes_df['Diagnosis'] == diagnosis)
        ]
        
        if atlas_data.empty:
            print(f"No data found for {diagnosis} in {atlas_name} atlas")
            continue
            
        plot_effect_sizes_on_brain(
            effect_sizes_df, atlas_name, diagnosis, 
            effect_type=effect_type, save_dir=save_dir,
            atlas_dir=atlas_dir, volume_suffix=volume_suffix
        )

def get_available_atlases_and_volumes(effect_sizes_df):
    """
    Analyze your data to see which atlases and volume types are available
    
    Parameters:
    -----------
    effect_sizes_df : pd.DataFrame
        Your effect sizes DataFrame
        
    Returns:
    --------
    dict : Dictionary with atlas names as keys and volume types as values
    """
    
    roi_names = effect_sizes_df['ROI_Name'].unique()
    
    # Extract atlas names and volume types
    atlas_volume_combinations = {}
    
    for roi_name in roi_names:
        parts = roi_name.split('_')
        if len(parts) >= 3:  # Expecting format: atlas_region_volume
            atlas_name = parts[0]
            volume_type = '_' + parts[-1]  # Last part is volume type
            
            if atlas_name not in atlas_volume_combinations:
                atlas_volume_combinations[atlas_name] = set()
            atlas_volume_combinations[atlas_name].add(volume_type)
    
    # Convert sets to lists for easier handling
    for atlas in atlas_volume_combinations:
        atlas_volume_combinations[atlas] = list(atlas_volume_combinations[atlas])
    
    return atlas_volume_combinations

# Interactive exploration function
def explore_roi_effects(effect_sizes_df, roi_name):
    """
    Explore effect sizes for a specific ROI across all diagnoses
    """
    roi_data = effect_sizes_df[effect_sizes_df['ROI_Name'] == roi_name]
    
    if roi_data.empty:
        print(f"No data found for ROI: {roi_name}")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Cliff's Delta plot
    plt.subplot(1, 2, 1)
    plt.bar(roi_data['Diagnosis'], roi_data['Cliffs_Delta'])
    plt.title(f"Cliff's Delta - {roi_name}")
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    
    # Cohen's d plot
    plt.subplot(1, 2, 2)
    plt.bar(roi_data['Diagnosis'], roi_data['Cohens_d'])
    plt.title(f"Cohen's d - {roi_name}")
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    return roi_data