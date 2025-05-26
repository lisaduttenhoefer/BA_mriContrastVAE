import os
os.environ["SCIPY_ARRAY_API"] = "1"
import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting, image
from nilearn.plotting import plot_stat_map, plot_roi
import xml.etree.ElementTree as ET
from matplotlib.colors import LinearSegmentedColormap
import warnings

# Define save path
save_path = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/analysis/saliency_map_results"
os.makedirs(save_path, exist_ok=True)

# Load ROI names
roi_csv = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/ROI_names/Aggregated_neuromorphometrics.csv"
rois = pd.read_csv(roi_csv, header=None).iloc[2:, 0].tolist()

# Load second CSV containing Cliff's Delta
data_csv = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/analysis/TESTING/deviation_results_norm_results_HC_0.7_all_20250521_0641_20250521_084349/regional_effect_sizes_vs_HC.csv"
df = pd.read_csv(data_csv)
df["Cleaned_ROI"] = df["ROI_Name"].str.replace("neuromorphometrics_", "").str.split("_").str[0]
#df["Cleaned_ROI"] = df["ROI_Name"].str.replace("_Vgm", "").str.split("_").str[0]

# Filter for ROIs that exist in our list
df_filtered = df[df["Cleaned_ROI"].isin(rois)]

# Find Cliff's Delta column
cliff_delta_col = None
for col in ['Cliffs_Delta', 'CliffsDelta', 'Cliff_Delta', 'cliffs_delta']:
    if col in df_filtered.columns:
        cliff_delta_col = col
        break

if cliff_delta_col is None:
    print("Warning: Could not find Cliff's Delta column. Available columns:")
    print(df_filtered.columns.tolist())
    df_extreme = df_filtered
else:
    # Filter for extreme values: |Cliff's Delta| > 0.1
    df_extreme = df_filtered[
        (df_filtered[cliff_delta_col] > 0.1) | 
        (df_filtered[cliff_delta_col] < -0.1)
    ].copy()
    print(f"Found {len(df_extreme)} ROIs with extreme effect sizes")

# Load the NII file (Atlas)
nii_file = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/atlases_niis/neuromorphometrics.nii"
nii_img = nib.load(nii_file)

# Load XML labels for better ROI mapping
xml_file = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/atlases_niis/atlases_labels/1103_3_glm_LabelMap.xml"
roi_label_mapping = {}

try:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for label_elem in root.findall("Label"): # Corrected to 'Label'
        name_elem = label_elem.find("Name")
        number_elem = label_elem.find("Number")

        if name_elem is not None and number_elem is not None:
            name = name_elem.text
            label = int(number_elem.text)
            roi_label_mapping[name] = label
    print(f"Loaded {len(roi_label_mapping)} ROI labels from XML")
    # --- ADD THIS DEBUG PRINT ---
    print("\nSample from roi_label_mapping:")
    for i, (name, label) in enumerate(roi_label_mapping.items()):
        if i < 10: # Print first 10 to check
            print(f"  '{name}': {label}")
        else:
            break
    if len(roi_label_mapping) > 10:
        print("  ...")
    # ---------------------------
except Exception as e:
    print(f"Could not load XML labels: {e}")



def create_enhanced_stat_map(nii_img, df_extreme, cliff_delta_col, roi_label_mapping):
    """Create a more robust statistical map from ROI effect sizes"""
    atlas_data = nii_img.get_fdata()
    stat_data = np.zeros_like(atlas_data)
    
    if cliff_delta_col is None or len(df_extreme) == 0:
        return nii_img
    
    # Create mapping from cleaned ROI names to effect sizes
    roi_to_effect = dict(zip(df_extreme["Cleaned_ROI"], df_extreme[cliff_delta_col]))
    
    # Get unique labels in the atlas
    unique_labels = np.unique(atlas_data)
    mapped_count = 0
    
    print(f"Processing {len(unique_labels)} unique atlas labels...")
    
    for label in unique_labels:
        if label == 0:  # Skip background
            continue
            
        # Try multiple matching strategies
        label_int = int(label)
        matched_roi = None
        matched_effect = 0
        
        # Strategy 1: Direct label lookup in XML mapping
        for roi_name, xml_label in roi_label_mapping.items():
            if xml_label == label_int:
                # Clean the XML ROI name to match our data
                cleaned_xml_roi = roi_name.replace("neuromorphometrics_", "").split("_")[0]
                if cleaned_xml_roi in roi_to_effect:
                    matched_roi = cleaned_xml_roi
                    matched_effect = roi_to_effect[cleaned_xml_roi]
                    break
        
        # Strategy 2: Try to match by ROI name patterns
        if matched_roi is None:
            for roi_name in roi_to_effect.keys():
                # Try various matching patterns
                if (str(label_int) in roi_name or 
                    roi_name in str(label_int) or
                    any(part in roi_name.lower() for part in str(label_int).split()) or
                    any(str(label_int) in part for part in roi_name.split('_'))):
                    matched_roi = roi_name
                    matched_effect = roi_to_effect[roi_name]
                    break
        
        # Apply the effect size to all voxels with this label
        if matched_roi is not None:
            stat_data[atlas_data == label] = matched_effect
            mapped_count += 1
            print(f"  Label {label_int} -> {matched_roi} (effect: {matched_effect:.3f})")
        else:
            print(f"  Label {label_int} NOT matched")
            print(f"Successfully mapped {mapped_count} labels to effect sizes")
    
    # Create new NIfTI image
    stat_img = nib.Nifti1Image(stat_data, nii_img.affine, nii_img.header)
    
    # Check if we have any non-zero values
    if np.any(stat_data != 0):
        print(f"Statistical map created with {np.sum(stat_data != 0)} non-zero voxels")
        print(f"Effect size range: {np.min(stat_data[stat_data != 0]):.3f} to {np.max(stat_data):.3f}")
    else:
        print("Warning: Statistical map is empty - no ROIs were successfully mapped")
        # Return original atlas for visualization
        return nii_img
    
    return stat_img

# Create enhanced statistical map
stat_img = create_enhanced_stat_map(nii_img, df_extreme, cliff_delta_col, roi_label_mapping)

def create_brain_slice_grid(img, output_path, title="Brain Visualization", cmap='hot', threshold=None):
    """Create a comprehensive brain slice visualization"""
    
    # Determine if this is a statistical map or atlas
    img_data = img.get_fdata()
    is_stat_map = cliff_delta_col is not None and np.any(img_data != 0) and len(df_extreme) > 0
    
    # Set up the figure
    fig = plt.figure(figsize=(20, 16), facecolor='black')
    fig.suptitle(title, color='white', fontsize=18, y=0.95)
    
    # Determine coordinates for slicing
    if is_stat_map:
        # For statistical maps, focus on areas with activation
        non_zero_indices = np.where(img_data != 0)
        if len(non_zero_indices[0]) > 0:
            # Convert voxel coordinates to world coordinates
            affine = img.affine
            voxel_coords = np.column_stack(non_zero_indices)
            world_coords = nib.affines.apply_affine(affine, voxel_coords)
            
            # Get ranges for each dimension
            x_range = [world_coords[:, 0].min(), world_coords[:, 0].max()]
            y_range = [world_coords[:, 1].min(), world_coords[:, 1].max()]
            z_range = [world_coords[:, 2].min(), world_coords[:, 2].max()]
        else:
            # Fallback to default ranges
            x_range, y_range, z_range = [-60, 60], [-80, 60], [-40, 80]
    else:
        # For atlas, use standard brain ranges
        x_range, y_range, z_range = [-60, 60], [-80, 60], [-40, 80]
    
    # Generate slice coordinates
    x_coords = np.linspace(x_range[0], x_range[1], 8)
    y_coords = np.linspace(y_range[0], y_range[1], 8)
    z_coords = np.linspace(z_range[0], z_range[1], 8)
    
    # Set visualization parameters
    if is_stat_map:
        plot_func = plot_stat_map
        vmin, vmax = np.min(img_data[img_data != 0]), np.max(img_data)
        plot_kwargs = {
            'cmap': cmap,
            'threshold': threshold or max(0.01, abs(vmin) * 0.1),
            'vmax': vmax,
            'symmetric_cbar': False,
            'black_bg': True,
            'annotate': False,
            'colorbar': False
        }
    else:
        plot_func = plot_roi
        plot_kwargs = {
            'cmap': 'Paired',
            'black_bg': True,
            'annotate': False,
            'colorbar': False
        }
    
    # Create subplot grid: 6 rows x 8 columns
    subplot_idx = 1
    
    # Row 1-2: Axial slices (z-direction)
    for i in range(16):
        ax = plt.subplot(6, 8, subplot_idx)
        if i < len(z_coords):
            try:
                plot_func(img, display_mode='z', cut_coords=[z_coords[i]], 
                         axes=ax, **plot_kwargs)
                ax.set_title(f'z={z_coords[i]:.0f}', color='white', fontsize=8)
            except Exception as e:
                ax.text(0.5, 0.5, f'z={z_coords[i]:.0f}\n(empty)', 
                       ha='center', va='center', color='white', transform=ax.transAxes)
                ax.set_facecolor('black')
        subplot_idx += 1
    
    # Row 3-4: Sagittal slices (x-direction)
    for i in range(16):
        ax = plt.subplot(6, 8, subplot_idx)
        if i < len(x_coords):
            try:
                plot_func(img, display_mode='x', cut_coords=[x_coords[i]], 
                         axes=ax, **plot_kwargs)
                ax.set_title(f'x={x_coords[i]:.0f}', color='white', fontsize=8)
            except Exception as e:
                ax.text(0.5, 0.5, f'x={x_coords[i]:.0f}\n(empty)', 
                       ha='center', va='center', color='white', transform=ax.transAxes)
                ax.set_facecolor('black')
        subplot_idx += 1
    
    # Row 5-6: Coronal slices (y-direction)
    for i in range(16):
        ax = plt.subplot(6, 8, subplot_idx)
        if i < len(y_coords):
            try:
                plot_func(img, display_mode='y', cut_coords=[y_coords[i]], 
                         axes=ax, **plot_kwargs)
                ax.set_title(f'y={y_coords[i]:.0f}', color='white', fontsize=8)
            except Exception as e:
                ax.text(0.5, 0.5, f'y={y_coords[i]:.0f}\n(empty)', 
                       ha='center', va='center', color='white', transform=ax.transAxes)
                ax.set_facecolor('black')
        subplot_idx += 1
    
    # Add view labels
    fig.text(0.02, 0.78, 'AXIAL', color='white', fontsize=14, fontweight='bold', rotation=90)
    fig.text(0.02, 0.52, 'SAGITTAL', color='white', fontsize=14, fontweight='bold', rotation=90)
    fig.text(0.02, 0.26, 'CORONAL', color='white', fontsize=14, fontweight='bold', rotation=90)
    
    # Add colorbar for statistical maps
    if is_stat_map and vmax > vmin:  # Only add colorbar for stat maps
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Effect Size (Cliff's Delta)", color='white', fontsize=12)
        cbar.ax.tick_params(colors='white')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.90, top=0.92, bottom=0.05)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"Brain slice grid saved at: {output_path}")

# Create visualizations
print("Creating brain slice visualizations...")

# Main statistical map visualization
if cliff_delta_col and len(df_extreme) > 0:
    stat_output = os.path.join(save_path, "brain_statistical_map_slices.png")
    create_brain_slice_grid(stat_img, stat_output, 
                           f"Brain Regions with Extreme Effect Sizes (n={len(df_extreme)})",
                           cmap='RdYlBu_r', threshold=0.05)

# Atlas visualization for reference
atlas_output = os.path.join(save_path, "brain_atlas_reference_slices.png")
create_brain_slice_grid(nii_img, atlas_output, "Neuromorphometrics Atlas Reference")

# Create separate visualizations for positive and negative effects
if cliff_delta_col and len(df_extreme) > 0:
    # Positive effects only
    df_positive = df_extreme[df_extreme[cliff_delta_col] > 0.1]
    if len(df_positive) > 0:
        stat_img_pos = create_enhanced_stat_map(nii_img, df_positive, cliff_delta_col, roi_label_mapping)
        pos_output = os.path.join(save_path, "brain_positive_effects_slices.png")
        create_brain_slice_grid(stat_img_pos, pos_output, 
                               f"Positive Effect Sizes (n={len(df_positive)})",
                               cmap='Reds', threshold=0.01)
    
    # Negative effects only
    df_negative = df_extreme[df_extreme[cliff_delta_col] < -0.1]
    if len(df_negative) > 0:
        stat_img_neg = create_enhanced_stat_map(nii_img, df_negative, cliff_delta_col, roi_label_mapping)
        neg_output = os.path.join(save_path, "brain_negative_effects_slices.png")
        create_brain_slice_grid(stat_img_neg, neg_output, 
                               f"Negative Effect Sizes (n={len(df_negative)})",
                               cmap='Blues', threshold=0.01)

# Print summary and save ROI information
if cliff_delta_col and len(df_extreme) > 0:
    print(f"\n=== SUMMARY ===")
    print(f"Total ROIs with extreme effect sizes: {len(df_extreme)}")
    print(f"Positive effects (>0.1): {len(df_extreme[df_extreme[cliff_delta_col] > 0.1])}")
    print(f"Negative effects (<-0.1): {len(df_extreme[df_extreme[cliff_delta_col] < -0.1])}")
    print(f"Effect size range: {df_extreme[cliff_delta_col].min():.3f} to {df_extreme[cliff_delta_col].max():.3f}")
    
    # Save detailed results
    results_df = df_extreme.sort_values(cliff_delta_col, ascending=False)[
        ['Cleaned_ROI', cliff_delta_col, 'ROI_Name']
    ].round(3)
    results_file = os.path.join(save_path, "extreme_effects_detailed.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Detailed results saved: {results_file}")
    
    # Show top results
    print(f"\nTop 10 ROIs by effect size:")
    print(results_df.head(10).to_string(index=False))
    
    print(f"\nBottom 10 ROIs by effect size:")
    print(results_df.tail(10).to_string(index=False))

print(f"\nAll visualizations saved in: {save_path}")
print("Files created:")
for file in os.listdir(save_path):
    if file.endswith('.png'):
        print(f"  - {file}")