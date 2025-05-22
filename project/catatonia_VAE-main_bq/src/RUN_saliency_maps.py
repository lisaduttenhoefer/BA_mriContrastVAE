import numpy as np
import pandas as pd
import nibabel as nib
import os
os.environ["SCIPY_ARRAY_API"] = "1"
from nilearn import plotting, datasets, image
import matplotlib.pyplot as plt
import seaborn as sns

from utils.saliency_maps import (
    load_neuromorphometrics_atlas,
    map_effect_sizes_to_brain,
    create_roi_name_mapping,
    visualize_all_diagnoses_for_atlas,
    compare_atlases_for_diagnosis,
    plot_effect_sizes_on_brain,
    get_available_atlases_and_volumes,
    plot_effect_sizes_on_brain, 
    explore_roi_effects,
)


# Load your data
effect_sizes_df = pd.read_csv("/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/analysis/TESTING/deviation_results_norm_results_HC_0.7_all_20250521_0641_20250521_084349/regional_effect_sizes_vs_HC.csv")
atlas_dir = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/atlases_niis"
save_dir = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/analysis/saliency_map_results"

# Alle Diagnosen visualisieren (außer HC)
diagnoses = [d for d in effect_sizes_df['Diagnosis'].unique() if d != 'HC']

# Visualize one diagnosis with neuromorphometrics atlas
plot_effect_sizes_on_brain(
    effect_sizes_df, 
    atlas_name="neuromorphometrics",  # Just the atlas name
    diagnosis="MDD",
    effect_type='Cliffs_Delta',
    save_dir=save_dir,
    atlas_dir=atlas_dir,  # Directory with your .nii files
    volume_suffix="_Vgm"
)


# See what's available in your data
available = get_available_atlases_and_volumes(effect_sizes_df)
print("Available atlases and volume types:")
for atlas, volumes in available.items():
    print(f"  {atlas}: {volumes}")

# Alle verfügbaren Diagnosen anzeigen
print("Verfügbare Diagnosen:", effect_sizes_df['Diagnosis'].unique())

# All diagnoses for neuromorphometrics
visualize_all_diagnoses_for_atlas(
    effect_sizes_df,
    #diagnosis=diagnoses, 
    atlas_name="neuromorphometrics",
    save_dir=save_dir,
    effect_type='Cliffs_Delta',
    atlas_dir=atlas_dir, 
    #atlas_name_prefix="neuromorphometrics_", 
    #volume_suffix="_Vgm"
)

# Compare MDD across multiple atlases
compare_atlases_for_diagnosis(
    effect_sizes_df,
    diagnosis="MDD",
    atlas_names=["neuromorphometrics", "suit", "lpba40", "thalamus", "thalamic_nuclei", "cobra"],
    save_dir=save_dir,
    atlas_dir=atlas_dir
)

for diagnosis in diagnoses:
    plot_effect_sizes_on_brain(
        effect_sizes_df=effect_sizes_df, 
        atlas_name="neuromorphometrics",
        diagnosis=diagnosis,
        effect_type='Cliffs_Delta',
        save_dir=save_dir, 
        atlas_dir=atlas_dir,
        volume_suffix="_Vgm"
    )