
import argparse
import os
import h5py
import sys
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import umap

def calculate_deviations(normative_models, data_tensor, annotations_df, device="cuda"):
    """
    Calculate deviation scores using bootstrap models, ensuring perfect alignment
    between data tensor and annotations.
    
    Args:
        normative_models: List of trained normative VAE models
        data_tensor: Tensor of input data to evaluate
        annotations_df: DataFrame with subject metadata
        device: Computing device
        
    Returns:
        DataFrame with deviation scores for each subject
    """
    # Verify data alignment
    total_models = len(normative_models)
    total_subjects = data_tensor.shape[0]
    
    # Check for size mismatch and report
    if total_subjects != len(annotations_df):
        print(f"WARNING: Size mismatch detected: {total_subjects} samples in data tensor vs {len(annotations_df)} rows in annotations")
        print("Creating properly aligned dataset by extracting common subjects...")

        # Get filenames in annotations_df
        filenames = annotations_df["Filename"].tolist()
        
        # Create a new annotations_df with only rows that have matching data
        valid_indices = list(range(min(total_subjects, len(annotations_df))))
        aligned_annotations = annotations_df.iloc[valid_indices].reset_index(drop=True)
        
        # Use these for processing
        annotations_df = aligned_annotations
        print(f"Aligned datasets - working with {len(annotations_df)} subjects")
    
    # Prepare arrays to store results
    all_recon_errors = np.zeros((total_subjects, total_models))
    all_kl_divs = np.zeros((total_subjects, total_models))
    all_z_scores = np.zeros((total_subjects, data_tensor.shape[1], total_models))
    
    # Process each model
    for i, model in enumerate(normative_models):
        model.eval()
        model.to(device)
        with torch.no_grad():
            batch_data = data_tensor.to(device)
            recon, mu, log_var = model(batch_data)
            
            #--------------------------------------------CALCULATE RECONSTRUCTION ERROR ------------------------------------------------------------
            # Mean squared error between original brain measurements and their reconstruction
            # -> how well the normative model can reproduce he brain pattern
            # -> Higher values indicate brain patterns deviating from normative expectations
            recon_error = torch.mean((batch_data - recon) ** 2, dim=1).cpu().numpy()
            all_recon_errors[:, i] = recon_error
            
            #------------------------------------------------CALCULATE KL DIVERGENCE ---------------------------------------------------------------
            # -> divergence between the encoded distribution and N(0,1)
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).cpu().numpy()
            all_kl_divs[:, i] = kl_div
            
            #---------------------------------------------CALCULATE REGION-WISE-Z-SCORES ---------------------------------------------------------------
            z_scores = ((batch_data - recon) ** 2).cpu().numpy()
            all_z_scores[:, :, i] = z_scores
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    # Average across bootstrap models
    mean_recon_error = np.mean(all_recon_errors, axis=1)
    std_recon_error = np.std(all_recon_errors, axis=1)
    mean_kl_div = np.mean(all_kl_divs, axis=1)
    std_kl_div = np.std(all_kl_divs, axis=1)
    
    # Calculate region-wise mean z-scores
    mean_region_z_scores = np.mean(all_z_scores, axis=2)
    
    # Create result DataFrame with the properly aligned annotations
    results_df = annotations_df[["Filename", "Diagnosis", "Age", "Sex", "Dataset"]].copy()
    
    # Now these should be the same length
    results_df["reconstruction_error"] = mean_recon_error
    results_df["reconstruction_error_std"] = std_recon_error
    results_df["kl_divergence"] = mean_kl_div
    results_df["kl_divergence_std"] = std_kl_div
    
    # # Add region-wise z-scores
    # for i in range(mean_region_z_scores.shape[1]):
    #     results_df[f"region_{i}_z_score"] = mean_region_z_scores[:, i]
    # Create a DataFrame with the new columns
    new_columns = pd.DataFrame(
        mean_region_z_scores, 
        columns=[f"region_{i}_z_score" for i in range(mean_region_z_scores.shape[1])]
    )

    # Efficiently concatenate instead of inserting columns one by one
    results_df = pd.concat([results_df, new_columns], axis=1)

    #-------------------------------------------- CALCULATE COMBINED DEVIATION SCORE ---------------------------------------------------------------
    # Normalize both metrics to 0-1 range for easier interpretation
    min_recon = results_df["reconstruction_error"].min()
    max_recon = results_df["reconstruction_error"].max()
    norm_recon = (results_df["reconstruction_error"] - min_recon) / (max_recon - min_recon)
    
    min_kl = results_df["kl_divergence"].min()
    max_kl = results_df["kl_divergence"].max()
    norm_kl = (results_df["kl_divergence"] - min_kl) / (max_kl - min_kl)
    
    # Combined deviation score (equal weighting of both metrics)
    results_df["deviation_score"] = (norm_recon + norm_kl) / 2
    
    return results_df

def plot_deviation_distributions(results_df, save_dir):
    """Plot distributions of deviation metrics by diagnosis group."""
    print(results_df["Diagnosis"].unique())  # Ensure expected categories exist

    os.makedirs(f"{save_dir}/figures/distributions", exist_ok=True)
    
    # Create color palette
    palette = sns.light_palette("blue", n_colors=4, reverse=True)
    diagnosis_order = ["HC", "SCHZ", "CTT", "MDD"]
    diagnosis_palette = dict(zip(diagnosis_order, palette))

    
    # Plot reconstruction error distributions
    plt.figure(figsize=(12, 8))
    sns.kdeplot(data=results_df, x="reconstruction_error", hue="Diagnosis", palette=diagnosis_palette, common_norm=False)
    plt.title("Reconstruction Error Distribution by Diagnosis", fontsize=16)
    plt.xlabel("Mean Reconstruction Error", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(title="Diagnosis", fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/distributions/recon_error_dist.png", dpi=300)
    plt.close()
    
    # Plot KL divergence distributions
    plt.figure(figsize=(12, 8))
    sns.kdeplot(data=results_df, x="kl_divergence", hue="Diagnosis", palette=diagnosis_palette, common_norm=False)
    plt.title("KL Divergence Distribution by Diagnosis", fontsize=16)
    plt.xlabel("Mean KL Divergence", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(title="Diagnosis", fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/distributions/kl_div_dist.png", dpi=300)
    plt.close()
    
    # Plot combined deviation score distributions
    plt.figure(figsize=(12, 8))
    sns.kdeplot(data=results_df, x="deviation_score", hue="Diagnosis", palette=diagnosis_palette, common_norm=False)
    plt.title("Combined Deviation Score Distribution by Diagnosis", fontsize=16)
    plt.xlabel("Deviation Score", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(title="Diagnosis", fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/distributions/deviation_score_dist.png", dpi=300)
    plt.close()
    
    # Plot violin plots for all metrics
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    sns.violinplot(data=results_df, x="Diagnosis", y="reconstruction_error", palette=diagnosis_palette)
    plt.title("Reconstruction Error by Diagnosis", fontsize=14)
    plt.xlabel("")
    
    plt.subplot(3, 1, 2)
    sns.violinplot(data=results_df, x="Diagnosis", y="kl_divergence", hue="Diagnosis", palette=diagnosis_palette, legend = False)
    plt.title("KL Divergence by Diagnosis", fontsize=14)
    plt.xlabel("")
    
    plt.subplot(3, 1, 3)
    sns.violinplot(data=results_df, x="Diagnosis", y="deviation_score", palette=diagnosis_palette)
    plt.title("Combined Deviation Score by Diagnosis", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/distributions/metrics_violin_plots.png", dpi=300)
    plt.close()

    selected_diagnoses = ["SCHZ", "CTT", "MDD","HC"]

    # Berechne Mittelwert, Standardabweichung und n für alle drei Metriken
    metrics = ["reconstruction_error", "kl_divergence", "deviation_score"]
    summary_dict = {}

    for metric in metrics:
        summary_df = (
            results_df[results_df["Diagnosis"].isin(selected_diagnoses)]
            .groupby("Diagnosis")[metric]
            .agg(['mean', 'std', 'count'])
            .reset_index()
        )

        # Berechne 95%-Konfidenzintervall
        summary_df["ci95"] = 1.96 * summary_df["std"] / np.sqrt(summary_df["count"])

        # Sortiere in gewünschter Reihenfolge (für Darstellung von unten nach oben)
        diagnosis_order = selected_diagnoses[::-1]
        summary_df["Diagnosis"] = pd.Categorical(summary_df["Diagnosis"], categories=diagnosis_order, ordered=True)
        summary_df = summary_df.sort_values("Diagnosis")

        # Speichere die Zusammenfassung für späteren Zugriff
        summary_dict[metric] = summary_df

        # Erstelle den Errorbar-Plot
        plt.figure(figsize=(6, 4))
        plt.errorbar(summary_df["mean"], summary_df["Diagnosis"], xerr=summary_df["ci95"],
                    fmt='s', color='black', capsize=5, markersize=5)
        plt.title(f"{metric.replace('_', ' ').title()} by Diagnosis", fontsize=14)
        plt.xlabel("Deviation Metric", fontsize=12)
        sns.despine()
        plt.tight_layout()

        # Speichere den Plot
        plt.savefig(f"{save_dir}/figures/distributions/{metric}_errorbar.png", dpi=300)
        plt.close()
    return

def visualize_embeddings(normative_models, data_tensor, annotations_df, device="cuda"):

    """Visualize data in the latent space of the normative model, ensuring alignment."""
    
    # Sicherstellen, dass die Anzahl der Samples übereinstimmt
    total_subjects = data_tensor.shape[0]
    
    if total_subjects != len(annotations_df):
        print(f"WARNING: Size mismatch detected: {total_subjects} samples in data tensor vs {len(annotations_df)} rows in annotations")
        print("Creating properly aligned dataset by extracting common subjects...")
        
        # Erstelle eine neue `annotations_df`, die nur die passenden Zeilen enthält
        valid_indices = list(range(min(total_subjects, len(annotations_df))))
        aligned_annotations = annotations_df.iloc[valid_indices].reset_index(drop=True)
        
        # Aktualisiere `annotations_df`
        annotations_df = aligned_annotations
        print(f"Aligned datasets - working with {len(annotations_df)} subjects")
    
    # Verwende das erste Modell für die Visualisierung
    model = normative_models[0]
    model.eval()
    model.to(device)
    
    all_embeddings = []
    batch_size = 32
    
    # Daten in Batches verarbeiten, um Speicherprobleme zu vermeiden
    data_loader = DataLoader(
        TensorDataset(data_tensor), 
        batch_size=batch_size,
        shuffle=False
    )
    
    with torch.no_grad():
        for batch_data, in data_loader:
            batch_data = batch_data.to(device)
            _, mu, _ = model(batch_data)
            all_embeddings.append(mu.cpu().numpy())
    
    # Kombiniere alle Embeddings
    embeddings = np.vstack(all_embeddings)
    
    # UMAP für Visualisierung
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    umap_embeddings = reducer.fit_transform(embeddings)
    
    # Erstelle DataFrame für das Plotten
    plot_df = annotations_df[["Diagnosis"]].copy()
    plot_df["umap_1"] = umap_embeddings[:, 0]
    plot_df["umap_2"] = umap_embeddings[:, 1]
    
    # Plot
    plt.figure(figsize=(12, 10))
    palette = sns.color_palette("colorblind", n_colors=4)
    diagnosis_order = ["HC", "SCHZ", "CTT", "MDD"]
    diagnosis_colors = dict(zip(diagnosis_order, palette))

    sns.scatterplot(
        data=plot_df,
        x="umap_1",
        y="umap_2",
        hue="Diagnosis",
        palette=diagnosis_colors,
        s=40,         # kleinere Punkte
        alpha=0.6     # etwas transparenter für bessere Lesbarkeit
    )

    plt.title("UMAP Visualization of Latent Space", fontsize=16)
    plt.xlabel("UMAP Dimension 1", fontsize=13)
    plt.ylabel("UMAP Dimension 2", fontsize=13)
    plt.legend(title="Diagnosis", fontsize=11, title_fontsize=12, loc="best", frameon=True)

    return plt.gcf(), plot_df


def extract_roi_names(h5_file_path, volume_type):
    """
    Extract ROI names from HDF5 file.
    
    Args:
        h5_file_path (str): Path to HDF5 file
        volume_type (str): Volume type to extract (Vgm, Vwm, csf)
        
    Returns:
        list: List of ROI names
    """
    import h5py
    
    roi_names = []
    try:
        with h5py.File(h5_file_path, 'r') as f:
            # Most HDF5 files store ROI names as attributes or as dataset
            if volume_type in f:
                # Get ROI names from dataset attributes if they exist
                if 'roi_names' in f[volume_type].attrs:
                    roi_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) 
                               for name in f[volume_type].attrs['roi_names']]
                # Get ROI names from specific dataset if it exists
                elif 'roi_names' in f[volume_type]:
                    roi_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) 
                               for name in f[volume_type]['roi_names'][:]]
                # Try to get indices/keys that correspond to measurements
                elif 'measurements' in f[volume_type]:
                    # Some HDF5 files have indices stored separately
                    if 'indices' in f[volume_type]:
                        roi_names = [str(idx) for idx in f[volume_type]['indices'][:]]
                    else:
                        num_rois = f[volume_type]['measurements'].shape[1]
                        roi_names = [f"ROI_{i+1}" for i in range(num_rois)]
            else:
                # Try to look for ROI names at the root level
                if 'roi_names' in f.attrs:
                    roi_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) 
                               for name in f.attrs['roi_names']]
                elif 'roi_names' in f:
                    roi_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) 
                               for name in f['roi_names'][:]]
                # Try to infer from top-level structure
                else:
                    # Sometimes ROIs are stored as separate datasets
                    roi_candidates = [key for key in f.keys() if key != 'metadata']
                    if roi_candidates:
                        roi_names = roi_candidates
    except Exception as e:
        print(f"Error extracting ROI names from {h5_file_path}: {e}")
    
    # If still no ROI names, create generic ones based on atlas name
    if not roi_names:
        from pathlib import Path
        atlas_name = Path(h5_file_path).stem
        # Try to get the number of measurements from the file
        try:
            with h5py.File(h5_file_path, 'r') as f:
                if volume_type in f and 'measurements' in f[volume_type]:
                    num_rois = f[volume_type]['measurements'].shape[1]
                else:
                    num_rois = 100  # Default assumption
                roi_names = [f"{atlas_name}_ROI_{i+1}" for i in range(num_rois)]
        except:
            roi_names = [f"{atlas_name}_ROI_{i+1}" for i in range(100)]  # Assume max 100 ROIs
    
    return roi_names

def analyze_regional_deviations(results_df, save_dir, clinical_data_path, volume_type, atlas_name, roi_names, norm_diagnosis):
    """
    Analyze regional deviations using ROI names extracted from HDF5 file.
    
    Args:
        results_df (pd.DataFrame): DataFrame with deviation scores
        save_dir (str): Directory to save results
        clinical_data_path (str): Path to HDF5 file with ROI names
        volume_type (str): Volume type to extract (Vgm, Vwm, csf)
        
    Returns:
        pd.DataFrame: DataFrame with regional effect sizes
    """
    
    # Get region columns from results
    region_cols = [col for col in results_df.columns if col.startswith("region_")]
    
    if len(roi_names) != len(region_cols):
        print(f"Warning: Number of ROI names ({len(roi_names)}) does not match number of region columns ({len(region_cols)}). Using generic names.")
        # Use generic names if counts don't match
        roi_names = [f"Region_{i+1}" for i in range(len(region_cols))]
    
    # Create mapping between region columns and ROI names
    roi_mapping = dict(zip(region_cols, roi_names))
    
    # Rename columns in a copy of results_df
    named_results_df = results_df.copy()
    named_results_df.rename(columns=roi_mapping, inplace=True)
    
    # Save the version with ROI names
    named_results_df.to_csv(f"{save_dir}/deviation_scores_with_roi_names.csv", index=False)
    
    # Get diagnoses
    diagnoses = results_df["Diagnosis"].unique()

    # Get normative group data
    norm_data = results_df[results_df["Diagnosis"] == norm_diagnosis]
    
    
    if len(norm_data) == 0:
        print(f"Warning: No data found for normative diagnosis '{norm_diagnosis}'. Cannot calculate comparisons.")
        return pd.DataFrame()  # Return empty DataFrame
    
    # Function to calculate Cliff's Delta
    def calculate_cliffs_delta(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
    
        # For each pair (x_i, y_j):
        #  +1 if x_i > y_j
        #  -1 if x_i < y_j
        #   0 if x_i == y_j
        dominance = np.zeros((len(x), len(y)))
        for i, x_i in enumerate(x):
            dominance[i] = np.sign(x_i - y)
        
        # Calculate Cliff's Delta as the mean of the dominance matrix
        delta = np.mean(dominance)
        
        return delta
    
    # Calculate effect sizes and create visualizations
    effect_sizes = []
    
    for diagnosis in diagnoses:
        if diagnosis == norm_diagnosis:
            continue  # Skip normative group as we use it for comparison
            
        # Filter data for this diagnosis
        dx_data = results_df[results_df["Diagnosis"] == diagnosis]
        
        if len(dx_data) == 0:
            print(f"No data found for diagnosis: {diagnosis}")
            continue
        
        print(f"Analyzing diagnosis: {diagnosis} (n={len(dx_data)}) vs {norm_diagnosis} (n={len(norm_data)})")
        
        # Analyze each region
        for i, region_col in enumerate(region_cols):
            roi_name = roi_names[i] if i < len(roi_names) else f"Region_{i+1}"
            
            # Get region values for both groups
            dx_region_values = dx_data[region_col].values
            norm_region_values = norm_data[region_col].values
            
            if len(dx_region_values) == 0 or len(norm_region_values) == 0:
                continue
                
            # Calculate statistics for each group
            dx_mean = np.mean(dx_region_values)
            dx_std = np.std(dx_region_values)
            norm_mean = np.mean(norm_region_values)
            norm_std = np.std(norm_region_values)
            
            # Calculate difference
            mean_diff = dx_mean - norm_mean
            
            # Calculate Cliff's Delta between this diagnosis and normative group for this region
            cliff_delta = calculate_cliffs_delta(dx_region_values, norm_region_values)
            
            # Calculate Cohen's d effect size
            pooled_std = np.sqrt(((len(dx_region_values) - 1) * dx_std**2 + 
                                  (len(norm_region_values) - 1) * norm_std**2) / 
                                 (len(dx_region_values) + len(norm_region_values) - 2))
            
            if pooled_std == 0:  # Avoid division by zero
                cohens_d = 0
            else:
                cohens_d = mean_diff / pooled_std
            
            # Save results
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
    
    # Create DataFrame with effect sizes
    effect_sizes_df = pd.DataFrame(effect_sizes)
    
    # Save complete effect sizes
    effect_sizes_df.to_csv(f"{save_dir}/regional_effect_sizes_vs_{norm_diagnosis}.csv", index=False)
    
    # Create visualization of top affected regions for each diagnosis
    for diagnosis in diagnoses:
        if diagnosis == norm_diagnosis:
            continue
            
        # Filter data for this diagnosis
        dx_effect_sizes = effect_sizes_df[effect_sizes_df["Diagnosis"] == diagnosis]
        
        if dx_effect_sizes.empty:
            continue
            
        # Sort by absolute effect size (Cliff's Delta)
        dx_effect_sizes["Abs_Cliffs_Delta"] = dx_effect_sizes["Cliffs_Delta"].abs()
        dx_effect_sizes_sorted = dx_effect_sizes.sort_values("Abs_Cliffs_Delta", ascending=False)
        
        # Take top 20 regions
        top_regions = dx_effect_sizes_sorted.head(20)
        
        # Create bar plot for Cliff's Delta
        plt.figure(figsize=(14, 10))
        bars = plt.barh(top_regions["ROI_Name"], top_regions["Cliffs_Delta"])
        
        # Color bars based on direction of effect
        for i, bar in enumerate(bars):
            if top_regions.iloc[i]["Cliffs_Delta"] < 0:
                bar.set_color("blue")
            else:
                bar.set_color("red")
        
        plt.axvline(x=0, color="black", linestyle="--", alpha=0.7)
        plt.title(f"Top 20 Regions with Largest Effect Sizes - {diagnosis} vs {norm_diagnosis}")
        plt.xlabel("Cliff's Delta")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/figures/top_regions_cliffs_delta_{diagnosis}_vs_{norm_diagnosis}.png", dpi=300)
        plt.close()
        
        # Create bar plot for Cohen's d
        plt.figure(figsize=(14, 10))
        bars = plt.barh(top_regions["ROI_Name"], top_regions["Cohens_d"])
        
        # Color bars based on direction of effect
        for i, bar in enumerate(bars):
            if top_regions.iloc[i]["Cohens_d"] < 0:
                bar.set_color("blue")
            else:
                bar.set_color("red")
                
        plt.axvline(x=0, color="black", linestyle="--", alpha=0.7)
        plt.title(f"Top 20 Regions with Largest Effect Sizes - {diagnosis} vs {norm_diagnosis}")
        plt.xlabel("Cohen's d")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/figures/top_regions_cohens_d_{diagnosis}_vs_{norm_diagnosis}.png", dpi=300)
        plt.close()
        
        # Also save the top regions data
        top_regions.to_csv(f"{save_dir}/top20_regions_{diagnosis}_vs_{norm_diagnosis}.csv", index=False)
    
    # Create a summary visualization showing the overall distribution of effects
    plt.figure(figsize=(10, 6))
    for diagnosis in diagnoses:
        if diagnosis == norm_diagnosis:
            continue
        
        dx_effect_sizes = effect_sizes_df[effect_sizes_df["Diagnosis"] == diagnosis]
        if not dx_effect_sizes.empty:
            sns.kdeplot(dx_effect_sizes["Cliffs_Delta"], label=diagnosis)
    
    plt.axvline(x=0, color="black", linestyle="--", alpha=0.7)
    plt.title(f"Distribution of Regional Effect Sizes vs {norm_diagnosis}")
    plt.xlabel("Cliff's Delta")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/effect_size_distributions_vs_{norm_diagnosis}.png", dpi=300)
    plt.close()
    
    # Create heatmap of top regions across diagnoses
    # Get top 30 regions overall by average absolute effect size
    region_avg_effects = effect_sizes_df.groupby("ROI_Name")["Abs_Cliffs_Delta"].mean().reset_index()
    top_regions_overall = region_avg_effects.sort_values("Abs_Cliffs_Delta", ascending=False).head(30)["ROI_Name"].values
    
    # Create matrix of effect sizes for these regions
    heatmap_data = []
    for region in top_regions_overall:
        row = {"ROI_Name": region}
        for diagnosis in diagnoses:
            if diagnosis == norm_diagnosis:
                continue
            
            region_data = effect_sizes_df[(effect_sizes_df["ROI_Name"] == region) & 
                                         (effect_sizes_df["Diagnosis"] == diagnosis)]
            if not region_data.empty:
                row[diagnosis] = region_data.iloc[0]["Cliffs_Delta"]
            else:
                row[diagnosis] = np.nan
        
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df.set_index("ROI_Name", inplace=True)
    
    # Plot heatmap if we have more than one diagnosis to compare
    if len(diagnoses) > 1 and len(heatmap_df.columns) > 0:
        plt.figure(figsize=(12, 14))
        sns.heatmap(heatmap_df, cmap="RdBu_r", center=0, annot=True, fmt=".2f", 
                   cbar_kws={"label": "Cliff's Delta"})
        plt.title(f"Top 30 Regions Effect Sizes vs {norm_diagnosis}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/figures/region_effect_heatmap_vs_{norm_diagnosis}.png", dpi=300)
        plt.close()
        
        # Save heatmap data
        heatmap_df.to_csv(f"{save_dir}/top_regions_heatmap_data_vs_{norm_diagnosis}.csv")
    
    return effect_sizes_df