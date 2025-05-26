
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
from sklearn.preprocessing import StandardScaler

def calculate_deviations(normative_models, data_tensor, norm_diagnosis, annotations_df, device="cuda"):
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
    #########

    #=== IMPROVED DEVIATION SCORE CALCULATION ===
    
    # Method 1: Z-score normalization (preserves relative differences better)
    scaler_recon = StandardScaler()
    scaler_kl = StandardScaler()
    
    z_norm_recon = scaler_recon.fit_transform(mean_recon_error.reshape(-1, 1)).flatten()
    z_norm_kl = scaler_kl.fit_transform(mean_kl_div.reshape(-1, 1)).flatten()
    
    # Combined deviation score (Z-score based)
    results_df["deviation_score_zscore"] = (z_norm_recon + z_norm_kl) / 2
    
    # Method 2: Percentile-based scoring
    recon_percentiles = stats.rankdata(mean_recon_error) / len(mean_recon_error)
    kl_percentiles = stats.rankdata(mean_kl_div) / len(mean_kl_div)
    results_df["deviation_score_percentile"] = (recon_percentiles + kl_percentiles) / 2
    ######
    # Method 3: Original min-max (FIXED)
    min_recon = results_df["reconstruction_error"].min()
    max_recon = results_df["reconstruction_error"].max()
    norm_recon = (results_df["reconstruction_error"] - min_recon) / (max_recon - min_recon)
    
    min_kl = results_df["kl_divergence"].min()
    max_kl = results_df["kl_divergence"].max()
    norm_kl = (results_df["kl_divergence"] - min_kl) / (max_kl - min_kl)
    
    # Combined deviation score (equal weighting of both metrics)
    results_df["deviation_score"] = (norm_recon + norm_kl) / 2
    
    #=== P-VALUE CALCULATION ===
    
    # Get control group data
    control_mask = results_df["Diagnosis"] == norm_diagnosis
    if not control_mask.any():
        print(f"WARNING: No control group '{norm_diagnosis}' found in data. Available diagnoses: {results_df['Diagnosis'].unique()}")
        # Use bottom 25% as reference if no explicit control group
        control_indices = np.argsort(results_df["deviation_score_zscore"])[:len(results_df)//4]
        control_mask = np.zeros(len(results_df), dtype=bool)
        control_mask[control_indices] = True
        print(f"Using bottom 25% ({control_mask.sum()} subjects) as reference group")
    
    control_recon = results_df.loc[control_mask, "reconstruction_error"]  #nur die norm Patienten -> norm errors
    control_kl = results_df.loc[control_mask, "kl_divergence"]  #nur die norm Patienten -> norm errors
    control_combined = results_df.loc[control_mask, "deviation_score_zscore"]  #nur die norm Patienten -> norm errors
    
    # Calculate p-values for each subject
    p_values_recon = []
    p_values_kl = []
    p_values_combined = []

    for idx, row in results_df.iterrows():
        # P-value: probability of observing this value or higher in control distribution
        p_recon = np.mean(control_recon >= row["reconstruction_error"])
        p_kl = np.mean(control_kl >= row["kl_divergence"])
        p_combined = np.mean(control_combined >= row["deviation_score_zscore"])
        
        # WICHTIG: Werte zu den Listen hinzufügen
        p_values_recon.append(p_recon)
        p_values_kl.append(p_kl)
        p_values_combined.append(p_combined)

    # Nach der Schleife zum DataFrame hinzufügen
    results_df["p_value_recon"] = p_values_recon
    results_df["p_value_kl"] = p_values_kl
    results_df["p_value_combined"] = p_values_combined

    return results_df

def plot_deviation_distributions(results_df, save_dir):
    """Plot distributions of deviation metrics by diagnosis group."""
    print(results_df["Diagnosis"].unique())  # Ensure expected categories exist

    os.makedirs(f"{save_dir}/figures/distributions", exist_ok=True)
    
    # Create color palette
    palette = sns.light_palette("blue", n_colors=4, reverse=True)
    diagnosis_order = ["HC", "SCHZ", "CTT", "MDD"]
    diagnosis_palette = dict(zip(diagnosis_order, palette))

    #########
    # Helper function to calculate group statistics including p-values
    def calculate_group_stats(results_df, metric, p_value_col):
        stats_dict = {}
        for diagnosis in diagnosis_order:
            group_data = results_df[results_df["Diagnosis"] == diagnosis]
            if len(group_data) > 0:
                stats_dict[diagnosis] = {
                    'mean': group_data[metric].mean(),
                    'std': group_data[metric].std(),
                    'n': len(group_data),
                    'mean_p_value': group_data[p_value_col].mean(),
                    'significant_count': (group_data[p_value_col] < 0.05).sum(),
                    'median_p_value': group_data[p_value_col].median()
                }
        return stats_dict
    
    # Plot reconstruction error distributions with p-value info
    plt.figure(figsize=(14, 10))
    
    # Main KDE plot
    plt.subplot(2, 1, 1)
    sns.kdeplot(data=results_df, x="reconstruction_error", hue="Diagnosis", 
                palette=diagnosis_palette, common_norm=False)
    plt.title("Reconstruction Error Distribution by Diagnosis", fontsize=16)
    plt.xlabel("Mean Reconstruction Error", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(title="Diagnosis", fontsize=12)
    
    # Add p-value histogram subplot
    plt.subplot(2, 1, 2)
    for diagnosis in diagnosis_order:
        group_data = results_df[results_df["Diagnosis"] == diagnosis]
        if len(group_data) > 0:
            plt.hist(group_data["p_value_recon"], bins=20, alpha=0.6, 
                    label=f'{diagnosis} (n={len(group_data)})', 
                    color=diagnosis_palette.get(diagnosis, 'gray'))
    
    plt.axvline(x=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
    plt.title("P-value Distribution for Reconstruction Error", fontsize=14)
    plt.xlabel("P-value", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend(fontsize=10)
    

    # Enhanced violin plots with p-value annotations
    plt.figure(figsize=(16, 12))
    
    metrics_info = [
        ("reconstruction_error", "p_value_recon", "Reconstruction Error"),
        ("kl_divergence", "p_value_kl", "KL Divergence"), 
        ("deviation_score", "p_value_combined", "Combined Deviation Score")
    ]
    
    for i, (metric, p_col, title) in enumerate(metrics_info, 1):
        plt.subplot(3, 1, i)
        
        # Create violin plot
        ax = sns.violinplot(data=results_df, x="Diagnosis", y=metric, 
                           palette=diagnosis_palette, order=diagnosis_order)
        
        # Calculate and add statistics annotations
        stats = calculate_group_stats(results_df, metric, p_col)
        
        # Add text annotations with statistics
        for j, diagnosis in enumerate(diagnosis_order):
            if diagnosis in stats:
                s = stats[diagnosis]
                annotation = (f"n={s['n']}\n"
                            f"μ={s['mean']:.3f}±{s['std']:.3f}\n"
                            f"p̄={s['mean_p_value']:.3f}\n"
                            f"sig: {s['significant_count']}/{s['n']}")
                
                # Position annotation above violin
                y_pos = results_df[results_df["Diagnosis"] == diagnosis][metric].max()
                ax.text(j, y_pos * 1.05, annotation, ha='center', va='bottom', 
                       fontsize=9, bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor='white', alpha=0.8))
        
        plt.title(f"{title} by Diagnosis", fontsize=14)
        if i < 3:
            plt.xlabel("")
        else:
            plt.xlabel("Diagnosis", fontsize=12)
            
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/distributions/enhanced_violin_plots.png", dpi=300)
    plt.close()
    
    # Create summary table with p-value statistics
    selected_diagnoses = ["SCHZ", "CTT", "MDD", "HC"]
    metrics = ["reconstruction_error", "kl_divergence", "deviation_score"]
    p_cols = ["p_value_recon", "p_value_kl", "p_value_combined"]
    
    summary_dict = {}
    
    for metric, p_col in zip(metrics, p_cols):
        summary_data = []
        
        for diagnosis in selected_diagnoses:
            group_data = results_df[results_df["Diagnosis"] == diagnosis]
            if len(group_data) > 0:
                summary_data.append({
                    'Diagnosis': diagnosis,
                    'n': len(group_data),
                    'mean': group_data[metric].mean(),
                    'std': group_data[metric].std(),
                    'mean_p_value': group_data[p_col].mean(),
                    'median_p_value': group_data[p_col].median(),
                    'significant_count': (group_data[p_col] < 0.05).sum(),
                    'percent_significant': (group_data[p_col] < 0.05).mean() * 100
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Calculate 95% confidence interval
        summary_df["ci95"] = 1.96 * summary_df["std"] / np.sqrt(summary_df["count"] if "count" in summary_df.columns else summary_df["n"])
        
        # Sort for plotting
        diagnosis_order_rev = selected_diagnoses[::-1]
        summary_df["Diagnosis"] = pd.Categorical(summary_df["Diagnosis"], 
                                               categories=diagnosis_order_rev, ordered=True)
        summary_df = summary_df.sort_values("Diagnosis")
        
        summary_dict[metric] = summary_df
        
        # Enhanced errorbar plot with p-value info
        plt.figure(figsize=(10, 6))
        
        # Main errorbar plot
        plt.subplot(1, 2, 1)
        plt.errorbar(summary_df["mean"], summary_df["Diagnosis"], 
                    xerr=summary_df.get("ci95", summary_df["std"]),
                    fmt='s', color='black', capsize=5, markersize=8)
        
        # Add mean p-value as color coding
        scatter = plt.scatter(summary_df["mean"], summary_df["Diagnosis"], 
                            c=summary_df["mean_p_value"], cmap='RdYlBu_r', 
                            s=100, alpha=0.7, edgecolors='black')
        
        #plt.colorbar(scatter, label='Mean P-value')
        plt.title(f"{metric.replace('_', ' ').title()}", fontsize=14)
        plt.xlabel("Deviation Metric", fontsize=12)
        
        # P-value significance bar chart
        plt.subplot(1, 2, 2)
        colors = ['red' if p < 0.05 else 'gray' for p in summary_df["percent_significant"]]
        bars = plt.barh(summary_df["Diagnosis"], summary_df["percent_significant"], 
                       color=colors, alpha=0.7)
        
        # Add percentage labels on bars
        for i, (bar, pct) in enumerate(zip(bars, summary_df["percent_significant"])):
            plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    f'{pct:.1f}%', va='center', fontsize=10)
        
        plt.axvline(x=5, color='red', linestyle='--', alpha=0.5, label='5% threshold')
        plt.title("% Subjects with p < 0.05", fontsize=14)
        plt.xlabel("Percentage", fontsize=12)
        plt.legend()
        
        sns.despine()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/figures/distributions/{metric}_enhanced_errorbar.png", dpi=300)
        plt.close()
    ############

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
    ############
    # Create comprehensive summary table and save as CSV
    comprehensive_summary = []
    for metric, p_col in zip(metrics, p_cols):
        for diagnosis in selected_diagnoses:
            group_data = results_df[results_df["Diagnosis"] == diagnosis]
            if len(group_data) > 0:
                comprehensive_summary.append({
                    'Metric': metric,
                    'Diagnosis': diagnosis,
                    'N': len(group_data),
                    'Mean': group_data[metric].mean(),
                    'Std': group_data[metric].std(),
                    'Mean_P_Value': group_data[p_col].mean(),
                    'Median_P_Value': group_data[p_col].median(),
                    'Significant_Count': (group_data[p_col] < 0.05).sum(),
                    'Percent_Significant': (group_data[p_col] < 0.05).mean() * 100,
                    'Min_P_Value': group_data[p_col].min(),
                    'Max_P_Value': group_data[p_col].max()
                })
    
    comprehensive_df = pd.DataFrame(comprehensive_summary)
    comprehensive_df.to_csv(f"{save_dir}/figures/distributions/summary_with_pvalues.csv", index=False)
    
    print(f"Summary with p-values saved to {save_dir}/figures/distributions/summary_with_pvalues.csv")
    #######
    return summary_dict

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
        atlas_name (str): Name of the atlas
        roi_names (list): List of ROI names
        norm_diagnosis (str): Normative diagnosis group label
        
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
    os.makedirs(save_dir, exist_ok=True)
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
    
    if effect_sizes_df.empty:
        print("No effect sizes calculated. Returning empty DataFrame.")
        return effect_sizes_df
    
    # BUGFIX: Add absolute Cliff's Delta column
    effect_sizes_df["Abs_Cliffs_Delta"] = effect_sizes_df["Cliffs_Delta"].abs()
    effect_sizes_df["Abs_Cohens_d"] = effect_sizes_df["Cohens_d"].abs()
    
    # Save complete effect sizes
    effect_sizes_df.to_csv(f"{save_dir}/regional_effect_sizes_vs_{norm_diagnosis}.csv", index=False)
    
    # Create figures directory
    os.makedirs(f"{save_dir}/figures", exist_ok=True)
    
    # Create visualization of top affected regions for each diagnosis
    for diagnosis in diagnoses:
        if diagnosis == norm_diagnosis:
            continue
            
        # Filter data for this diagnosis
        dx_effect_sizes = effect_sizes_df[effect_sizes_df["Diagnosis"] == diagnosis].copy()
        
        if dx_effect_sizes.empty:
            continue
            
        # Sort by absolute effect size (Cliff's Delta)
        dx_effect_sizes_sorted = dx_effect_sizes.sort_values("Abs_Cliffs_Delta", ascending=False)
        
        # Take top 20 regions
        top_regions = dx_effect_sizes_sorted.head(20)
        
        # Create bar plot for Cliff's Delta
        plt.figure(figsize=(14, 10))
        bars = plt.barh(range(len(top_regions)), top_regions["Cliffs_Delta"])
        
        # Color bars based on direction of effect
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
        
        # Color bars based on direction of effect
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
    plt.savefig(f"{save_dir}/figures/effect_size_distributions_vs_{norm_diagnosis}.png", dpi=300, bbox_inches='tight')
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
        plt.savefig(f"{save_dir}/figures/region_effect_heatmap_vs_{norm_diagnosis}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save heatmap data
        heatmap_df.to_csv(f"{save_dir}/top_regions_heatmap_data_vs_{norm_diagnosis}.csv")
    
    print(f"\nRegional analysis completed. Results saved to {save_dir}")
    print(f"Total effect sizes calculated: {len(effect_sizes_df)}")
    print(f"Average absolute Cliff's Delta: {effect_sizes_df['Abs_Cliffs_Delta'].mean():.3f}")
    print(f"Max absolute Cliff's Delta: {effect_sizes_df['Abs_Cliffs_Delta'].max():.3f}")
    
    return effect_sizes_df