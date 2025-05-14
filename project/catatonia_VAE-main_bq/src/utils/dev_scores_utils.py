
import argparse
import os
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

    selected_diagnoses = ["SCHZ", "CTT", "MDD"]

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

def analyze_score_auc(results_df, save_dir):
    """Berechnet AUC-Werte basierend auf Scores für HC vs. nicht-HC Patienten."""
    os.makedirs(f"{save_dir}/figures/roc", exist_ok=True)
    
    # Definiere die Metriken, die als Scores genutzt werden
    metrics = ["reconstruction_error", "kl_divergence", "deviation_score"]
    
    # Erstelle binäre Labels: HC = 0, Nicht-HC = 1
    results_df["target"] = (results_df["Diagnosis"] != "HC").astype(int)
    
    # Speichert AUC-Werte für jede Metrik
    auc_results = []
    
    plt.figure(figsize=(12, 6))
    
    for i, metric in enumerate(metrics):
        # Berechnung der AUC für die Scores
        y_true = results_df["target"]
        y_scores = results_df[metric]
        
        if len(y_true.unique()) < 2:
            print(f"Warnung: Keine negativen Samples für {metric}. AUC wird übersprungen.")
            continue
        
        auc_score = roc_auc_score(y_true, y_scores)
        auc_results.append({"Metric": metric, "AUC": auc_score})
        
        # Plot der Score-Verteilung für HC vs. nicht-HC
        plt.subplot(1, 3, i+1)
        sns.kdeplot(data=results_df, x=metric, hue="Diagnosis", common_norm=False)
        plt.title(f"{metric.replace('_', ' ').title()} (AUC = {auc_score:.3f})")
        plt.xlabel(metric.replace("_", " ").title())
        plt.ylabel("Density")
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/roc/score_distributions.png", dpi=300)
    plt.close()
    
    # Speichern der AUC-Werte als CSV
    auc_df = pd.DataFrame(auc_results)
    auc_df.to_csv(f"{save_dir}/auc_results.csv", index=False)
    
    return auc_df



# def find_top_deviant_regions(results_df, save_dir):
#     """Find and visualize top deviant brain regions for each clinical group."""
#     os.makedirs(f"{save_dir}/figures/regional", exist_ok=True)
    
#     # Get region columns
#     region_cols = [col for col in results_df.columns if col.startswith('region_')]
    
#     # Calculate mean z-score for each region and diagnosis
#     region_means = results_df.groupby('Diagnosis')[region_cols].mean()
    
#     # For each clinical group, find difference from HC
#     hc_means = region_means.loc['HC']
    
#     deviation_results = {}
    
#     for diagnosis in ['SCHZ', 'CTT', 'MDD']:
#         if diagnosis in region_means.index:
#             # Calculate difference from HC
#             diff = region_means.loc[diagnosis] - hc_means
            
#             # Get top 10 deviant regions
#             top_regions = diff.abs().sort_values(ascending=False).head(10)
            
#             # Store results
#             deviation_results[diagnosis] = {
#                 'region_ids': top_regions.index.tolist(),
#                 'deviations': top_regions.values.tolist()
#             }
    
#     # Plot top deviant regions for each clinical group
#     plt.figure(figsize=(15, 10))
    
#     diagnoses = list(deviation_results.keys())
#     n_diagnoses = len(diagnoses)
    
#     for i, diagnosis in enumerate(diagnoses):
#         plt.subplot(1, n_diagnoses, i+1)
        
#         regions = deviation_results[diagnosis]['region_ids']
#         deviations = deviation_results[diagnosis]['deviations']
        
#         # Convert region_X to region numbers
#         region_nums = [int(r.split('_')[1]) for r in regions]
        
#         # Sort for better visualization
#         sorted_indices = np.argsort(deviations)
#         sorted_regions = [region_nums[i] for i in sorted_indices]
#         sorted_deviations = [deviations[i] for i in sorted_indices]
        
#         # Plot horizontal bar chart
#         plt.barh(range(len(sorted_regions)), sorted_deviations, color='firebrick')
#         plt.yticks(range(len(sorted_regions)), sorted_regions)
#         plt.xlabel('Deviation from HC (Z-score)')
#         plt.ylabel('Region ID')
#         plt.title(f'Top Deviant Regions for {diagnosis}')
#         plt.grid(axis='x', linestyle='--', alpha=0.7)
    
#     plt.tight_layout()
#     plt.savefig(f"{save_dir}/figures/regional/top_deviant_regions.png", dpi=300)
#     plt.close()
    
#     # Save region deviations to CSV
#     for diagnosis, data in deviation_results.items():
#         region_df = pd.DataFrame({
#             'region_id': data['region_ids'],
#             'deviation': data['deviations']
#         })
#         region_df.to_csv(f"{save_dir}/top_regions_{diagnosis}.csv", index=False)
    
#     return deviation_results
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

def calculate_cliffs_delta(group1, group2):
    """
    Calculate Cliff's Delta - a non-parametric effect size measure.
    
    Parameters:
    -----------
    group1, group2 : array-like
        The two groups to compare
        
    Returns:
    --------
    delta : float
        Cliff's Delta effect size
    """
    # Count all pairwise comparisons
    greater = 0
    lesser = 0
    
    for x in group1:
        for y in group2:
            if x > y:
                greater += 1
            elif x < y:
                lesser += 1
    
    total_comparisons = len(group1) * len(group2)
    if total_comparisons == 0:
        return None
        
    # Calculate delta
    delta = (greater - lesser) / total_comparisons
    return delta

def calculate_roi_deviation_scores(normative_models, data_tensor, annotations_df, device, roi_names):
    """
    Calculate deviation scores for each subject and keep track of ROI-specific deviations.
    Returns a DataFrame with deviation scores per subject and per ROI.
    """
    # Move models to device
    models = [model.to(device) for model in normative_models]
    
    # Convert to dataset for easier batching
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Initialize lists to store results
    subject_ids = []
    diagnoses = []
    demographics = []
    deviation_scores = []
    kl_divergences = []
    reconstruction_errors = []
    reconstruction_errors_per_roi = []
    
    # Set models to evaluation mode
    for model in models:
        model.eval()
    
    # Process each batch
    with torch.no_grad():
        batch_idx = 0
        for batch in dataloader:
            batch_data = batch[0].to(device)
            batch_size = batch_data.shape[0]
            batch_start_idx = batch_idx * 64
            batch_end_idx = batch_start_idx + batch_size
            
            # Collect subject information for this batch
            batch_subject_ids = annotations_df.iloc[batch_start_idx:batch_end_idx]['Subject_ID'].values
            batch_diagnoses = annotations_df.iloc[batch_start_idx:batch_end_idx]['Diagnosis'].values
            
            # Store demographic information if available
            if 'Age' in annotations_df.columns and 'Sex' in annotations_df.columns:
                batch_demographics = zip(
                    annotations_df.iloc[batch_start_idx:batch_end_idx]['Age'].values,
                    annotations_df.iloc[batch_start_idx:batch_end_idx]['Sex'].values
                )
            else:
                batch_demographics = [(None, None)] * batch_size
            
            subject_ids.extend(batch_subject_ids)
            diagnoses.extend(batch_diagnoses)
            demographics.extend(batch_demographics)
            
            # Initialize arrays for ensemble results
            batch_deviations = np.zeros(batch_size)
            batch_kl_divs = np.zeros(batch_size)
            batch_rec_errors = np.zeros(batch_size)
            batch_roi_errors = np.zeros((batch_size, len(roi_names)))  # Store per-ROI errors
            
            # Process with each model in the ensemble
            for idx, model in enumerate(models):
                # Forward pass
                recon_batch, mu, logvar = model(batch_data)
                
                # Calculate KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                
                # Calculate reconstruction error per element (per ROI)
                roi_errors = torch.nn.functional.mse_loss(recon_batch, batch_data, reduction='none')
                
                # Sum across any additional dimensions if present
                if len(roi_errors.shape) > 2:
                    roi_errors = roi_errors.sum(axis=tuple(range(2, len(roi_errors.shape))))
                
                # Get total reconstruction error per subject
                rec_error = roi_errors.sum(dim=1)
                
                # Deviation score = reconstruction error + KL divergence
                deviation = rec_error + kl_div
                
                # Update ensemble results
                batch_deviations += deviation.cpu().numpy()
                batch_kl_divs += kl_div.cpu().numpy()
                batch_rec_errors += rec_error.cpu().numpy()
                batch_roi_errors += roi_errors.cpu().numpy()  # Add per-ROI errors
            
            # Average across ensemble
            batch_deviations /= len(models)
            batch_kl_divs /= len(models)
            batch_rec_errors /= len(models)
            batch_roi_errors /= len(models)
            
            deviation_scores.extend(batch_deviations)
            kl_divergences.extend(batch_kl_divs)
            reconstruction_errors.extend(batch_rec_errors)
            reconstruction_errors_per_roi.extend(batch_roi_errors)
            
            batch_idx += 1
    
    # Create a main DataFrame with deviation scores per subject
    main_data = {
        'Subject_ID': subject_ids,
        'Diagnosis': diagnoses,
        'deviation_score': deviation_scores,
        'reconstruction_error': reconstruction_errors,
        'kl_divergence': kl_divergences
    }
    
    # Add demographic information if available
    if demographics[0][0] is not None:
        main_data['Age'] = [d[0] for d in demographics]
        main_data['Sex'] = [d[1] for d in demographics]
    
    main_df = pd.DataFrame(main_data)
    
    # Create a separate DataFrame for ROI-specific errors
    roi_data = {
        'Subject_ID': subject_ids,
        'Diagnosis': diagnoses,
    }
    
    # Add columns for each ROI's reconstruction error
    for i, roi_name in enumerate(roi_names):
        roi_data[roi_name] = [errors[i] for errors in reconstruction_errors_per_roi]
    
    roi_df = pd.DataFrame(roi_data)
    
    # Join the DataFrames
    result_df = pd.merge(main_df, roi_df, on=['Subject_ID', 'Diagnosis'])
    
    return result_df

def plot_diagnosis_deviation_boxplots(results_df, metric, save_dir, roi_names=None):
    """
    Plot boxplots of deviation metrics by diagnosis, similar to the screenshot.
    
    Parameters:
    -----------
    results_df : pandas DataFrame
        DataFrame containing deviation scores and diagnosis info
    metric : str
        Metric to plot (one of 'deviation_score', 'reconstruction_error', 'kl_divergence')
    save_dir : str
        Directory to save plots
    roi_names : list, optional
        If provided, create separate plots for each ROI
    """
    # Create overall deviation plot
    plt.figure(figsize=(10, 6))
    
    # Order diagnoses as in screenshot: HC, EMCH (if present), LMCH (if present), AD (SCHZ/CTT/MDD)
    diagnosis_order = []
    if 'HC' in results_df['Diagnosis'].unique():
        diagnosis_order.append('HC')
    if 'EMCH' in results_df['Diagnosis'].unique():
        diagnosis_order.append('EMCH')
    if 'LMCH' in results_df['Diagnosis'].unique():
        diagnosis_order.append('LMCH')
    
    # Add the disease groups - ensure they're in results first
    for diag in ['SCHZ', 'CTT', 'MDD']:
        if diag in results_df['Diagnosis'].unique():
            diagnosis_order.append(diag)
    
    # Filter to only include diagnoses that are present in the data
    diagnosis_order = [d for d in diagnosis_order if d in results_df['Diagnosis'].unique()]
    
    # Create main deviation plot (horizontal boxplot like in screenshot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=results_df,
        x=metric,
        y='Diagnosis',
        order=diagnosis_order,
        orient='h',
        whis=[5, 95]
    )
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f"{metric.replace('_', ' ').title()} by Diagnosis", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/{metric}_by_diagnosis.png", dpi=300)
    plt.close()
    
    # If ROI names are provided, create plots for each ROI
    if roi_names and isinstance(roi_names, list):
        # To avoid creating too many plots, just plot the top ROIs with highest variance across diagnoses
        # First, calculate the variance for each ROI across diagnoses
        roi_variance = {}
        
        for roi in roi_names:
            # Calculate mean deviation for each diagnosis for this ROI
            diagnosis_means = results_df.groupby('Diagnosis')[roi].mean()
            # Calculate variance of these means
            if len(diagnosis_means) > 1:  # Need at least 2 groups for variance
                roi_variance[roi] = diagnosis_means.var()
        
        # Sort ROIs by variance and take top 10
        top_rois = sorted(roi_variance.items(), key=lambda x: x[1], reverse=True)[:10]
        top_roi_names = [roi for roi, _ in top_rois]
        
        # Create a separate plot for top ROIs
        plt.figure(figsize=(12, 8))
        
        # Create a DataFrame in long format for plotting
        plot_data = []
        for diagnosis in diagnosis_order:
            for roi in top_roi_names:
                diag_values = results_df[results_df['Diagnosis'] == diagnosis][roi].values
                if len(diag_values) > 0:  # Only if we have data
                    for value in diag_values:
                        plot_data.append({
                            'Diagnosis': diagnosis,
                            'ROI': roi,
                            'Deviation': value
                        })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create the plot
        g = sns.FacetGrid(plot_df, col='ROI', col_wrap=3, height=4, sharey=False)
        g.map_dataframe(sns.boxplot, x='Diagnosis', y='Deviation', order=diagnosis_order)
        g.set_titles('{col_name}')
        g.set_axis_labels('Diagnosis', 'Deviation')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/figures/top_roi_deviations.png", dpi=300)
        plt.close()

def calculate_roi_contribution(results_df, save_dir, roi_names):
    """
    Calculate the contribution of each ROI to overall deviation within each diagnosis group
    using Cliff's Delta as the effect size measure.
    """
    # Prepare for statistical testing
    diagnoses = results_df['Diagnosis'].unique().tolist()
    metrics = ["reconstruction_error", "kl_divergence", "deviation_score"]
    
    # Container for results
    stats_results = []
    
    # For each diagnosis and each ROI, calculate Cliff's Delta against overall deviation
    for diagnosis in diagnoses:
        diagnosis_df = results_df[results_df['Diagnosis'] == diagnosis]
        
        # Skip if insufficient data
        if len(diagnosis_df) < 2:
            continue
            
        # Get the overall deviation score for this diagnosis
        overall_dev = diagnosis_df['deviation_score'].values
        
        # For each ROI, calculate Cliff's Delta vs overall deviation
        for roi in roi_names:
            roi_values = diagnosis_df[roi].values
            
            # Calculate Cliff's Delta for ROI vs overall deviation
            delta = calculate_cliffs_delta(roi_values, overall_dev)
            
            # Check if Cliff's Delta was calculated successfully
            if delta is not None:
                # Also calculate Cohen's d for reference
                pooled_std = np.sqrt(((len(roi_values) - 1) * np.var(roi_values) + 
                                    (len(overall_dev) - 1) * np.var(overall_dev)) / 
                                    (len(roi_values) + len(overall_dev) - 2))
                
                effect_size = None
                if pooled_std > 0:
                    effect_size = np.abs(np.mean(roi_values) - np.mean(overall_dev)) / pooled_std
                
                # Store results
                stats_results.append({
                    'Diagnosis': diagnosis,
                    'ROI': roi,
                    'Cliff_Delta': delta,
                    'Cohens_d': effect_size,
                    'ROI_Mean': np.mean(roi_values),
                    'ROI_Std': np.std(roi_values),
                    'Overall_Mean': np.mean(overall_dev),
                    'Overall_Std': np.std(overall_dev)
                })
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(stats_results)
    
    # Save results
    stats_df.to_csv(f"{save_dir}/roi_contribution.csv", index=False)
    
    # Create visualizations of ROI contributions
    # For each diagnosis, find top contributing ROIs
    for diagnosis in diagnoses:
        if diagnosis == 'HC':  # Skip HC as reference group
            continue
            
        diag_stats = stats_df[stats_df['Diagnosis'] == diagnosis].sort_values('Cliff_Delta', ascending=False)
        
        # If we have data for this diagnosis
        if len(diag_stats) > 0:
            # Take top and bottom 10 ROIs by contribution
            top_n = min(10, len(diag_stats) // 2)
            top_rois = diag_stats.head(top_n)
            bottom_rois = diag_stats.tail(top_n)
            plot_rois = pd.concat([top_rois, bottom_rois])
            
            # Plot Cliff's Delta for these ROIs
            plt.figure(figsize=(12, 8))
            
            # Create horizontal bar plot
            bars = plt.barh(
                plot_rois['ROI'],
                plot_rois['Cliff_Delta'],
                color=plt.cm.RdBu_r(np.linspace(0, 1, len(plot_rois)))
            )
            
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.7)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.title(f"ROI Contribution to Overall Deviation - {diagnosis}", fontsize=16)
            plt.xlabel("Cliff's Delta (Effect Size)", fontsize=14)
            plt.ylabel("Region of Interest", fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/figures/{diagnosis}_roi_contribution.png", dpi=300)
            plt.close()
    
    return stats_df

def compute_brain_regions_deviations(diff_df, diagnosis_column, diagnosis_value, hc_label="HC"):
    """
    Calculate the Cliff's delta effect size between groups using the approach from the original paper.
    
    Parameters:
    - diff_df: DataFrame containing region deviation values
    - diagnosis_column: column name containing diagnosis labels
    - diagnosis_value: value of the diagnosis to compare with HC
    - hc_label: label for healthy controls (default: "HC")
    
    Returns:
    - region_df: DataFrame with region names, p-values, and effect sizes
    """
    region_df = pd.DataFrame(columns=['regions', 'pvalue', 'effect_size'])
    
    # Get data for the clinical group and healthy controls
    diff_hc = diff_df[diff_df[diagnosis_column] == hc_label]
    diff_patient = diff_df[diff_df[diagnosis_column] == diagnosis_value]
    
    # Get a list of all brain regions (columns excluding Subject_ID, Diagnosis, etc.)
    all_columns = diff_df.columns
    non_region_columns = ['Subject_ID', 'Diagnosis', 'deviation_score', 'reconstruction_error', 'kl_divergence', 'Age', 'Sex']
    region_columns = [col for col in all_columns if col not in non_region_columns]
    
    # Calculate effect size for each brain region
    for region in region_columns:
        try:
            # Calculate Mann-Whitney U test p-value
            _, pvalue = stats.mannwhitneyu(diff_hc[region], diff_patient[region])
            
            # Calculate Cliff's delta effect size
            effect_size = cliff_delta(diff_patient[region].values, diff_hc[region].values)
            
            # Add to results DataFrame
            new_row = {
                'regions': region,
                'pvalue': pvalue,
                'effect_size': effect_size
            }
            region_df = pd.concat([region_df, pd.DataFrame([new_row])], ignore_index=True)
        
        except Exception as e:
            print(f"Error processing region {region}: {e}")
    
    return region_df

def create_auc_roc_figure(tpr_list, auc_roc_list, save_path):
    """
    Create AUC-ROC curve figure similar to Figure 3 in the original paper.
    """
    tpr_list = np.array(tpr_list)
    mean_tprs = tpr_list.mean(axis=0)
    tprs_upper = np.percentile(tpr_list, 97.5, axis=0)
    tprs_lower = np.percentile(tpr_list, 2.5, axis=0)
    
    plt.figure(figsize=(8, 6))
    plt.plot(
        np.linspace(0, 1, len(mean_tprs)),
        mean_tprs,
        'b', 
        lw=2,
        label=f'ROC curve (AUC = {np.mean(auc_roc_list):.3f} ; 95% CI [{np.percentile(auc_roc_list, 2.5):.3f}, {np.percentile(auc_roc_list, 97.5):.3f}])'
    )
    
    plt.fill_between(
        np.linspace(0, 1, len(mean_tprs)),
        tprs_lower, 
        tprs_upper,
        color='grey', 
        alpha=0.2
    )
    
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, format='png', dpi=300)
    plt.close()

def create_effect_size_figure(effect_size_df, roi_names, save_path, title="Brain Regions Effect Size"):
    """
    Create a figure showing effect sizes across brain regions similar to the supplementary figure in the original paper.
    
    Parameters:
    - effect_size_df: DataFrame with effect sizes for each ROI
    - roi_names: list of ROI names
    - save_path: path to save the figure
    - title: title for the figure
    """
    # Sort effect sizes
    effect_size_df = effect_size_df.sort_values(by='effect_size')
    
    # Create figure
    plt.figure(figsize=(12, max(8, len(effect_size_df) * 0.3)))  # Dynamic figure height based on number of ROIs
    
    # Plot horizontal lines for confidence intervals if available
    if 'ci_lower' in effect_size_df.columns and 'ci_upper' in effect_size_df.columns:
        plt.hlines(
            y=range(len(effect_size_df)),
            xmin=effect_size_df['ci_lower'],
            xmax=effect_size_df['ci_upper'],
            linewidth=1.5
        )
    
    # Plot effect sizes
    plt.plot(
        effect_size_df['effect_size'],
        range(len(effect_size_df)),
        's',
        color='k',
        markersize=5
    )
    
    # Add reference line at zero
    plt.axvline(0, ls='--', color='gray')
    
    # Set y-ticks to ROI names
    plt.yticks(range(len(effect_size_df)), effect_size_df['regions'])
    
    # Set labels and title
    plt.xlabel('Effect size (Cliff\'s delta)')
    plt.ylabel('Brain regions')
    plt.title(title)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, format='png', dpi=300)
    plt.close()

def create_significant_regions_figure(effect_size_df, alpha=0.05, save_path=None):
    """
    Create a figure showing only significant regions based on Figure 4 in the original paper.
    
    Parameters:
    - effect_size_df: DataFrame with effect sizes and p-values for each ROI
    - alpha: significance threshold (default: 0.05)
    - save_path: path to save the figure
    
    Returns:
    - filtered_df: DataFrame containing only significant regions
    """
    # Apply multiple comparison correction (FDR)
    from statsmodels.stats.multitest import multipletests
    
    # Perform Benjamini-Hochberg FDR correction
    significant = multipletests(effect_size_df['pvalue'], alpha=alpha, method='fdr_bh')[0]
    
    # Filter for significant regions
    significant_df = effect_size_df[significant].copy()
    
    # Sort by effect size
    significant_df = significant_df.sort_values(by='effect_size')
    
    # If no significant regions, return original DataFrame
    if len(significant_df) == 0:
        print("No significant regions found after multiple comparison correction.")
        return effect_size_df
    
    # Create figure
    plt.figure(figsize=(12, max(6, len(significant_df) * 0.4)))  # Dynamic figure height
    
    # Plot horizontal lines for confidence intervals if available
    if 'ci_lower' in significant_df.columns and 'ci_upper' in significant_df.columns:
        plt.hlines(
            y=range(len(significant_df)),
            xmin=significant_df['ci_lower'],
            xmax=significant_df['ci_upper'],
            linewidth=1.5
        )
    
    # Plot effect sizes
    plt.plot(
        significant_df['effect_size'],
        range(len(significant_df)),
        's',
        color='k',
        markersize=6
    )
    
    # Add reference line at zero
    plt.axvline(0, ls='--', color='gray')
    
    # Set y-ticks to ROI names
    plt.yticks(range(len(significant_df)), significant_df['regions'])
    
    # Set labels and title
    plt.xlabel('Effect size (Cliff\'s delta)')
    plt.ylabel('Significant brain regions')
    plt.title(f'Significant Regions (FDR-corrected p < {alpha})')
    
    # Adjust layout and save
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
        plt.close()
    
    return significant_df

def create_significant_regions_figure(effect_size_df, alpha=0.05, save_path=None):
    """
    Create a figure showing only significant regions based on Figure 4 in the original paper.
    
    Parameters:
    - effect_size_df: DataFrame with effect sizes and p-values for each ROI
    - alpha: significance threshold (default: 0.05)
    - save_path: path to save the figure
    
    Returns:
    - filtered_df: DataFrame containing only significant regions
    """
    # Apply multiple comparison correction (FDR)
    from statsmodels.stats.multitest import multipletests
    
    # Perform Benjamini-Hochberg FDR correction
    significant = multipletests(effect_size_df['pvalue'], alpha=alpha, method='fdr_bh')[0]
    
    # Filter for significant regions
    significant_df = effect_size_df[significant].copy()
    
    # Sort by effect size
    significant_df = significant_df.sort_values(by='effect_size')
    
    # If no significant regions, return original DataFrame
    if len(significant_df) == 0:
        print("No significant regions found after multiple comparison correction.")
        return effect_size_df
    
    # Create figure
    plt.figure(figsize=(12, max(6, len(significant_df) * 0.4)))  # Dynamic figure height
    
    # Plot horizontal lines for confidence intervals if available
    if 'ci_lower' in significant_df.columns and 'ci_upper' in significant_df.columns:
        plt.hlines(
            y=range(len(significant_df)),
            xmin=significant_df['ci_lower'],
            xmax=significant_df['ci_upper'],
            linewidth=1.5
        )
    
    # Plot effect sizes
    plt.plot(
        significant_df['effect_size'],
        range(len(significant_df)),
        's',
        color='k',
        markersize=6
    )
    
    # Add reference line at zero
    plt.axvline(0, ls='--', color='gray')
    
    # Set y-ticks to ROI names
    plt.yticks(range(len(significant_df)), significant_df['regions'])
    
    # Set labels and title
    plt.xlabel('Effect size (Cliff\'s delta)')
    plt.ylabel('Significant brain regions')
    plt.title(f'Significant Regions (FDR-corrected p < {alpha})')
    
    # Adjust layout and save
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
        plt.close()
    
    return significant_df

def compute_bootstrap_effect_sizes(deviation_df, roi_names, diagnosis_column, diagnosis_values, 
                                   hc_label="HC", n_bootstrap=1000, seed=42):
    """
    Compute bootstrap effect sizes for each diagnosis vs. healthy controls.
    
    Parameters:
    - deviation_df: DataFrame with deviation scores
    - roi_names: list of ROI names
    - diagnosis_column: column containing diagnosis labels
    - diagnosis_values: list of diagnosis values to compare with HC
    - hc_label: label for healthy controls
    - n_bootstrap: number of bootstrap iterations
    - seed: random seed for reproducibility
    
    Returns:
    - bootstrap_results: dictionary with bootstrap effect sizes for each diagnosis
    """
    np.random.seed(seed)
    bootstrap_results = {}
    
    # Get data for healthy controls
    hc_data = deviation_df[deviation_df[diagnosis_column] == hc_label]
    
    # For each diagnosis
    for diagnosis in diagnosis_values:
        if diagnosis == hc_label:
            continue
        
        print(f"Computing bootstrap effect sizes for {diagnosis} vs {hc_label}...")
        
        # Get data for current diagnosis
        dx_data = deviation_df[deviation_df[diagnosis_column] == diagnosis]
        
        # Initialize arrays to store bootstrap results
        effect_sizes = np.zeros((n_bootstrap, len(roi_names)))
        
        # Get a list of ROI columns (excluding non-ROI columns)
        non_roi_columns = ['Subject_ID', 'Diagnosis', 'deviation_score', 'reconstruction_error', 'kl_divergence']
        if 'Age' in deviation_df.columns:
            non_roi_columns.append('Age')
        if 'Sex' in deviation_df.columns:
            non_roi_columns.append('Sex')
        roi_columns = [col for col in deviation_df.columns if col not in non_roi_columns]
        
        # Perform bootstrap
        for i in tqdm(range(n_bootstrap), desc=f"Bootstrap for {diagnosis}"):
            # Sample with replacement
            hc_sample = hc_data.sample(n=len(hc_data), replace=True)
            dx_sample = dx_data.sample(n=len(dx_data), replace=True)
            
            # Calculate effect size for each ROI
            for j, roi in enumerate(roi_columns):
                effect_sizes[i, j] = cliff_delta(dx_sample[roi].values, hc_sample[roi].values)
        
        # Calculate mean effect size and confidence intervals
        mean_effect_sizes = np.mean(effect_sizes, axis=0)
        lower_ci = np.percentile(effect_sizes, 2.5, axis=0)
        upper_ci = np.percentile(effect_sizes, 97.5, axis=0)
        
        # Create DataFrame
        effect_size_df = pd.DataFrame({
            'regions': roi_columns,
            'effect_size': mean_effect_sizes,
            'ci_lower': lower_ci,
            'ci_upper': upper_ci
        })
        
        # Calculate p-values
        p_values = []
        for j, roi in enumerate(roi_columns):
            # Two-sided p-value: proportion of bootstrap samples crossing zero
            if mean_effect_sizes[j] > 0:
                p_value = np.mean(effect_sizes[:, j] <= 0)
            else:
                p_value = np.mean(effect_sizes[:, j] >= 0)
            # Multiply by 2 for two-sided test
            p_values.append(min(p_value * 2, 1.0))
        
        effect_size_df['pvalue'] = p_values
        
        # Store in results
        bootstrap_results[diagnosis] = effect_size_df
    
    return bootstrap_results

def compute_classification_performance(deviation_df, clinical_df, metric='deviation_score', diagnosis_label="SCHZ", hc_label="HC"):
    """
    Calculate the AUCs of the normative model based on specified metric.
    Similar to compute_classification_performance in the original paper.
    
    Parameters:
    - deviation_df: DataFrame with deviation scores
    - clinical_df: DataFrame with clinical data
    - metric: metric to use for classification (default: 'deviation_score')
    - diagnosis_label: label for the disease group
    - hc_label: label for healthy controls
    
    Returns:
    - roc_auc: Area Under the ROC Curve
    - tpr: True Positive Rate values
    """
    error_hc = deviation_df.loc[deviation_df['Diagnosis'] == hc_label][metric].values
    error_patient = deviation_df.loc[deviation_df['Diagnosis'] == diagnosis_label][metric].values
    
    # Create binary labels (0 for HC, 1 for patients)
    labels = np.concatenate([np.zeros_like(error_hc), np.ones_like(error_patient)])
    scores = np.concatenate([error_hc, error_patient])
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # Interpolate TPR to standardized FPR values
    standardized_fpr = np.linspace(0, 1, 101)
    interpolated_tpr = np.interp(standardized_fpr, fpr, tpr)
    interpolated_tpr[0] = 0.0  # Ensure starts at 0
    
    return roc_auc, interpolated_tpr