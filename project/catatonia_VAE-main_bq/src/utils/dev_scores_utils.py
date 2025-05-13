
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
    Calculate deviation scores using bootstrap models.
    
    Args:
        normative_models: List of trained normative VAE models
        data_tensor: Tensor of input data to evaluate
        annotations_df: DataFrame with subject metadata
        device: Computing device
        
    Returns:
        DataFrame with deviation scores for each subject
    """
    total_models = len(normative_models)
    total_subjects = data_tensor.shape[0]
    
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
            #Mean squared error between original brain measurements and their reconstruction
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
    
    # Create result DataFrame
    results_df = annotations_df[["Filename", "Diagnosis", "Age", "Sex", "Dataset"]].copy()
    results_df["reconstruction_error"] = mean_recon_error
    results_df["reconstruction_error_std"] = std_recon_error
    results_df["kl_divergence"] = mean_kl_div
    results_df["kl_divergence_std"] = std_kl_div
    
    # Add region-wise z-scores
    for i in range(mean_region_z_scores.shape[1]):
        results_df[f"region_{i}_z_score"] = mean_region_z_scores[:, i]
    
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
    os.makedirs(f"{save_dir}/figures/distributions", exist_ok=True)
    
    # Create color palette
    palette = {
        "HC": "blue",
        "SCHZ": "red",
        "CTT": "green",
        "MDD": "purple"
    }
    
    # Plot reconstruction error distributions
    plt.figure(figsize=(12, 8))
    sns.kdeplot(data=results_df, x="reconstruction_error", hue="Diagnosis", palette=palette, common_norm=False)
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
    sns.kdeplot(data=results_df, x="kl_divergence", hue="Diagnosis", palette=palette, common_norm=False)
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
    sns.kdeplot(data=results_df, x="deviation_score", hue="Diagnosis", palette=palette, common_norm=False)
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
    sns.violinplot(data=results_df, x="Diagnosis", y="reconstruction_error", palette=palette)
    plt.title("Reconstruction Error by Diagnosis", fontsize=14)
    plt.xlabel("")
    
    plt.subplot(3, 1, 2)
    sns.violinplot(data=results_df, x="Diagnosis", y="kl_divergence", palette=palette)
    plt.title("KL Divergence by Diagnosis", fontsize=14)
    plt.xlabel("")
    
    plt.subplot(3, 1, 3)
    sns.violinplot(data=results_df, x="Diagnosis", y="deviation_score", palette=palette)
    plt.title("Combined Deviation Score by Diagnosis", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/distributions/metrics_violin_plots.png", dpi=300)
    plt.close()
    
    return



def analyze_roc_curves(results_df, save_dir):
    """Analyze ROC curves for each patient group vs. healthy controls."""
    os.makedirs(f"{save_dir}/figures/roc", exist_ok=True)
    
    # Prepare binary classification datasets
    diagnoses = ["SCHZ", "CTT", "MDD"]
    metrics = ["reconstruction_error", "kl_divergence", "deviation_score"]
    
    plt.figure(figsize=(18, 6))
    
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        
        for diagnosis in diagnoses:
            # Create binary labels (clinical group vs HC)
            binary_df = results_df[results_df["Diagnosis"].isin(["HC", diagnosis])].copy()
            binary_df["target"] = (binary_df["Diagnosis"] == diagnosis).astype(int)
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(binary_df["target"], binary_df[metric])
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, label=f'{diagnosis} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves for {metric.replace("_", " ").title()}')
        plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/roc/roc_curves.png", dpi=300)
    plt.close()
    
    # Create summary table of AUC values
    auc_results = []
    
    for diagnosis in diagnoses:
        for metric in metrics:
            binary_df = results_df[results_df["Diagnosis"].isin(["HC", diagnosis])].copy()
            binary_df["target"] = (binary_df["Diagnosis"] == diagnosis).astype(int)
            
            fpr, tpr, _ = roc_curve(binary_df["target"], binary_df[metric])
            roc_auc = auc(fpr, tpr)
            
            auc_results.append({
                "Diagnosis": diagnosis,
                "Metric": metric,
                "AUC": roc_auc
            })
    
    auc_df = pd.DataFrame(auc_results)
    auc_df.to_csv(f"{save_dir}/auc_results.csv", index=False)
    
    return auc_df


def find_top_deviant_regions(results_df, save_dir):
    """Find and visualize top deviant brain regions for each clinical group."""
    os.makedirs(f"{save_dir}/figures/regional", exist_ok=True)
    
    # Get region columns
    region_cols = [col for col in results_df.columns if col.startswith('region_')]
    
    # Calculate mean z-score for each region and diagnosis
    region_means = results_df.groupby('Diagnosis')[region_cols].mean()
    
    # For each clinical group, find difference from HC
    hc_means = region_means.loc['HC']
    
    deviation_results = {}
    
    for diagnosis in ['SCHZ', 'CTT', 'MDD']:
        if diagnosis in region_means.index:
            # Calculate difference from HC
            diff = region_means.loc[diagnosis] - hc_means
            
            # Get top 10 deviant regions
            top_regions = diff.abs().sort_values(ascending=False).head(10)
            
            # Store results
            deviation_results[diagnosis] = {
                'region_ids': top_regions.index.tolist(),
                'deviations': top_regions.values.tolist()
            }
    
    # Plot top deviant regions for each clinical group
    plt.figure(figsize=(15, 10))
    
    diagnoses = list(deviation_results.keys())
    n_diagnoses = len(diagnoses)
    
    for i, diagnosis in enumerate(diagnoses):
        plt.subplot(1, n_diagnoses, i+1)
        
        regions = deviation_results[diagnosis]['region_ids']
        deviations = deviation_results[diagnosis]['deviations']
        
        # Convert region_X to region numbers
        region_nums = [int(r.split('_')[1]) for r in regions]
        
        # Sort for better visualization
        sorted_indices = np.argsort(deviations)
        sorted_regions = [region_nums[i] for i in sorted_indices]
        sorted_deviations = [deviations[i] for i in sorted_indices]
        
        # Plot horizontal bar chart
        plt.barh(range(len(sorted_regions)), sorted_deviations, color='firebrick')
        plt.yticks(range(len(sorted_regions)), sorted_regions)
        plt.xlabel('Deviation from HC (Z-score)')
        plt.ylabel('Region ID')
        plt.title(f'Top Deviant Regions for {diagnosis}')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/regional/top_deviant_regions.png", dpi=300)
    plt.close()
    
    # Save region deviations to CSV
    for diagnosis, data in deviation_results.items():
        region_df = pd.DataFrame({
            'region_id': data['region_ids'],
            'deviation': data['deviations']
        })
        region_df.to_csv(f"{save_dir}/top_regions_{diagnosis}.csv", index=False)
    
    return deviation_results

def visualize_embeddings(normative_models, data_tensor, annotations_df, device="cuda"):
    """Visualize data in the latent space of the normative model."""
    # Use the first model for visualization
    model = normative_models[0]
    model.eval()
    model.to(device)
    
    all_embeddings = []
    batch_size = 32
    
    # Process in batches to avoid OOM
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
    
    # Combine all embeddings
    embeddings = np.vstack(all_embeddings)
    
    # UMAP for visualization
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    umap_embeddings = reducer.fit_transform(embeddings)
    
    # Create DataFrame for plotting
    plot_df = annotations_df[["Diagnosis"]].copy()
    plot_df["umap_1"] = umap_embeddings[:, 0]
    plot_df["umap_2"] = umap_embeddings[:, 1]
    
    # Plot
    plt.figure(figsize=(12, 10))
    palette = {
        "HC": "blue",
        "SCHZ": "red",
        "CTT": "green",
        "MDD": "purple"
    }
    
    sns.scatterplot(
        data=plot_df,
        x="umap_1",
        y="umap_2",
        hue="Diagnosis",
        palette=palette,
        s=100,
        alpha=0.7
    )
    
    plt.title("UMAP Visualization of Latent Space", fontsize=16)
    plt.xlabel("UMAP Dimension 1", fontsize=14)
    plt.ylabel("UMAP Dimension 2", fontsize=14)
    plt.legend(title="Diagnosis", fontsize=12)
    
    return plt.gcf(), plot_df

def calculate_cliffs_delta(x, y):
    """
    Calculate Cliff's Delta effect size (non-parametric effect size measure)
    
    Parameters:
    -----------
    x, y : array_like
        The two groups to compare
        
    Returns:
    --------
    delta : float
        Cliff's Delta effect size
    """
    # Count all pairwise comparisons
    greater = 0
    lesser = 0
    
    for i in x:
        for j in y:
            if i > j:
                greater += 1
            elif i < j:
                lesser += 1
    
    # Calculate delta
    delta = (greater - lesser) / (len(x) * len(y))
    return delta