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
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from scipy import stats as scipy_stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import umap
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests

import warnings
warnings.filterwarnings('ignore')

#helper function 1 for dev_score plotting
# Creates a summary table showing statistics for each diagnosis group colored by the color column
#for colored jitter plots 
def create_color_summary_table(data, metric, color_col, diagnoses, save_dir):
    
    summary_stats = []
    for diagnosis in diagnoses:
        diag_data = data[data['Diagnosis_x'] == diagnosis]
        
        # Basic stats for the metric
        metric_stats = {
            'Diagnosis': diagnosis,
            'N': len(diag_data),
            f'{metric}_mean': diag_data[metric].mean(),
            f'{metric}_std': diag_data[metric].std(),
        }
        
        # Handle categorical vs continuous variables for color column
        if diag_data[color_col].dtype == 'object' or color_col in ['Sex', 'Co_Diagnosis', 'Dataset']:
            # For categorical variables, show counts and percentages
            value_counts = diag_data[color_col].value_counts()
            for val, count in value_counts.items():
                metric_stats[f'{color_col}_{val}_count'] = count
                metric_stats[f'{color_col}_{val}_percent'] = (count / len(diag_data)) * 100
        else:
            # For continuous variables, show mean, std, min, max
            metric_stats.update({
                f'{color_col}_mean': diag_data[color_col].mean(),
                f'{color_col}_std': diag_data[color_col].std(),
                f'{color_col}_min': diag_data[color_col].min(),
                f'{color_col}_max': diag_data[color_col].max()
            })
        
        summary_stats.append(metric_stats)
    
    summary_df = pd.DataFrame(summary_stats)
    
    return summary_df

def create_colored_jitter_plots(data, metadata_df, metric, summary_df, plot_order, norm_diagnosis,
                               save_dir, color_columns, diagnosis_palette, split_ctt=False, custom_colors=None):
    """Create jitter plots colored by numerical values from specified columns
    
    Args:
        data: results dataset containing the metric and diagnosis information
        metadata_df: Additional dataframe containing metadata columns for coloring (scores etc)
        split_ctt: If True, keep CTT-SCHZ and CTT-MDD separate. If False, combine as CTT
        custom_colors: Optional dict with custom color mapping for diagnoses
    """
    
    os.makedirs(f"{save_dir}/figures/distributions/colored_by_columns", exist_ok=True)
    
    # Handle CTT splitting option
    data_processed = data.copy()
    if not split_ctt:
        # Combine CTT-SCHZ and CTT-MDD into CTT
        data_processed.loc[data_processed['Diagnosis'].isin(['CTT-SCHZ', 'CTT-MDD']), 'Diagnosis'] = 'CTT'
    
    # Check if we can merge on filename or need to use index
    merged_data = pd.merge(data_processed, metadata_df, on='Filename', how='inner')
    print(f"Merged data on 'Filename' column. Merged data shape: {merged_data.shape}")
    
    if merged_data.empty:
        print("Error: Could not merge data and metadata. Check if they have common identifiers.")
        return
    #changed column names after merging
    merged_data = merged_data.rename(columns={'Age_x': 'Age', 'Sex_x': 'Sex', 'Dataset_x': 'Dataset'})
    
    # Filter color_columns to only include ones that exist in merged_data
    available_color_columns = [col for col in color_columns if col in merged_data.columns]
    
    # Define columns that have complete data (all patients) for all diagnoses vs. limited diagnoses (WhiteCAT & NSS metadata)
    complete_data_columns = ['Age', 'Sex', 'Dataset']  # Assuming these have data for all diagnoses
    limited_data_columns = [col for col in available_color_columns if col not in complete_data_columns]
    
    for color_col in available_color_columns:
        print(f"Creating plot for column: {color_col}")
        
        
        if color_col in complete_data_columns:
            # Use all diagnoses for Age and Sex -> got metadata for all
            current_plot_order = plot_order.copy()
            # Adjust plot order based on CTT splitting
            if not split_ctt and 'CTT-SCHZ' in current_plot_order and 'CTT-MDD' in current_plot_order:
                current_plot_order = [d for d in current_plot_order if d not in ['CTT-SCHZ', 'CTT-MDD']]
                if 'CTT' not in current_plot_order:
                    current_plot_order.append('CTT')
            filtered_data = merged_data.copy()
            plot_title_suffix = "All Diagnoses"
        else:
            # Use only CTT-SCHZ and CTT-MDD for other columns -> got metadata only for WhiteCAT and NSS patients
            if split_ctt:
                current_plot_order = ['CTT-SCHZ', 'CTT-MDD']
                filtered_data = merged_data[merged_data['Diagnosis_x'].isin(current_plot_order)].copy()
                plot_title_suffix = "CTT-SCHZ vs CTT-MDD"
            else:
                current_plot_order = ['CTT']
                filtered_data = merged_data[merged_data['Diagnosis_x'] == 'CTT'].copy()
                plot_title_suffix = "CTT Combined"
        
        filtered_data = filtered_data.dropna(subset=[color_col, metric])
        
        if len(filtered_data) == 0:
            print(f"Warning: No data available for {color_col} after removing missing values. Skipping this column.")
            continue
        
    
        plt.figure(figsize=(14, 6))
        color_values = filtered_data[color_col].copy()
        # Handle categorical variables by converting to numeric
        if color_values.dtype == 'object' or color_col in ['Sex', 'Co_Diagnosis', 'Dataset']:
            unique_values = color_values.unique()
            value_to_code = {val: i for i, val in enumerate(unique_values)}
            color_values_numeric = color_values.map(value_to_code)
            if color_col == 'Sex':
                colors = custom_colors.get('Sex', ['#ff69b4', '#4169e1']) if custom_colors else ['#ff69b4', '#4169e1']
                if len(unique_values) == 2:
                    cmap = LinearSegmentedColormap.from_list('sex_colors', colors, N=2)
                else:
                    cmap = plt.cm.Set1
            else:
                cmap = plt.cm.Set1
                
            color_values = color_values_numeric
            categorical_labels = unique_values
            is_categorical = True
        else:
            cmap = plt.cm.viridis
            categorical_labels = None
            is_categorical = False
        
        scatter = plt.scatter(filtered_data[metric],
                            [current_plot_order.index(diag) for diag in filtered_data['Diagnosis_x']],
                            c=color_values,
                            cmap=cmap,
                            s=30,
                            alpha=0.7,
                            edgecolors='white',
                            linewidth=0.5)
       
        y_positions = [current_plot_order.index(diag) for diag in filtered_data['Diagnosis_x']]
        jitter_strength = 0.3
        y_jittered = [y + np.random.uniform(-jitter_strength, jitter_strength) for y in y_positions]
        
        # Clear the previous scatter and create new one with jittered positions
        plt.clf()
        plt.figure(figsize=(14, 6))
        
        scatter = plt.scatter(filtered_data[metric],
                            y_jittered,
                            c=color_values,
                            cmap=cmap,
                            s=30,
                            alpha=0.7,
                            edgecolors='white',
                            linewidth=0.5)
        
        # Add colorbar with appropriate labels
        cbar = plt.colorbar(scatter)
        if is_categorical and categorical_labels is not None:
            cbar.set_ticks(range(len(categorical_labels)))
            cbar.set_ticklabels(categorical_labels)
            cbar.set_label(f'{color_col.replace("_", " ").title()}', rotation=270, labelpad=20)
        else:
            cbar.set_label(f'{color_col.replace("_", " ").title()}', rotation=270, labelpad=20)
        
        # # Add errorbars and statistics
        # current_summary = summary_df[summary_df["Diagnosis"].isin(current_plot_order)]
        
        # for i, diagnosis in enumerate(current_plot_order):
        #     diagnosis_data = current_summary[current_summary["Diagnosis"] == diagnosis]
        #     if len(diagnosis_data) > 0:
        #         mean_val = diagnosis_data["mean"].iloc[0]
        #         ci_val = diagnosis_data["ci95"].iloc[0]
                
        #         # Add errorbar
        #         plt.errorbar(mean_val, i, xerr=ci_val, fmt='none',
        #                    color='black', capsize=4, capthick=2,
        #                    elinewidth=2, alpha=0.9, zorder=10)
                
        #         # Add mean value as a larger marker
        #         plt.scatter(mean_val, i, color='black', s=100,
        #                   marker='D', alpha=0.9, zorder=11,
        #                   edgecolors='white', linewidth=1)
        
    
        plt.yticks(range(len(current_plot_order)), current_plot_order)
        plt.title(f"{metric.replace('_', ' ').title()} by Diagnosis\nColored by {color_col.replace('_', ' ').title()} ({plot_title_suffix})",
                 fontsize=14, pad=20)
        plt.xlabel(f"{metric.replace('_', ' ').title()}", fontsize=12)
        plt.ylabel("Diagnosis", fontsize=12)
        plt.grid(True, alpha=0.3, axis='x')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        ctt_suffix = "split" if split_ctt else "combined"
        filename = f"{metric}_jitterplot_colored_by_{color_col}_ctt_{ctt_suffix}.png"
        plt.savefig(f"{save_dir}/figures/distributions/colored_by_columns/{filename}",
                   dpi=300, bbox_inches='tight')
        plt.close()
        create_color_summary_table(filtered_data, metric, color_col, current_plot_order, save_dir)
 
def calculate_deviations(normative_models, data_tensor, norm_diagnosis, annotations_df, device="cuda"):
   
    # Calculate deviation scores using bootstrap models

    total_models = len(normative_models)
    total_subjects = data_tensor.shape[0]
    
    if total_subjects != len(annotations_df):
        print(f"WARNING: Size mismatch detected: {total_subjects} samples in data tensor vs {len(annotations_df)} rows in annotations")
    
        # Get filenames in annotations_df
        filenames = annotations_df["Filename"].tolist()
        
        # Create a new annotations_df with only rows that have matching data
        valid_indices = list(range(min(total_subjects, len(annotations_df))))
        aligned_annotations = annotations_df.iloc[valid_indices].reset_index(drop=True)
      
        annotations_df = aligned_annotations

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
            #does not get used anymore
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
    
    # Create a DataFrame with the new columns
    new_columns = pd.DataFrame(
        mean_region_z_scores, 
        columns=[f"region_{i}_z_score" for i in range(mean_region_z_scores.shape[1])]
    )

    results_df = pd.concat([results_df, new_columns], axis=1)

    #-------------------------------------------- CALCULATE COMBINED DEVIATION SCORE ---------------------------------------------------------------
    # Normalize both metrics to 0-1 range for easier interpretation
    #Z-score normalization
    scaler_recon = StandardScaler()
    scaler_kl = StandardScaler()
    
    z_norm_recon = scaler_recon.fit_transform(mean_recon_error.reshape(-1, 1)).flatten()
    z_norm_kl = scaler_kl.fit_transform(mean_kl_div.reshape(-1, 1)).flatten()
    
    # Combined deviation score (Z-score based)
    results_df["deviation_score_zscore"] = (z_norm_recon + z_norm_kl) / 2
    
    #Percentile-based scoring
    recon_percentiles = stats.rankdata(mean_recon_error) / len(mean_recon_error)
    kl_percentiles = stats.rankdata(mean_kl_div) / len(mean_kl_div)
    results_df["deviation_score_percentile"] = (recon_percentiles + kl_percentiles) / 2
    
    #Original min-max
    min_recon = results_df["reconstruction_error"].min()
    max_recon = results_df["reconstruction_error"].max()
    norm_recon = (results_df["reconstruction_error"] - min_recon) / (max_recon - min_recon)
    
    min_kl = results_df["kl_divergence"].min()
    max_kl = results_df["kl_divergence"].max()
    norm_kl = (results_df["kl_divergence"] - min_kl) / (max_kl - min_kl)
    
    # Combined deviation score (equal weighting of both metrics)
    results_df["deviation_score"] = (norm_recon + norm_kl) / 2

    return results_df


def calculate_group_pvalues(results_df, norm_diagnosis, split_ctt=False):
    """Calculate p-values for each diagnosis group compared to the control group
    
    Args:
        results_df: DataFrame with results
        norm_diagnosis: Control group diagnosis
        split_ctt: If True, keep CTT-SCHZ and CTT-MDD separate. If False, combine as CTT
    """

    # Handle CTT splitting
    results_processed = results_df.copy()
    if not split_ctt:
        # Combine CTT-SCHZ and CTT-MDD into CTT
        results_processed.loc[results_processed['Diagnosis'].isin(['CTT-SCHZ', 'CTT-MDD']), 'Diagnosis'] = 'CTT'

    # Get control group data
    control_mask = results_processed["Diagnosis"] == norm_diagnosis
    if not control_mask.any():
        print(f"WARNING: No control group '{norm_diagnosis}' found in data. Available diagnoses: {results_processed['Diagnosis'].unique()}")
        # Use bottom 25% as reference if no explicit control group
        control_indices = np.argsort(results_processed["deviation_score_zscore"])[:len(results_processed)//4]
        control_mask = np.zeros(len(results_processed), dtype=bool)
        control_mask[control_indices] = True
        print(f"Using bottom 25% ({control_mask.sum()} subjects) as reference group")
    
    control_data = results_processed[control_mask]
    print(f"Control group ({norm_diagnosis}) size: {len(control_data)}")
    
    # Metrics to test
    metrics = ["reconstruction_error", "kl_divergence", "deviation_score"]
    
    # Calculate p-values for each diagnosis group vs control
    group_pvalues = {}
    
    diagnoses = results_processed["Diagnosis"].unique()
    diagnoses = [d for d in diagnoses if d != norm_diagnosis]  # Exclude control group
    
    for metric in metrics:
        group_pvalues[metric] = {}
        control_values = control_data[metric].values
        
        for diagnosis in diagnoses:
            group_data = results_processed[results_processed["Diagnosis"] == diagnosis]
            if len(group_data) > 0:
                group_values = group_data[metric].values
               
                # Use Mann-Whitney U test (non-parametric)
                try:
                    statistic, p_value = scipy_stats.mannwhitneyu(
                        group_values, control_values,
                        alternative='two-sided'
                    )
                    print(f"    Mann-Whitney U: statistic={statistic:.2f}, p={p_value:.6f}")
                    
                    # Double-check with t-test for comparison
                    t_stat, t_pval = scipy_stats.ttest_ind(
                        group_values, control_values,
                        equal_var=False
                    )
                    print(f"    T-test (comparison): t={t_stat:.2f}, p={t_pval:.6f}")
                    
                    group_pvalues[metric][diagnosis] = p_value
                except Exception as e:
                    print(f"Error with statistical tests")
    
    return group_pvalues

def create_diagnosis_palette(split_ctt=False, custom_colors=None):
    """Create consistent diagnosis color palette
    
    Args:
        split_ctt: If True, separate colors for CTT-SCHZ and CTT-MDD. If False, single CTT color
        custom_colors: Optional dict with custom color mapping
    """
    
    if custom_colors:
        return custom_colors
    
    # Default color palette
    base_palette = sns.light_palette("blue", n_colors=6, reverse=True)
    
    if split_ctt:
        diagnosis_order = ["HC", "SCHZ", "MDD", "CTT", "CTT-MDD", "CTT-SCHZ"]
    else:
        diagnosis_order = ["HC", "SCHZ", "MDD", "CTT"]
        base_palette = base_palette[:4]  # Use fewer colors when not splitting CTT
    
    diagnosis_palette = dict(zip(diagnosis_order, base_palette))
    
    return diagnosis_palette

def plot_deviation_distributions(results_df, save_dir, col_jitter, norm_diagnosis, name,
                                split_ctt=False, custom_colors=None):
    """Plot distributions of deviation metrics by diagnosis group with group p-values
    
    Args:
        results_df: DataFrame with results
        save_dir: Directory to save plots
        col_jitter: Whether to create colored jitter plots
        norm_diagnosis: Control group diagnosis
        split_ctt: If True, keep CTT-SCHZ and CTT-MDD separate. If False, combine as CTT
        custom_colors: Optional dict with custom color mapping for diagnoses
    """

    os.makedirs(f"{save_dir}/figures/distributions", exist_ok=True)
    
    # Handle CTT splitting
    results_processed = results_df.copy()
    if not split_ctt:
        # Combine CTT-SCHZ and CTT-MDD into CTT
        results_processed.loc[results_processed['Diagnosis'].isin(['CTT-SCHZ', 'CTT-MDD']), 'Diagnosis'] = 'CTT'
    
    # Create color palette
    diagnosis_palette = create_diagnosis_palette(split_ctt, custom_colors)

    # Calculate group p-values
    group_pvalues = calculate_group_pvalues(results_processed, norm_diagnosis, split_ctt)

    # Determine selected diagnoses based on CTT splitting
    if split_ctt:
        selected_diagnoses = ["HC", "SCHZ", "MDD", "CTT", "CTT-MDD", "CTT-SCHZ"]
    else:
        selected_diagnoses = ["HC", "SCHZ", "MDD", "CTT"]

    # Filter to only include diagnoses that exist in the data
    available_diagnoses = [d for d in selected_diagnoses if d in results_processed["Diagnosis"].unique()]

    # Plot reconstruction error distributions
    plt.figure(figsize=(12, 8))
    sns.kdeplot(data=results_processed[results_processed['Diagnosis'].isin(available_diagnoses)], 
                x="reconstruction_error", hue="Diagnosis", palette=diagnosis_palette, common_norm=False)
    plt.title(name, fontsize=16)
    plt.xlabel("Mean Reconstruction Error", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(title="Diagnosis", fontsize=12)
    plt.tight_layout()
    ctt_suffix = "split" if split_ctt else "combined"
    plt.savefig(f"{save_dir}/figures/distributions/recon_error_dist_ctt_{ctt_suffix}.png", dpi=300)
    plt.close()
    
    # Plot KL divergence distributions
    plt.figure(figsize=(12, 8))
    sns.kdeplot(data=results_processed[results_processed['Diagnosis'].isin(available_diagnoses)], 
                x="kl_divergence", hue="Diagnosis", palette=diagnosis_palette, common_norm=False)
    plt.title(name, fontsize=16)
    plt.xlabel("Mean KL Divergence", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(title="Diagnosis", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/distributions/kl_div_dist_ctt_{ctt_suffix}.png", dpi=300)
    plt.close()
    
    # Plot combined deviation score distributions
    plt.figure(figsize=(12, 8))
    sns.kdeplot(data=results_processed[results_processed['Diagnosis'].isin(available_diagnoses)], 
                x="deviation_score", hue="Diagnosis", palette=diagnosis_palette, common_norm=False)
    plt.title(name, fontsize=16)
    plt.xlabel("Deviation Score", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(title="Diagnosis", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/distributions/deviation_score_dist_ctt_{ctt_suffix}.png", dpi=300)
    plt.close()
    
    # Plot violin plots for all metrics
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 1, 1)
    sns.violinplot(data=results_processed[results_processed['Diagnosis'].isin(available_diagnoses)], 
                   x="Diagnosis", y="reconstruction_error", palette=diagnosis_palette, order=available_diagnoses)
    plt.title("Reconstruction Error by Diagnosis", fontsize=14)
    plt.xlabel("")
    plt.subplot(3, 1, 2)
    sns.violinplot(data=results_processed[results_processed['Diagnosis'].isin(available_diagnoses)], 
                   x="Diagnosis", y="kl_divergence", hue="Diagnosis", palette=diagnosis_palette, 
                   legend=False, order=available_diagnoses)
    plt.title("KL Divergence by Diagnosis", fontsize=14)
    plt.xlabel("")
    plt.subplot(3, 1, 3)
    sns.violinplot(data=results_processed[results_processed['Diagnosis'].isin(available_diagnoses)], 
                   x="Diagnosis", y="deviation_score", palette=diagnosis_palette, order=available_diagnoses)
    plt.title("Combined Deviation Score by Diagnosis", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/distributions/metrics_violin_plots_ctt_{ctt_suffix}.png", dpi=300)
    plt.close()

    # Calculate summary statistics for errorbar plots
    metrics = ["reconstruction_error", "kl_divergence", "deviation_score"]
    summary_dict = {}

    for metric in metrics:
        # Filter data for selected diagnoses
        filtered_data = results_processed[results_processed["Diagnosis"].isin(available_diagnoses)]
        
        summary_df = (
            filtered_data
            .groupby("Diagnosis")[metric]
            .agg(['mean', 'std', 'count'])
            .reset_index()
        )
        
        # Calculate 95 confidence interval
        summary_df["ci95"] = 1.96 * summary_df["std"] / np.sqrt(summary_df["count"])
        
        # Add group p-values
        summary_df["p_value"] = summary_df["Diagnosis"].map(
            lambda d: group_pvalues[metric].get(d, np.nan) if d != norm_diagnosis else np.nan
        )
       
        # Sort in desired order (bottom to top)
        diagnosis_order_plot = available_diagnoses[::-1]
        summary_df["Diagnosis"] = pd.Categorical(summary_df["Diagnosis"], categories=diagnosis_order_plot, ordered=True)
        summary_df = summary_df.sort_values("Diagnosis")
        
        summary_dict[metric] = summary_df
        
        # Simple errorbar plot -> Pinaya paper
        plt.figure(figsize=(8, 6))
        
        # Filter only diagnoses that actually have data
        plot_order = [d for d in diagnosis_order_plot if d in filtered_data["Diagnosis"].unique()]
        
        plt.errorbar(summary_df["mean"], summary_df["Diagnosis"],
                    xerr=summary_df["ci95"],
                    fmt='s', color='black', capsize=5, markersize=8)
        
        # Add mean p-value as color coding (like in original)
        summary_df_plot = summary_df[summary_df["Diagnosis"].isin(plot_order)]
        # Use group p-values for coloring
        p_values_for_color = summary_df_plot["p_value"].fillna(0.5)  # Fill NaN with neutral value
        scatter = plt.scatter(summary_df_plot["mean"], summary_df_plot["Diagnosis"],
                            c=p_values_for_color, cmap='RdYlBu_r',
                            s=100, alpha=0.7, edgecolors='black')
        
        plt.title(f"Norm Diagnosis: {norm_diagnosis} \n {name}", fontsize=14)
        plt.xlabel(f"{metric.replace('_', ' ').title()}", fontsize=12)
        plt.ylabel("Diagnosis", fontsize=12)    
        plt.tight_layout()
        plt.savefig(f"{save_dir}/figures/distributions/{metric}_errorbar_ctt_{ctt_suffix}.png", dpi=300)
        plt.close()
        
        # Create jitterplot with p-values and mean values
        plt.figure(figsize=(12, 6))  # Made wider to accommodate value labels
        
        # Use consistent color from palette
        if 'MDD' in diagnosis_palette:
            plot_color = diagnosis_palette['MDD']
        else:
            plot_color = '#4c72b0'  # fallback color
        
        sns.stripplot(data=filtered_data, y="Diagnosis", x=metric,
                    order=plot_order, color=plot_color,
                    size=3, alpha=0.6, jitter=0.3)
        
        # Add errorbars, p-values, and mean values
        for i, diagnosis in enumerate(plot_order):
            diagnosis_data = summary_df[summary_df["Diagnosis"] == diagnosis]
            if len(diagnosis_data) > 0:
                mean_val = diagnosis_data["mean"].iloc[0]
                ci_val = diagnosis_data["ci95"].iloc[0]
                p_val = diagnosis_data["p_value"].iloc[0]
                n_val = diagnosis_data["count"].iloc[0]

                plt.errorbar(mean_val, i, xerr=ci_val, fmt='none',
                            color='black', capsize=4, capthick=1.5,
                            elinewidth=1.5, alpha=0.8)
        
        plt.title(f"{name} (vs {norm_diagnosis})", fontsize=14)
        plt.xlabel(f"{metric.replace('_', ' ').title()}", fontsize=12)
        plt.ylabel("Diagnosis", fontsize=12)
        plt.subplots_adjust(left=0.25)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/figures/distributions/{metric}_jitterplot_with_values_ctt_{ctt_suffix}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()

        if col_jitter:
            metadata_df = pd.read_csv('/workspace/project/catatonia_VAE-main_bq/metadata_20250110/full_data_with_codiagnosis_and_scores.csv')
            #column names in the metadata df that should be used for coloring
            potential_color_columns = ['Age', 'Sex', 'Dataset',
                                       'GAF_Score', 'PANSS_Positive', 'PANSS_Negative',
                                       'PANSS_General', 'PANSS_Total', 'BPRS_Total', 'NCRS_Motor',
                                       'NCRS_Affective', 'NCRS_Behavioral', 'NCRS_Total', 'NSS_Motor', 'NSS_Total']

            color_columns = [col for col in potential_color_columns if col in metadata_df.columns]
            print(f"Found {len(color_columns)} columns for coloring: {color_columns}")

            if len(color_columns) == 0:
                print("No color columns found! Please check your column names in the metadata dataframe.")
            else:
                
                create_colored_jitter_plots(
                    data=filtered_data,
                    metadata_df=metadata_df,
                    metric=metric,    
                    summary_df=summary_df,
                    plot_order=plot_order,
                    norm_diagnosis=norm_diagnosis,
                    save_dir=save_dir,
                    color_columns=color_columns,
                    diagnosis_palette=diagnosis_palette,
                    split_ctt=split_ctt,
                    custom_colors=custom_colors
                )

    return summary_dict

def setup_plotting_parameters(split_ctt=False, custom_colors=None):
    """Setup consistent plotting parameters for all functions
    
    Args:
        split_ctt: If True, keep CTT-SCHZ and CTT-MDD separate. If False, combine as CTT
        custom_colors: Optional dict with custom color mapping for diagnoses
        
    Returns:
        dict: Dictionary with plotting parameters
    """
    
    return {
        'split_ctt': split_ctt,
        'custom_colors': custom_colors,
        'diagnosis_palette': create_diagnosis_palette(split_ctt, custom_colors)
    }

def run_analysis_with_options(results_df, save_dir, col_jitter, norm_diagnosis, name,
                             split_ctt=False, custom_colors=None):
    """Run complete analysis with CTT splitting and color options
    
    Args:
        results_df: DataFrame with deviation results
        save_dir: Directory to save outputs
        col_jitter: Whether to create colored jitter plots
        norm_diagnosis: Control group diagnosis
        split_ctt: If True, keep CTT-SCHZ and CTT-MDD separate. If False, combine as CTT
        custom_colors: Optional dict with custom color mapping for diagnoses
        
    Example:
        # Run with CTT combined and default colors
        run_analysis_with_options(results_df, save_dir, True, "HC", split_ctt=False)
        
        # Run with CTT split and custom colors
        custom_colors = {
            "HC": "#2E8B57",      # Sea Green
            "SCHZ": "#DC143C",    # Crimson
            "MDD": "#4169E1",     # Royal Blue
            "CTT": "#FF8C00",     # Dark Orange
            "CTT-SCHZ": "#FF6347", # Tomato
            "CTT-MDD": "#FFD700"   # Gold
        }
        run_analysis_with_options(results_df, save_dir, True, "HC", 
                                split_ctt=True, custom_colors=custom_colors)
    """
    
    print(f"Running analysis with CTT {'split' if split_ctt else 'combined'}")
    if custom_colors:
        print(f"Using custom colors: {custom_colors}")
    
    # Run the main plotting function with new parameters
    summary_dict = plot_deviation_distributions(
        results_df=results_df,
        save_dir=save_dir,
        col_jitter=col_jitter,
        norm_diagnosis=norm_diagnosis,
        split_ctt=split_ctt,
        custom_colors=custom_colors,
        name=name
    )
    
    return summary_dict

def visualize_embeddings_multiple(normative_models, data_tensor, annotations_df, 
                                 columns_to_plot=None, device="cuda", figsize=(12, 10)):
    
    #visualizes the latent space and colores the data depending on given metadata -> X-Cov control 
    #returns dictionary with column names as keys and (figure, plot_df) tuples as values
    
    total_subjects = data_tensor.shape[0]
    if total_subjects != len(annotations_df):
        print(f"WARNING: Size mismatch detected: {total_subjects} samples in data tensor vs {len(annotations_df)} rows in annotations")
        print("Creating properly aligned dataset by extracting common subjects...")
        valid_indices = list(range(min(total_subjects, len(annotations_df))))
        aligned_annotations = annotations_df.iloc[valid_indices].reset_index(drop=True)
        annotations_df = aligned_annotations
        print(f"Aligned datasets - working with {len(annotations_df)} subjects")
    
    # Use first model for visualization
    model = normative_models[0]
    model.eval()
    model.to(device)
    
    all_embeddings = []
    batch_size = 16
    
    data_loader = DataLoader(
        TensorDataset(data_tensor),
        batch_size=batch_size,
        shuffle=False
    )
    
    print("Generating embeddings...")
    with torch.no_grad():
        for batch_data, in data_loader:
            batch_data = batch_data.to(device)
            _, mu, _ = model(batch_data)
            all_embeddings.append(mu.cpu().numpy())
    
    # Combine all embeddings
    embeddings = np.vstack(all_embeddings)
    
    # UMAP for visualization (only need to do this once)
    print("Computing UMAP projection...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)
    
    # Determine which columns in metadata df to plot
    if columns_to_plot is None:
        # Automatically detect categorical columns (excluding purely numerical ones)
        columns_to_plot = []
        for col in annotations_df.columns:
            if annotations_df[col].dtype == 'object' or annotations_df[col].nunique() <= 20:
                columns_to_plot.append(col)
        print(f"Auto-detected columns for visualization: {columns_to_plot}")
    
    # Create visualizations for each column
    results = {}
    
    for col in columns_to_plot:
        if col not in annotations_df.columns:
            print(f"Warning: Column '{col}' not found in annotations_df. Skipping.")
            continue
            
        print(f"Creating visualization for column: {col}")
        
        plot_df = annotations_df[[col]].copy()
        plot_df["umap_1"] = umap_embeddings[:, 0]
        plot_df["umap_2"] = umap_embeddings[:, 1]
        
        plot_df = plot_df.dropna(subset=[col])
        
        plt.figure(figsize=figsize)
        unique_values = plot_df[col].nunique()

        custom_palette = [
            "#125E8A",  # Lapis Lazuli
            "#3E885B",  # Sea Green  
            "#BEDCFE",  # Uranian Blue
            "#2F4B26",  # Cal Poly Green
            "#A67DB8",  # Indian Red 
            "#160C28"   # Dark Purple
        ]
        #continous vs binary color palettes depending on data
        if unique_values <= len(custom_palette):
            palette = custom_palette[:unique_values]  # schneidet auf die Anzahl an Klassen zu
        else:
            palette = sns.color_palette("viridis", n_colors=unique_values)
        
        if plot_df[col].dtype in ['object', 'category'] or unique_values <= 20:
            sns.scatterplot(
                data=plot_df,
                x="umap_1",
                y="umap_2",
                hue=col,
                palette=palette,
                s=40,
                alpha=0.7
            )
        else:
            scatter = plt.scatter(
                plot_df["umap_1"],
                plot_df["umap_2"],
                c=plot_df[col],
                cmap=palette,
                s=40,
                alpha=0.7
            )
            plt.colorbar(scatter, label=col)
        
        plt.title(f"UMAP Visualization - Colored by {col}", fontsize=16)
        plt.xlabel("UMAP 1", fontsize=13)
        plt.ylabel("UMAP 2", fontsize=13)
        
        if plt.gca().get_legend() is not None:
            plt.legend(title=col, fontsize=10, title_fontsize=11, 
                      bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        
        plt.tight_layout()
        
        results[col] = (plt.gcf(), plot_df.copy())
        plt.show()
    
    return results


def save_latent_visualizations(results, output_dir, dpi=300):
   
    os.makedirs(output_dir, exist_ok=True)
    
    for col_name, (fig, plot_df) in results.items():
        clean_name = col_name.replace(" ", "_").replace("/", "_").replace("\\", "_")

        fig.savefig(
            os.path.join(output_dir, f"umap_{clean_name}.png"),
            dpi=dpi,
            bbox_inches='tight',
            facecolor='white'
        )
        print(f"Saved visualization for '{col_name}'")


def extract_roi_names(h5_file_path, volume_type):
   
    #Extract ROI names from HDF5 file
    roi_names = []
    try:
        with h5py.File(h5_file_path, 'r') as f:
            # different options depending on if the files store ROI names as attributes or as dataset
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
            roi_names = [f"{atlas_name}_ROI_{i+1}" for i in range(100)]  
    return roi_names

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
def create_catatonia_subgroups(results_df, metadata_df, subgroup_columns, high_low_thresholds):
    #Create subgroups of Catatonia patients based on extended WHiteCAT & NSS metadata
    subgroups = {}
    
    # Get Catatonia patients
    ctt_patients = results_df[results_df["Diagnosis"].str.startswith("CTT")].copy()
    print(f"Found Catatonia diagnoses: {ctt_patients['Diagnosis'].unique()}")
        
    if len(ctt_patients) == 0:
        print("No CTT patients found for subgroup analysis")
        return subgroups
    
    # Merge with metadata
    if 'Filename' in ctt_patients.columns and 'Filename' in metadata_df.columns:
        ctt_with_metadata = ctt_patients.merge(metadata_df, on='Filename', how='left')
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
            print(f"Created CTT-high_{col} subgroup: n={len(high_group)}")
        
        if len(low_group) > 0:
            subgroups[f"CTT-low_{col}"] = low_group
            print(f"Created CTT-low_{col} subgroup: n={len(low_group)}")
    
    return subgroups

def analyze_regional_deviations(
        results_df, 
        save_dir, 
        clinical_data_path, 
        volume_type, 
        atlas_name, 
        roi_names, 
        norm_diagnosis,
        name, 
        add_catatonia_subgroups=True, 
        metadata_path=None, 
        subgroup_columns=None, 
        high_low_thresholds=None,
        merge_ctt_groups=True):  # NEW PARAMETER
    #Analyze regional deviations using ROI names 
    #regional effect sizes -> Cliff's Delta / Cohen's D
    
    # NEW: Merge CTT groups if requested
    if merge_ctt_groups:
        results_df = results_df.copy()
        results_df.loc[results_df['Diagnosis'].isin(['CTT-SCHZ', 'CTT-MDD']), 'Diagnosis'] = 'CTT'
        print("Merged CTT-SCHZ and CTT-MDD into single CTT group")
    
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

    effect_sizes = []
    
    # Create Catatonia subgroups if requested
    catatonia_subgroups = {}
    if add_catatonia_subgroups and metadata_path and subgroup_columns:
        try:
            metadata_df = pd.read_csv(metadata_path)
            # MODIFIED: Apply CTT merger to metadata as well
            if merge_ctt_groups:
                metadata_df = metadata_df.copy()
                metadata_df.loc[metadata_df['Diagnosis'].isin(['CTT-SCHZ', 'CTT-MDD']), 'Diagnosis'] = 'CTT'
            
            catatonia_subgroups = create_catatonia_subgroups(
                results_df, metadata_df, subgroup_columns, 
                high_low_thresholds
            )
        except Exception as e:
            print(f"Error loading metadata or creating subgroups: {e}")

    # Skip normative group as we use it for comparison
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

    # Skip normative group as we use it for comparison
    effect_sizes_df = pd.DataFrame(effect_sizes)
    
    if effect_sizes_df.empty:
        print("No effect sizes calculated. Returning empty DataFrame.")
        return effect_sizes_df
    
    
    effect_sizes_df["Abs_Cliffs_Delta"] = effect_sizes_df["Cliffs_Delta"].abs()
    effect_sizes_df["Abs_Cohens_d"] = effect_sizes_df["Cohens_d"].abs()
    os.makedirs(f"{save_dir}/figures", exist_ok=True)
    
    for diagnosis in diagnoses:
        if diagnosis == norm_diagnosis:
            continue
        
        dx_effect_sizes = effect_sizes_df[effect_sizes_df["Diagnosis"] == diagnosis].copy()
        if dx_effect_sizes.empty:
            continue
        
        # Sort by absolute effect size (Cliff's Delta)
        dx_effect_sizes_sorted = dx_effect_sizes.sort_values("Abs_Cliffs_Delta", ascending=False)
        
        # Take top 15-20 regions
        top_regions = dx_effect_sizes_sorted.head(16)  # Matching paper length
        
        # Create plot EXACTLY like in the paper - SCHMALER
        fig, ax = plt.subplots(figsize=(3, 6))  # Noch schmaler: von 4 auf 3
        
        # Create horizontal bar plot with confidence intervals
        y_pos = np.arange(len(top_regions))
        
        # Calculate confidence intervals for effect sizes (no bootstrap testing)
        # Standard error of the effect size (Cliff's Delta or Cohen's d)
        n1 = len(dx_data)  # diagnosis group size
        n2 = len(norm_data)  # control group size
        
        # For Cliff's Delta: approximate standard error
        se_cliffs = np.sqrt((n1 + n2 + 1) / (3 * n1 * n2)) * np.sqrt(top_regions["Abs_Cliffs_Delta"])
        
        # Alternative: Standard error for Cohen's d
        # se_cohens = np.sqrt((n1 + n2) / (n1 * n2) + top_regions["Cohens_d"]**2 / (2 * (n1 + n2)))
        
        ci_width = 1.96 * se_cliffs  # 95% confidence interval
        
        # Plot the confidence intervals as horizontal lines (like in paper)
        for i, (idx, row) in enumerate(top_regions.iterrows()):
            effect = row["Cliffs_Delta"]
            ci_low = effect - ci_width.iloc[i]
            ci_high = effect + ci_width.iloc[i]
            
            # Draw confidence interval line
            ax.plot([ci_low, ci_high], [i, i], 'k-', linewidth=1.5, alpha=0.8)
            
            # Draw the point estimate as a circle (like in paper)
            ax.plot(effect, i, 'ko', markersize=4, markerfacecolor='black', markeredgecolor='black')
        
        # Customize the plot to match paper EXACTLY
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_regions["ROI_Name"], fontsize=9)
        ax.invert_yaxis()  # Highest effect at top
        
        # Add vertical dashed line at x=0 (like in paper)
        ax.axvline(x=0, color="blue", linestyle="--", linewidth=1, alpha=0.7)
        
        # AUTOMATISCHE SKALIERUNG: Nur der benötigte Bereich
        # Berechne Min/Max der tatsächlichen Konfidenzintervalle
        all_ci_values = []
        for i, (idx, row) in enumerate(top_regions.iterrows()):
            effect = row["Cliffs_Delta"]
            ci_low = effect - ci_width.iloc[i]
            ci_high = effect + ci_width.iloc[i]
            all_ci_values.extend([ci_low, ci_high])
        
        min_value = min(all_ci_values)
        max_value = max(all_ci_values)
        
        # Kleine Puffer hinzufügen (5% des Bereichs)
        value_range = max_value - min_value
        buffer = value_range * 0.05
        
        ax.set_xlim(min_value - buffer, max_value + buffer)
        ax.set_xlabel("Effect size", fontsize=10)
        
        # Title exactly like paper format
        ax.set_title(f"Top 20 Regions {diagnosis} vs. {norm_diagnosis} \n ({atlas_name[0]} - {volume_type[0]})",fontsize=11, fontweight='bold', pad=10)
        
        # Keep ALL frame lines (like in paper) - don't remove any spines
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        
        # Keep all ticks
        ax.tick_params(axis='both', which='major', labelsize=9)
        
        # No grid (paper doesn't have grid)
        ax.grid(False)
        
        # Tight layout
        plt.tight_layout()
        
        # Save
        plt.savefig(f"{save_dir}/figures/paper_style_{diagnosis}_vs_{norm_diagnosis}.png", 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()


        
    custom_palette = [
        "#125E8A",  # Lapis Lazuli
        "#3E885B",  # Sea Green  
        "#BEDCFE",  # Uranian Blue
        "#2F4B26",  # Cal Poly Green
        "#A67DB8",  # Violet
        "#160C28"   # Dark Purple
    ]

    plt.figure(figsize=(10, 6))

    # Zähler für Farben
    color_idx = 0

    for diagnosis in diagnoses:
        if diagnosis == norm_diagnosis:
            continue

        dx_effect_sizes = effect_sizes_df[effect_sizes_df["Diagnosis"] == diagnosis]
        if not dx_effect_sizes.empty:
            color = custom_palette[color_idx % len(custom_palette)]
            sns.kdeplot(dx_effect_sizes["Cliffs_Delta"], label=diagnosis, color=color)
            color_idx += 1

    plt.axvline(x=0, color="black", linestyle="--", alpha=0.7)
    plt.title(f"Distribution of Regional Effect Sizes vs {norm_diagnosis} \n {name}")
    plt.xlabel("Cliff's Delta")
    plt.legend()
    plt.tight_layout()  
    plt.savefig(f"{save_dir}/figures/effect_size_distributions_vs_{norm_diagnosis}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
   
# 1. First calculate CTT-based top regions
    if merge_ctt_groups:
        ctt_effects = effect_sizes_df[effect_sizes_df["Diagnosis"] == "CTT"]
    else:
        ctt_effects = effect_sizes_df[effect_sizes_df["Diagnosis"].isin(["CTT-SCHZ", "CTT-MDD"])]

    if not ctt_effects.empty:
        if merge_ctt_groups:
            ctt_top_regions = ctt_effects.sort_values("Abs_Cliffs_Delta", ascending=False).head(30)["ROI_Name"].values
            print(f"Selected top 30 regions based on CTT effect sizes (n={len(ctt_effects)} regions)")
        else:
            ctt_region_avg = ctt_effects.groupby("ROI_Name")["Abs_Cliffs_Delta"].mean().reset_index()
            ctt_top_regions = ctt_region_avg.sort_values("Abs_Cliffs_Delta", ascending=False).head(30)["ROI_Name"].values
            print(f"Selected top 30 regions based on average CTT-SCHZ and CTT-MDD effect sizes")
        
        print(f"Top 30 CTT-affected regions: {ctt_top_regions[:10]}...")
    else:
        print("Warning: No CTT data found. Using empty list for CTT top regions.")
        ctt_top_regions = []

    # 2. Calculate overall averaged top regions (original method)
    region_avg_effects = effect_sizes_df.groupby("ROI_Name")["Abs_Cliffs_Delta"].mean().reset_index()
    overall_top_regions = region_avg_effects.sort_values("Abs_Cliffs_Delta", ascending=False).head(30)["ROI_Name"].values
    print(f"Top 30 overall averaged regions: {overall_top_regions[:10]}...")

    # Create all heatmap data first
    heatmap_data = []
    all_diagnoses = effect_sizes_df["Diagnosis"].unique()

    # Create data for all regions (we'll filter later)
    all_regions = effect_sizes_df["ROI_Name"].unique()
    for region in all_regions:
        row = {"ROI_Name": region}
        for diagnosis in all_diagnoses:
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
    heatmap_df = heatmap_df.dropna(axis=1, how='all')

    # HEATMAP 1: 3 Diagnoses with CTT Top 30 regions
    print("\n=== Creating Heatmap 1: 3 Diagnoses with CTT Top 30 ===")
    if merge_ctt_groups:
        desired_diagnoses = ['MDD', 'SCHZ', 'CTT']
    else:
        desired_diagnoses = ['MDD', 'SCHZ', 'CTT-SCHZ', 'CTT-MDD']

    available_diagnoses = [diag for diag in desired_diagnoses if diag in heatmap_df.columns]

    if len(available_diagnoses) > 0 and len(ctt_top_regions) > 0:
        # Filter by CTT top regions
        heatmap_ctt_regions = heatmap_df.loc[heatmap_df.index.isin(ctt_top_regions), available_diagnoses]
        
        if not heatmap_ctt_regions.empty and not heatmap_ctt_regions.isna().all().all():
            fig_width = max(12, len(available_diagnoses) * 3)
            plt.figure(figsize=(fig_width, 16))
            
            mask = heatmap_ctt_regions.isna()
            sns.heatmap(heatmap_ctt_regions, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                    cbar_kws={"label": "Cliff's Delta"}, mask=mask,
                    square=False, linewidths=0.5)
            
            if merge_ctt_groups:
                plt.title(f"Top 30 CTT-Affected Regions vs {norm_diagnosis}\n {name}")
            else:
                plt.title(f"Top 30 CTT-Affected Regions vs {norm_diagnosis}\n {name}")
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{save_dir}/figures/heatmap_1_ctt_regions_3diagnoses_vs_{norm_diagnosis}.png", 
                    dpi=300, bbox_inches='tight')
            plt.close()
            
            heatmap_ctt_regions.to_csv(f"{save_dir}/heatmap_1_ctt_regions_3diagnoses_vs_{norm_diagnosis}.csv")
            print(f"Heatmap 1 created: {heatmap_ctt_regions.shape[0]} regions, {len(available_diagnoses)} diagnoses")
        else:
            print("No data available for Heatmap 1")
    else:
        print("Cannot create Heatmap 1 - missing diagnoses or CTT regions")

    # HEATMAP 2: 3 Diagnoses with Overall Top 30 regions
    print("\n=== Creating Heatmap 2: 3 Diagnoses with Overall Top 30 ===")
    if len(available_diagnoses) > 0:
        # Filter by overall top regions
        heatmap_overall_regions = heatmap_df.loc[heatmap_df.index.isin(overall_top_regions), available_diagnoses]
        
        if not heatmap_overall_regions.empty and not heatmap_overall_regions.isna().all().all():
            fig_width = max(12, len(available_diagnoses) * 3)
            plt.figure(figsize=(fig_width, 16))
            
            mask = heatmap_overall_regions.isna()
            sns.heatmap(heatmap_overall_regions, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                    cbar_kws={"label": "Cliff's Delta"}, mask=mask,
                    square=False, linewidths=0.5)
            
            if merge_ctt_groups:
                plt.title(f"Top 30 overall-Affected Regions vs {norm_diagnosis}\n {name}")
            else:
                plt.title(f"Top 30 overall-Affected Regions vs {norm_diagnosis}\n {name}")
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{save_dir}/figures/heatmap_2_overall_regions_3diagnoses_vs_{norm_diagnosis}.png", 
                    dpi=300, bbox_inches='tight')
            plt.close()
            
            heatmap_overall_regions.to_csv(f"{save_dir}/heatmap_2_overall_regions_3diagnoses_vs_{norm_diagnosis}.csv")
            print(f"Heatmap 2 created: {heatmap_overall_regions.shape[0]} regions, {len(available_diagnoses)} diagnoses")
        else:
            print("No data available for Heatmap 2")
    else:
        print("Cannot create Heatmap 2 - missing diagnoses")

    # HEATMAP 3: CTT Subgroups with CTT Top 30 regions (excluding SCHZ and MDD)
    print("\n=== Creating Heatmap 3: CTT Subgroups with CTT Top 30 ===")
    if len(ctt_top_regions) > 0:
        # Filter by CTT top regions first
        heatmap_subgroups = heatmap_df.loc[heatmap_df.index.isin(ctt_top_regions)]
        
        # Remove SCHZ and MDD columns
        columns_to_exclude = ['SCHZ', 'MDD']
        remaining_columns = [col for col in heatmap_subgroups.columns if col not in columns_to_exclude]
        
        if len(remaining_columns) > 0:
            heatmap_subgroups = heatmap_subgroups[remaining_columns]
            
            if not heatmap_subgroups.empty and not heatmap_subgroups.isna().all().all():
                fig_width = max(16, len(remaining_columns) * 1.5)
                plt.figure(figsize=(fig_width, 16))
                
                mask = heatmap_subgroups.isna()
                sns.heatmap(heatmap_subgroups, cmap="RdBu_r", center=0, annot=True, fmt=".2f", 
                        cbar_kws={"label": "Cliff's Delta"}, mask=mask,
                        linewidths=0.5)
                
                plt.title(f"Top Regions (CTT) vs {norm_diagnosis}\n{name}")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(f"{save_dir}/figures/heatmap_3_ctt_subgroups_vs_{norm_diagnosis}.png", 
                        dpi=300, bbox_inches='tight')
                plt.close()
                
                heatmap_subgroups.to_csv(f"{save_dir}/heatmap_3_ctt_subgroups_vs_{norm_diagnosis}.csv")
                print(f"Heatmap 3 created: {heatmap_subgroups.shape[0]} regions, {len(remaining_columns)} subgroups")
                print(f"Subgroups included: {remaining_columns}")
            else:
                print("No data available for Heatmap 3")
    
    print(f"\n=== Summary ===")
    print(f"Heatmap 1: 3 main diagnoses with top 30 CTT-affected regions")
    print(f"Heatmap 2: 3 main diagnoses with top 30 overall-affected regions") 
    print(f"Heatmap 3: CTT subgroups only with top 30 CTT-affected regions")

    if len(ctt_top_regions) > 0:
        top_regions_overall = ctt_top_regions
        print(f"Using CTT top regions for Dataset-Split Heatmap")
    else:
        top_regions_overall = overall_top_regions
        print(f"Using overall top regions for Dataset-Split Heatmap")

    #---------------------Dataset-Split Heatmap (Modified Version)-----------------

    print("Creating dataset-split heatmap...")

    # Check if Dataset column exists
    if 'Dataset' not in results_df.columns:
        print("No 'Dataset' column found - skipping dataset-split heatmap")
    else:
        # Define dataset categories (flexible for different spellings)
        available_datasets = results_df['Dataset'].unique()
        print(f"Available datasets: {available_datasets}")
        
        dataset_categories = {
            'whiteCAT': [d for d in available_datasets if 'whitecat' in str(d).lower() or 'white_cat' in str(d).lower()],
            'NSS': [d for d in available_datasets if 'nss' in str(d).lower()],
            'others': [d for d in available_datasets if not any(x in str(d).lower() for x in ['whitecat', 'white_cat', 'nss'])]
        }
        
        print(f"Dataset categories: {dataset_categories}")
        
        # Calculate effect sizes for dataset splits
        dataset_split_effects = []
        
        # MODIFIED: Use merged diagnosis groups
        if merge_ctt_groups:
            main_diagnoses = ['SCHZ', 'MDD', 'CTT']
        else:
            main_diagnoses = ['SCHZ', 'MDD', 'CTT-SCHZ', 'CTT-MDD']
        
        for diagnosis in main_diagnoses:
            if diagnosis == norm_diagnosis:
                continue
                
            # All patients of this diagnosis
            dx_data = results_df[results_df["Diagnosis"] == diagnosis]
            if dx_data.empty:
                print(f"No data found for {diagnosis}")
                continue
                
            print(f"Processing {diagnosis} (total n={len(dx_data)})")
            
            # For each dataset category
            for category_name, dataset_list in dataset_categories.items():
                if not dataset_list:
                    continue
                    
                # Filter data for this category
                category_data = dx_data[dx_data['Dataset'].isin(dataset_list)]
                
                if category_data.empty:
                    print(f"  No {category_name} data for {diagnosis}")
                    continue
                    
                print(f"  {diagnosis}-{category_name}: n={len(category_data)}")
                
                # Calculate effect sizes for all regions
                for i, region_col in enumerate(region_cols):
                    roi_name = roi_names[i] if i < len(roi_names) else f"Region_{i+1}"
                    
                    category_values = category_data[region_col].values
                    norm_values = norm_data[region_col].values
                    
                    if len(category_values) == 0 or len(norm_values) == 0:
                        continue
                    
                    # Calculate Cliff's Delta
                    cliff_delta = calculate_cliffs_delta(category_values, norm_values)
                    
                    dataset_split_effects.append({
                        'Diagnosis_Dataset': f"{diagnosis}-{category_name}",
                        'Diagnosis': diagnosis,
                        'Dataset_Category': category_name,
                        'ROI_Name': roi_name,
                        'Cliffs_Delta': cliff_delta,
                        'N_Subjects': len(category_values)
                    })
        
        if dataset_split_effects:
            # Create DataFrame
            dataset_effects_df = pd.DataFrame(dataset_split_effects)
            
            # Create pivot table for heatmap
            heatmap_dataset = dataset_effects_df.pivot(
                index='ROI_Name', 
                columns='Diagnosis_Dataset', 
                values='Cliffs_Delta'
            )
            
            # Use only top regions
            heatmap_dataset_top = heatmap_dataset[heatmap_dataset.index.isin(top_regions_overall)]
            
            # Sort columns in desired order
            column_order = []
            for diagnosis in main_diagnoses:
                if diagnosis == norm_diagnosis:
                    continue
                for category in ['whiteCAT', 'NSS', 'others']:
                    col_name = f"{diagnosis}-{category}"
                    if col_name in heatmap_dataset_top.columns:
                        column_order.append(col_name)
            
            heatmap_dataset_ordered = heatmap_dataset_top[column_order]
            
            # Create heatmap
            if not heatmap_dataset_ordered.empty and len(heatmap_dataset_ordered.columns) > 0:
                fig_width = max(16, len(heatmap_dataset_ordered.columns) * 2.5)
                fig_height = max(14, len(heatmap_dataset_ordered) * 0.4)
                
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                
                # NaN mask
                mask = heatmap_dataset_ordered.isna()
                
                # Plot heatmap
                sns.heatmap(heatmap_dataset_ordered, 
                        cmap="RdBu_r", 
                        center=0, 
                        annot=True, 
                        fmt=".2f", 
                        mask=mask,
                        cbar_kws={"label": "Cliff's Delta"}, 
                        linewidths=0.5,
                        ax=ax)
                
                # Title and labels
                ax.set_title(f"Regional Effect Sizes vs {norm_diagnosis}\n(Split by Dataset: whiteCAT, NSS, others) \n {name}", 
                            fontsize=16, pad=20)
                ax.set_xlabel("Diagnosis-Dataset", fontsize=12)
                ax.set_ylabel("Brain Region", fontsize=12)
                
                # Rotate x-axis labels
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                
                # Add vertical lines between diagnoses
                col_positions = []
                current_pos = 0
                diagnosis_positions = {}
                
                for diagnosis in main_diagnoses:
                    if diagnosis == norm_diagnosis:
                        continue
                        
                    diagnosis_cols = [col for col in column_order if col.startswith(f"{diagnosis}-")]
                    if diagnosis_cols:
                        diagnosis_positions[diagnosis] = current_pos + len(diagnosis_cols)/2
                        current_pos += len(diagnosis_cols)
                        col_positions.append(current_pos)
                
                # Draw separator lines (except after last group)
                for pos in col_positions[:-1]:
                    ax.axvline(x=pos, color='black', linewidth=2, alpha=0.8)
                
                plt.tight_layout()
                
                # Save
                plt.savefig(f"{save_dir}/figures/region_effect_heatmap_dataset_split_vs_{norm_diagnosis}.png",
                        dpi=300, bbox_inches='tight')
                plt.close()
                
                # Save data
                heatmap_dataset_ordered.to_csv(f"{save_dir}/top_regions_heatmap_dataset_split_vs_{norm_diagnosis}.csv")
                
                # Print summary
                print(f"\nDataset-split heatmap created successfully!")
                print(f"Shape: {heatmap_dataset_ordered.shape}")
                print(f"Columns: {list(heatmap_dataset_ordered.columns)}")
                
                # Data overview per column
                print("\nData availability per column:")
                for col in heatmap_dataset_ordered.columns:
                    non_nan = heatmap_dataset_ordered[col].notna().sum()
                    total = len(heatmap_dataset_ordered)
                    pct = (non_nan/total*100) if total > 0 else 0
                    print(f"  {col}: {non_nan}/{total} regions ({pct:.1f}%)")
                    
                # Sample size overview
                print("\nSample sizes per group:")
                sample_sizes = dataset_effects_df.groupby('Diagnosis_Dataset')['N_Subjects'].first()
                for group, n in sample_sizes.items():
                    print(f"  {group}: n={n}")
                    
            else:
                print("No data available for dataset-split heatmap")
                
        else:
            print("No effect sizes calculated for dataset splits")

        # Save data
        effect_sizes_df.to_csv(f"{save_dir}/effect_sizes_{norm_diagnosis}.csv")
            
        print(f"\nRegional analysis completed. Results saved to {save_dir}")
        print(f"Total effect sizes calculated: {len(effect_sizes_df)}")
        print(f"Average absolute Cliff's Delta: {effect_sizes_df['Abs_Cliffs_Delta'].mean():.3f}")
        print(f"Max absolute Cliff's Delta: {effect_sizes_df['Abs_Cliffs_Delta'].max():.3f}")
        
        if catatonia_subgroups:
            print(f"\nCatatonia subgroups created: {list(catatonia_subgroups.keys())}")
        
    return effect_sizes_df

######################################################## CORRELATION ANALYSIS ################################################################


def create_corrected_correlation_heatmap(results_df, metadata_df, save_dir, name,
                                       correction_method='fdr_bh',
                                       alpha=0.05,
                                       merge_ctt_groups=True):
    """
    Erstellt eine Heatmap mit korrigierten Korrelationen zwischen Deviation Scores 
    der Patientengruppen und klinischen Scores
    
    Parameters:
    -----------
    results_df : DataFrame
        DataFrame mit Deviation Scores und Diagnosen
    metadata_df : path 
    save_dir : str
        Pfad zum Speichern der Heatmap
    correction_method : str
        'bonferroni', 'fdr_bh', 'fdr_by', 'holm'
    alpha : float
        Signifikanzlevel (default: 0.05)
    merge_ctt_groups : bool
        CTT-SCHZ und CTT-MDD zu CTT zusammenfassen (default: True)
    """
    metadata_df = pd.read_csv(metadata_df)
    # CTT Gruppen zusammenfassen falls gewünscht
    if merge_ctt_groups:
        results_df = results_df.copy()
        metadata_df = metadata_df.copy()
        results_df.loc[results_df['Diagnosis'].isin(['CTT-SCHZ', 'CTT-MDD']), 'Diagnosis'] = 'CTT'
        metadata_df.loc[metadata_df['Diagnosis'].isin(['CTT-SCHZ', 'CTT-MDD']), 'Diagnosis'] = 'CTT'
        print("CTT-SCHZ und CTT-MDD zu CTT zusammengefasst")
    
    # Merge der DataFrames
    merged_data = pd.merge(results_df, metadata_df, on='Filename', how='inner')
    merged_data = merged_data.rename(columns={'Age_x': 'Age', 'Sex_x': 'Sex', 'Dataset_x': 'Dataset'})
    
    # Nur Patientengruppen (keine HC)
    patient_data = merged_data[merged_data['Diagnosis_x'] != 'HC']
    
    # Definiere Score-Spalten
    score_columns = ['GAF_Score', 'PANSS_Positive', 'PANSS_Negative', 
                     'PANSS_General', 'PANSS_Total', 'BPRS_Total', 
                     'NCRS_Motor', 'NCRS_Affective', 'NCRS_Behavioral', 
                     'NCRS_Total', 'NSS_Motor', 'NSS_Total']
    
    # Filtere verfügbare Scores
    available_scores = [col for col in score_columns if col in patient_data.columns]
    
    # Patientengruppen identifizieren
    diagnoses = [d for d in patient_data['Diagnosis_x'].unique() if d != 'HC']
    
    print(f"Analysiere Korrelationen für {len(diagnoses)} Patientengruppen und {len(available_scores)} Scores")
    print(f"Patientengruppen: {diagnoses}")
    print(f"Verfügbare Scores: {available_scores}")
    
    # Korrelationen berechnen
    correlation_matrix = np.full((len(diagnoses), len(available_scores)), np.nan)
    p_value_matrix = np.full((len(diagnoses), len(available_scores)), np.nan)
    
    all_p_values = []
    correlation_info = []
    
    for i, diagnosis in enumerate(diagnoses):
        diag_data = patient_data[patient_data['Diagnosis_x'] == diagnosis]
        
        for j, score in enumerate(available_scores):
            valid_data = diag_data[['deviation_score', score]].dropna()
            
            if len(valid_data) >= 3:  # Mindestens 3 Datenpunkte
                r, p = pearsonr(valid_data['deviation_score'], valid_data[score])
                correlation_matrix[i, j] = r
                p_value_matrix[i, j] = p
                all_p_values.append(p)
                correlation_info.append((i, j, diagnosis, score, len(valid_data), r, p))
    
    # Multiple Testing Correction
    if len(all_p_values) > 0:
        rejected, corrected_p_values, _, _ = multipletests(
            all_p_values, alpha=alpha, method=correction_method
        )
        
        # Korrigierte p-Werte in Matrix einsetzen
        corrected_p_matrix = np.full((len(diagnoses), len(available_scores)), np.nan)
        significance_matrix = np.full((len(diagnoses), len(available_scores)), False)
        
        for idx, (i, j, diagnosis, score, n, r, p) in enumerate(correlation_info):
            corrected_p_matrix[i, j] = corrected_p_values[idx]
            significance_matrix[i, j] = rejected[idx]
    
    # Annotationen erstellen
    annotations = []
    for i in range(len(diagnoses)):
        row_annotations = []
        for j in range(len(available_scores)):
            if np.isnan(correlation_matrix[i, j]):
                row_annotations.append('')
            else:
                r_val = correlation_matrix[i, j]
                p_val = corrected_p_matrix[i, j]
                
                # Signifikanz-Sterne basierend auf korrigierten p-Werten
                if p_val < 0.001:
                    stars = '***'
                elif p_val < 0.01:
                    stars = '**'
                elif p_val < alpha:
                    stars = '*'
                else:
                    stars = ''
                
                annotation = f'{r_val:.2f}{stars}'
                row_annotations.append(annotation)
        annotations.append(row_annotations)
    
    # Heatmap erstellen
    plt.figure(figsize=(16, max(6, len(diagnoses) * 0.8)))
    
    mask = np.isnan(correlation_matrix)
    
    sns.heatmap(correlation_matrix,
                xticklabels=available_scores,
                yticklabels=diagnoses,
                annot=annotations,
                fmt='',
                cmap='RdBu_r',
                center=0,
                mask=mask,
                square=False,
                cbar_kws={'label': 'Pearson Correlation Coefficient'},
                linewidths=0.5,
                linecolor='white')
    
    plt.title(f'Deviation Score Correlations - {name}\n'
              f'({correction_method.upper()} Corrected, α={alpha})\n'
              f'(* p<{alpha}, ** p<0.01, *** p<0.001)', 
              fontsize=14, pad=20)
    plt.xlabel('Clinical Scores', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Speichern
    import os
    os.makedirs(f"{save_dir}/figures", exist_ok=True)
    filename = f"{save_dir}/figures/patient_correlations_{correction_method}_corrected.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Zusammenfassung ausgeben
    total_tests = len([p for p in all_p_values if not np.isnan(p)])
    significant_corrected = np.sum(significance_matrix)
    
    print(f"\n=== ZUSAMMENFASSUNG ===")
    print(f"Gesamte Tests: {total_tests}")
    print(f"Signifikante Korrelationen (korrigiert): {significant_corrected}")
    print(f"Korrekturmethode: {correction_method}")
    print(f"Alpha-Level: {alpha}")
    print(f"Heatmap gespeichert: {filename}")
    
    # Detaillierte Ergebnisse für signifikante Korrelationen
    print(f"\n=== SIGNIFIKANTE KORRELATIONEN ===")
    for idx, (i, j, diagnosis, score, n, r, p_orig) in enumerate(correlation_info):
        if significance_matrix[i, j]:
            p_corr = corrected_p_matrix[i, j]
            print(f"{diagnosis} - {score}: r={r:.3f}, p_orig={p_orig:.3f}, p_corr={p_corr:.3f}, n={n}")
    
    return correlation_matrix, corrected_p_matrix, significance_matrix
