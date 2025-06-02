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
        if diag_data[color_col].dtype == 'object' or color_col in ['Sex', 'Co_Diagnosis']:
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
                               save_dir, color_columns, diagnosis_palette):
    #Create jitter plots colored by numerical values from specified columns
    #data: results dataset containing the metric and diagnosis information
    #metadata_df: Additional dataframe containing metadata columns for coloring (scores etc)
    
    os.makedirs(f"{save_dir}/figures/distributions/colored_by_columns", exist_ok=True)
    
    # Check if we can merge on filename or need to use index
    merged_data = pd.merge(data, metadata_df, on='Filename', how='inner')
    print(f"Merged data on 'Filename' column. Merged data shape: {merged_data.shape}")
    
    if merged_data.empty:
        print("Error: Could not merge data and metadata. Check if they have common identifiers.")
        return
    #changed column names after merging
    merged_data = merged_data.rename(columns={'Age_x': 'Age', 'Sex_x': 'Sex'})
    
    # Filter color_columns to only include ones that exist in merged_data
    available_color_columns = [col for col in color_columns if col in merged_data.columns]
    
    # Define columns that have complete data (all patients) for all diagnoses vs. limited diagnoses (WhiteCAT & NSS metadata)
    complete_data_columns = ['Age', 'Sex']  # Assuming these have data for all diagnoses
    limited_data_columns = [col for col in available_color_columns if col not in complete_data_columns]
    
    for color_col in available_color_columns:
        print(f"Creating plot for column: {color_col}")
        
        
        if color_col in complete_data_columns:
            # Use all diagnoses for Age and Sex -> got metadata for all
            current_plot_order = plot_order
            filtered_data = merged_data.copy()
            plot_title_suffix = "All Diagnoses"
        else:
            # Use only CTT-SCHZ and CTT-MDD for other columns -> got metadata only for WhiteCAT and NSS patients
            current_plot_order = ['CTT-SCHZ', 'CTT-MDD']
            filtered_data = merged_data[merged_data['Diagnosis_x'].isin(current_plot_order)].copy()
            plot_title_suffix = "CTT-SCHZ vs CTT-MDD"
        
        filtered_data = filtered_data.dropna(subset=[color_col, metric])
        
        if len(filtered_data) == 0:
            print(f"Warning: No data available for {color_col} after removing missing values. Skipping this column.")
            continue
        
        
        plt.figure(figsize=(14, 6))
        color_values = filtered_data[color_col].copy()
        # Handle categorical variables by converting to numeric
        if color_values.dtype == 'object' or color_col in ['Sex', 'Co_Diagnosis']:
            unique_values = color_values.unique()
            value_to_code = {val: i for i, val in enumerate(unique_values)}
            color_values_numeric = color_values.map(value_to_code)
            if color_col == 'Sex':
                colors = ['#ff69b4', '#4169e1'] 
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
        
        # Add errorbars and statistics
        current_summary = summary_df[summary_df["Diagnosis"].isin(current_plot_order)]
        
        for i, diagnosis in enumerate(current_plot_order):
            diagnosis_data = current_summary[current_summary["Diagnosis"] == diagnosis]
            if len(diagnosis_data) > 0:
                mean_val = diagnosis_data["mean"].iloc[0]
                ci_val = diagnosis_data["ci95"].iloc[0]
                
                # Add errorbar
                plt.errorbar(mean_val, i, xerr=ci_val, fmt='none',
                           color='black', capsize=4, capthick=2,
                           elinewidth=2, alpha=0.9, zorder=10)
                
                # Add mean value as a larger marker
                plt.scatter(mean_val, i, color='black', s=100, 
                          marker='D', alpha=0.9, zorder=11, 
                          edgecolors='white', linewidth=1)
        
    
        plt.yticks(range(len(current_plot_order)), current_plot_order)
        plt.title(f"{metric.replace('_', ' ').title()} by Diagnosis\nColored by {color_col.replace('_', ' ').title()} ({plot_title_suffix})", 
                 fontsize=14, pad=20)
        plt.xlabel(f"{metric.replace('_', ' ').title()}", fontsize=12)
        plt.ylabel("Diagnosis", fontsize=12)
        plt.grid(True, alpha=0.3, axis='x')
        plt.gca().invert_yaxis()
        sns.despine()
        plt.tight_layout()
        
        filename = f"{metric}_jitterplot_colored_by_{color_col}.png"
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


def calculate_group_pvalues(results_df, norm_diagnosis):
    #Calculate p-values for each diagnosis group compared to the control group

    # Get control group data
    control_mask = results_df["Diagnosis"] == norm_diagnosis
    if not control_mask.any():
        print(f"WARNING: No control group '{norm_diagnosis}' found in data. Available diagnoses: {results_df['Diagnosis'].unique()}")
        # Use bottom 25% as reference if no explicit control group
        control_indices = np.argsort(results_df["deviation_score_zscore"])[:len(results_df)//4]
        control_mask = np.zeros(len(results_df), dtype=bool)
        control_mask[control_indices] = True
        print(f"Using bottom 25% ({control_mask.sum()} subjects) as reference group")
    
    control_data = results_df[control_mask]
    print(f"Control group ({norm_diagnosis}) size: {len(control_data)}")
    
    # Metrics to test
    metrics = ["reconstruction_error", "kl_divergence", "deviation_score"]
    
    # Calculate p-values for each diagnosis group vs control
    group_pvalues = {}
    
    diagnoses = results_df["Diagnosis"].unique()
    diagnoses = [d for d in diagnoses if d != norm_diagnosis]  # Exclude control group
    
    for metric in metrics:
        group_pvalues[metric] = {}
        control_values = control_data[metric].values
        
        for diagnosis in diagnoses:
            group_data = results_df[results_df["Diagnosis"] == diagnosis]
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
    
    return group_pvalues

def plot_deviation_distributions(results_df, save_dir, col_jitter, norm_diagnosis):
    #Plot distributions of deviation metrics by diagnosis group with group p-values

    os.makedirs(f"{save_dir}/figures/distributions", exist_ok=True)
    
    # Create color palette
    palette = sns.light_palette("blue", n_colors=6, reverse=True)
    diagnosis_order = ["HC", "SCHZ", "MDD", "CTT", "CTT-MDD", "CTT-SCHZ"]
    diagnosis_palette = dict(zip(diagnosis_order, palette))

    # Calculate group p-values
    group_pvalues = calculate_group_pvalues(results_df, norm_diagnosis)

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

    selected_diagnoses = ["HC", "SCHZ", "MDD", "CTT", "CTT-MDD", "CTT-SCHZ"]

    # Calculate summary statistics for errorbar plots
    metrics = ["reconstruction_error", "kl_divergence", "deviation_score"]
    summary_dict = {}

    for metric in metrics:
        # Filter data for selected diagnoses
        filtered_data = results_df[results_df["Diagnosis"].isin(selected_diagnoses)]
        
        summary_df = (
            filtered_data
            .groupby("Diagnosis")[metric]
            .agg(['mean', 'std', 'count'])
            .reset_index()
        )
        
        # Calculate 95% confidence interval
        summary_df["ci95"] = 1.96 * summary_df["std"] / np.sqrt(summary_df["count"])
        
        # Add group p-values
        summary_df["p_value"] = summary_df["Diagnosis"].map(
            lambda d: group_pvalues[metric].get(d, np.nan) if d != norm_diagnosis else np.nan
        )
        ###########################################################################################################################
        # Sort in desired order (for display from bottom to top)
        diagnosis_order_plot = selected_diagnoses[::-1]
        summary_df["Diagnosis"] = pd.Categorical(summary_df["Diagnosis"], categories=diagnosis_order_plot, ordered=True)
        summary_df = summary_df.sort_values("Diagnosis")
        
        summary_dict[metric] = summary_df
        
        # Create ORIGINAL ERRORBAR PLOT (from first code) - without jitter
        plt.figure(figsize=(8, 6))
        
        # Filter only diagnoses that actually have data
        available_diagnoses = filtered_data["Diagnosis"].unique()
        plot_order = [d for d in diagnosis_order_plot if d in available_diagnoses]
        
        # Original errorbar plot
        plt.errorbar(summary_df["mean"], summary_df["Diagnosis"], 
                    xerr=summary_df["ci95"],
                    fmt='s', color='black', capsize=5, markersize=8)
        
        # Add mean p-value as color coding (like in original)
        summary_df_plot = summary_df[summary_df["Diagnosis"].isin(plot_order)]
        # Use group p-values instead of mean p-values for coloring
        p_values_for_color = summary_df_plot["p_value"].fillna(0.5)  # Fill NaN with neutral value
        scatter = plt.scatter(summary_df_plot["mean"], summary_df_plot["Diagnosis"], 
                            c=p_values_for_color, cmap='RdYlBu_r', 
                            s=100, alpha=0.7, edgecolors='black')
        
        plt.title(f"{metric.replace('_', ' ').title()}", fontsize=14)
        plt.xlabel("Deviation Metric", fontsize=12)
        plt.ylabel("Diagnosis", fontsize=12)
        
        sns.despine()
        plt.tight_layout()
        
        # Save the original errorbar plot
        plt.savefig(f"{save_dir}/figures/distributions/{metric}_errorbar.png", dpi=300)
        plt.close()
        
        # Create jitterplot with p-values and mean values (current version)
        plt.figure(figsize=(12, 6))  # Made wider to accommodate value labels
        
        # Get MDD color from the palette
        mdd_color = diagnosis_palette.get('MDD', '#4c72b0')  # fallback to blue if MDD not found
        
        # Jitterplot with uniform color (MDD color) and smaller points
        sns.stripplot(data=filtered_data, y="Diagnosis", x=metric, 
                    order=plot_order, color=mdd_color, 
                    size=3, alpha=0.6, jitter=0.3)
        
        # Add errorbars, p-values, and mean values
        for i, diagnosis in enumerate(plot_order):
            diagnosis_data = summary_df[summary_df["Diagnosis"] == diagnosis]
            if len(diagnosis_data) > 0:  # Check if data is available
                mean_val = diagnosis_data["mean"].iloc[0]
                ci_val = diagnosis_data["ci95"].iloc[0]
                p_val = diagnosis_data["p_value"].iloc[0]
                n_val = diagnosis_data["count"].iloc[0]
                
                # Add errorbar
                plt.errorbar(mean_val, i, xerr=ci_val, fmt='none', 
                            color='black', capsize=4, capthick=1.5, 
                            elinewidth=1.5, alpha=0.8)
            
                # Add p-value text to the right of errorbar (only if not control group and p-value exists)
                # if diagnosis != norm_diagnosis and not np.isnan(p_val):
                #     # Format p-value
                #     if p_val < 0.001:
                #         p_text = "p<0.001***"
                #     elif p_val < 0.01:
                #         p_text = f"p<0.01**"
                #     elif p_val < 0.05:
                #         p_text = f"p={p_val:.3f}*"
                #     else:
                #         p_text = f"p={p_val:.3f}"
                    
                #     # Position text to the right of the errorbar
                #     text_x = mean_val + ci_val + (plt.xlim()[1] - plt.xlim()[0]) * 0.02
                #     plt.text(text_x, i, p_text, va='center', ha='left', 
                #             fontsize=9, fontweight='bold' if p_val < 0.05 else 'normal',
                #             color='red' if p_val < 0.05 else 'black')
        
        plt.title(f"{metric.replace('_', ' ').title()} by Diagnosis (vs {norm_diagnosis})", fontsize=14)
        plt.xlabel("Deviation Metric", fontsize=12)
        plt.ylabel("Diagnosis", fontsize=12)
        
        # Adjust layout to accommodate value labels
        plt.subplots_adjust(left=0.25)  # Make room for value labels on the left
        
        sns.despine()
        plt.tight_layout()
        
        # Save the jitterplot
        plt.savefig(f"{save_dir}/figures/distributions/{metric}_jitterplot_with_values.png", dpi=300, bbox_inches='tight')
        plt.close()

        if col_jitter: 
            metadata_df = pd.read_csv('/workspace/project/catatonia_VAE-main_bq/metadata_20250110/full_data_with_codiagnosis_and_scores.csv')  # Replace with your actual path
            # Define columns to use for coloring from the metadata dataframe
            # Based on your image, these should be the column names in your metadata
            potential_color_columns = ['Age', 'Sex',
                                'GAF_Score', 'PANSS_Positive', 'PANSS_Negative', 
                                    'PANSS_General', 'PANSS_Total', 'BPRS_Total', 'NCRS_Motor', 
                                    'NCRS_Affective', 'NCRS_Behavioral', 'NCRS_Total', 'NSS_Motor', 'NSS_Total']

            # Filter to only include columns that actually exist in your metadata dataframe
            color_columns = [col for col in potential_color_columns if col in metadata_df.columns]
            print(f"Found {len(color_columns)} columns for coloring: {color_columns}")

            if len(color_columns) == 0:
                print("No color columns found! Please check your column names in the metadata dataframe.")
            else:
                # Create colored jitter plots with metadata
                create_colored_jitter_plots(
                    data=filtered_data,  # your main dataset with deviation scores
                    metadata_df=metadata_df,  # your metadata dataframe
                    metric=metric,  # your current metric  
                    summary_df=summary_df,  # your summary statistics
                    plot_order=plot_order,  # your diagnosis order
                    norm_diagnosis=norm_diagnosis,  # your reference diagnosis
                    save_dir=save_dir,  # your save directory
                    color_columns=color_columns,  # columns to use for coloring
                    diagnosis_palette=diagnosis_palette  # your color palette
                )

    return summary_dict

def visualize_embeddings_multiple(normative_models, data_tensor, annotations_df, 
                                 columns_to_plot=None, device="cuda", figsize=(12, 10)):
    """
    Visualize data in the latent space with multiple colorations based on different metadata columns.
    
    Args:
        normative_models: List of trained models
        data_tensor: Input data tensor
        annotations_df: DataFrame with metadata
        columns_to_plot: List of column names to use for coloring. If None, uses all categorical columns
        device: Device to run model on
        figsize: Figure size for each plot
    
    Returns:
        Dictionary with column names as keys and (figure, plot_df) tuples as values
    """
    
    # Ensure alignment between data tensor and annotations
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
    
    # Generate embeddings (only need to do this once)
    all_embeddings = []
    batch_size = 32
    
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
    
    # Determine which columns to plot
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
        
        # Create plot DataFrame
        plot_df = annotations_df[[col]].copy()
        plot_df["umap_1"] = umap_embeddings[:, 0]
        plot_df["umap_2"] = umap_embeddings[:, 1]
        
        # Handle missing values
        plot_df = plot_df.dropna(subset=[col])
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Determine color palette based on number of unique values
        unique_values = plot_df[col].nunique()
        
        if unique_values <= 10:
            # Use categorical palette for few categories
            palette = sns.color_palette("colorblind", n_colors=unique_values)
        elif unique_values <= 20:
            # Use larger palette for more categories
            palette = sns.color_palette("tab20", n_colors=unique_values)
        else:
            # Use continuous colormap for many values
            palette = "viridis"
        
        # Create scatter plot
        if plot_df[col].dtype in ['object', 'category'] or unique_values <= 20:
            # Categorical coloring
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
            # Continuous coloring for numerical data
            scatter = plt.scatter(
                plot_df["umap_1"],
                plot_df["umap_2"],
                c=plot_df[col],
                cmap=palette,
                s=40,
                alpha=0.7
            )
            plt.colorbar(scatter, label=col)
        
        # Customize plot
        plt.title(f"UMAP Visualization - Colored by {col}", fontsize=16)
        plt.xlabel("UMAP Dimension 1", fontsize=13)
        plt.ylabel("UMAP Dimension 2", fontsize=13)
        
        # Adjust legend if it exists
        if plt.gca().get_legend() is not None:
            plt.legend(title=col, fontsize=10, title_fontsize=11, 
                      bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        
        plt.tight_layout()
        
        # Store results
        results[col] = (plt.gcf(), plot_df.copy())
        
        # Show plot
        plt.show()
    
    return results


def save_latent_visualizations(results, output_dir, dpi=300):
    """
    Save all generated visualizations to files.
    
    Args:
        results: Dictionary returned by visualize_embeddings_multiple
        output_dir: Directory to save plots
        dpi: Resolution for saved images
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for col_name, (fig, plot_df) in results.items():
        # Clean column name for filename
        clean_name = col_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        
        # Save figure
        fig.savefig(
            os.path.join(output_dir, f"umap_{clean_name}.png"),
            dpi=dpi,
            bbox_inches='tight',
            facecolor='white'
        )
        
        print(f"Saved visualization for '{col_name}'")


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
    # os.makedirs(save_dir, exist_ok=True)
    # named_results_df.to_csv(f"{save_dir}/deviation_scores_with_roi_names.csv", index=False)
    
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
    # effect_sizes_df.to_csv(f"{save_dir}/regional_effect_sizes_vs_{norm_diagnosis}.csv", index=False)
    
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
        # top_regions.to_csv(f"{save_dir}/top20_regions_{diagnosis}_vs_{norm_diagnosis}.csv", index=False)
    
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
        # heatmap_df.to_csv(f"{save_dir}/top_regions_heatmap_data_vs_{norm_diagnosis}.csv")
    
    print(f"\nRegional analysis completed. Results saved to {save_dir}")
    print(f"Total effect sizes calculated: {len(effect_sizes_df)}")
    print(f"Average absolute Cliff's Delta: {effect_sizes_df['Abs_Cliffs_Delta'].mean():.3f}")
    print(f"Max absolute Cliff's Delta: {effect_sizes_df['Abs_Cliffs_Delta'].max():.3f}")
    
    return effect_sizes_df