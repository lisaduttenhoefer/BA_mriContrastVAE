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
        if color_values.dtype == 'object' or color_col in ['Sex', 'Co_Diagnosis', 'Dataset']:
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
                except Exception as e:
                    print(f"Error with statistical tests")
    
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
        
        # Calculate 95 confidence interval
        summary_df["ci95"] = 1.96 * summary_df["std"] / np.sqrt(summary_df["count"])
        
        # Add group p-values
        summary_df["p_value"] = summary_df["Diagnosis"].map(
            lambda d: group_pvalues[metric].get(d, np.nan) if d != norm_diagnosis else np.nan
        )
       
        # Sort in desired order (bottom to top)
        diagnosis_order_plot = selected_diagnoses[::-1]
        summary_df["Diagnosis"] = pd.Categorical(summary_df["Diagnosis"], categories=diagnosis_order_plot, ordered=True)
        summary_df = summary_df.sort_values("Diagnosis")
        
        summary_dict[metric] = summary_df
        
        #simple errorbar plot -> Pinaya paper
        plt.figure(figsize=(8, 6))
        
        # Filter only diagnoses that actually have data
        available_diagnoses = filtered_data["Diagnosis"].unique()
        plot_order = [d for d in diagnosis_order_plot if d in available_diagnoses]
        
        
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
        
        plt.title(f"{metric.replace('_', ' ').title()}", fontsize=14)
        plt.xlabel("Deviation Metric", fontsize=12)
        plt.ylabel("Diagnosis", fontsize=12)    
        sns.despine()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/figures/distributions/{metric}_errorbar.png", dpi=300)
        plt.close()
        
        # Create jitterplot with p-values and mean values (current version)
        plt.figure(figsize=(12, 6))  # Made wider to accommodate value labels
        
        mdd_color = diagnosis_palette.get('MDD', '#4c72b0')  # get specific color (pretty blue) -> fallback to blue if MDD not found
        
        sns.stripplot(data=filtered_data, y="Diagnosis", x=metric, 
                    order=plot_order, color=mdd_color, 
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
        plt.subplots_adjust(left=0.25) 
        sns.despine()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/figures/distributions/{metric}_jitterplot_with_values.png", dpi=300, bbox_inches='tight')
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
                    diagnosis_palette=diagnosis_palette  
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
        
        #continous vs binary color palettes depending on data
        if unique_values <= 10:
            palette = sns.color_palette("colorblind", n_colors=unique_values)
        elif unique_values <= 20:
            palette = sns.color_palette("tab20", n_colors=unique_values)
        else:
            palette = "viridis"
        
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
        plt.xlabel("UMAP Dimension 1", fontsize=13)
        plt.ylabel("UMAP Dimension 2", fontsize=13)
        
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

# Function to create subgroups for Catatonia patients
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
        add_catatonia_subgroups=False, 
        metadata_path=None, 
        subgroup_columns=None, 
        high_low_thresholds=None):
    #Analyze regional deviations using ROI names 
    #regional effect sizes -> Cliff's Delta / Cohen's D
    
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
    
    # Create visualization of top affected regions for each diagnosis
    for diagnosis in diagnoses:
        if diagnosis == norm_diagnosis:
            continue
            
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
    
    region_avg_effects = effect_sizes_df.groupby("ROI_Name")["Abs_Cliffs_Delta"].mean().reset_index()
    top_regions_overall = region_avg_effects.sort_values("Abs_Cliffs_Delta", ascending=False).head(30)["ROI_Name"].values

#---------------mit extended metadata subgroups-----------------------

    # Create matrix of effect sizes for these regions including subgroups
    heatmap_data = []

    # Get ALL diagnoses from the effect_sizes_df (includes original + subgroups)
    all_diagnoses = effect_sizes_df["Diagnosis"].unique()  # This now includes subgroups
    print(f"All diagnoses for heatmap: {all_diagnoses}")  # Debug print

    for region in top_regions_overall:
        row = {"ROI_Name": region}
        for diagnosis in all_diagnoses:
            if diagnosis == norm_diagnosis:  # Skip normative diagnosis
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

    # Remove columns that are all NaN (if any)
    heatmap_df = heatmap_df.dropna(axis=1, how='all')

    if len(heatmap_df.columns) > 0:
        # Adjust figure size based on number of columns
        fig_width = max(12, len(heatmap_df.columns) * 1.5)
        plt.figure(figsize=(fig_width, 14))
        
        # Create a mask for NaN values to show them differently
        mask = heatmap_df.isna()
        
        sns.heatmap(heatmap_df, cmap="RdBu_r", center=0, annot=True, fmt=".2f", 
                cbar_kws={"label": "Cliff's Delta"}, mask=mask)
        plt.title(f"Top 30 Regions Effect Sizes vs {norm_diagnosis} (Including Subgroups)")
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        plt.tight_layout()
        plt.savefig(f"{save_dir}/figures/region_effect_heatmap_with_subgroups_vs_{norm_diagnosis}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save heatmap data
        heatmap_df.to_csv(f"{save_dir}/top_regions_heatmap_data_with_subgroups_vs_{norm_diagnosis}.csv")
        
        print(f"Heatmap created with {len(heatmap_df.columns)} diagnosis groups (including subgroups)")
        print(f"Diagnosis groups in heatmap: {list(heatmap_df.columns)}")
    else:
        print("No data available for heatmap creation")

    #---------------------heatmap condensed (only diagnosis)-----------------

     # Create filtered heatmap with only MDD, SCHZ, CTT-SCHZ, CTT-MDD
    desired_diagnoses = ['MDD', 'SCHZ', 'CTT-SCHZ', 'CTT-MDD']

    # Filter the heatmap dataframe to only include desired columns that exist
    available_diagnoses = [diag for diag in desired_diagnoses if diag in heatmap_df.columns]

    if len(available_diagnoses) > 0:
        heatmap_filtered = heatmap_df[available_diagnoses].copy()
        
        # Check if we have any data (not all NaN)
        if not heatmap_filtered.isna().all().all():
            # Adjust figure size for the filtered version - make it wider
            fig_width = max(12, len(available_diagnoses) * 3)  # Increased base width and multiplier
            plt.figure(figsize=(fig_width, 16))  # Also increased height slightly
            
            # Create mask for NaN values
            mask_filtered = heatmap_filtered.isna()
            
            # Create the filtered heatmap
            sns.heatmap(heatmap_filtered, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                    cbar_kws={"label": "Cliff's Delta"}, mask=mask_filtered,
                    square=False, linewidths=0.5)  # Added square=False and linewidths for better appearance
            plt.title(f"Top 30 Regions Effect Sizes vs {norm_diagnosis}\n(MDD, SCHZ, CTT-SCHZ, CTT-MDD)")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{save_dir}/figures/region_effect_heatmap_filtered_vs_{norm_diagnosis}.png", 
                    dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save filtered heatmap data
            heatmap_filtered.to_csv(f"{save_dir}/top_regions_heatmap_data_filtered_vs_{norm_diagnosis}.csv")
            
            print(f"Filtered heatmap created with diagnoses: {available_diagnoses}")
        else:
            print("No data available for filtered heatmap - all values are NaN")
    else:
        print(f"None of the desired diagnoses {desired_diagnoses} found in the data")
        print(f"Available diagnoses in heatmap: {list(heatmap_df.columns)}")

    # Add this code after the filtered heatmap section in your analyze_regional_deviations function

        #---------------------heatmap with dataset split-----------------
        # Ersetzen Sie den gesamten Abschnitt "heatmap with dataset split" mit diesem Code:

    #---------------------Dataset-Split Heatmap (Fixed Version)-----------------

    print("Creating dataset-split heatmap...")

    # Prüfen ob Dataset Spalte existiert
    if 'Dataset' not in results_df.columns:
        print("No 'Dataset' column found - skipping dataset-split heatmap")
    else:
        # Dataset-Kategorien definieren (flexibel für verschiedene Schreibweisen)
        available_datasets = results_df['Dataset'].unique()
        print(f"Available datasets: {available_datasets}")
        
        dataset_categories = {
            'whiteCAT': [d for d in available_datasets if 'whitecat' in str(d).lower() or 'white_cat' in str(d).lower()],
            'NSS': [d for d in available_datasets if 'nss' in str(d).lower()],
            'others': [d for d in available_datasets if not any(x in str(d).lower() for x in ['whitecat', 'white_cat', 'nss'])]
        }
        
        print(f"Dataset categories: {dataset_categories}")
        
        # Effect sizes für Dataset-Splits berechnen
        dataset_split_effects = []
        
        # Für jede Hauptdiagnose
        main_diagnoses = ['SCHZ', 'MDD', 'CTT-SCHZ', 'CTT-MDD']
        
        for diagnosis in main_diagnoses:
            if diagnosis == norm_diagnosis:
                continue
                
            # Alle Patienten dieser Diagnose
            dx_data = results_df[results_df["Diagnosis"] == diagnosis]
            if dx_data.empty:
                print(f"No data found for {diagnosis}")
                continue
                
            print(f"Processing {diagnosis} (total n={len(dx_data)})")
            
            # Für jede Dataset-Kategorie
            for category_name, dataset_list in dataset_categories.items():
                if not dataset_list:
                    continue
                    
                # Daten für diese Kategorie filtern
                category_data = dx_data[dx_data['Dataset'].isin(dataset_list)]
                
                if category_data.empty:
                    print(f"  No {category_name} data for {diagnosis}")
                    continue
                    
                print(f"  {diagnosis}-{category_name}: n={len(category_data)}")
                
                # Effect sizes für alle Regionen berechnen
                for i, region_col in enumerate(region_cols):
                    roi_name = roi_names[i] if i < len(roi_names) else f"Region_{i+1}"
                    
                    category_values = category_data[region_col].values
                    norm_values = norm_data[region_col].values
                    
                    if len(category_values) == 0 or len(norm_values) == 0:
                        continue
                    
                    # Cliff's Delta berechnen
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
            # DataFrame erstellen
            dataset_effects_df = pd.DataFrame(dataset_split_effects)
            
            # Pivot table für Heatmap erstellen
            heatmap_dataset = dataset_effects_df.pivot(
                index='ROI_Name', 
                columns='Diagnosis_Dataset', 
                values='Cliffs_Delta'
            )
            
            # Nur top Regionen verwenden
            heatmap_dataset_top = heatmap_dataset[heatmap_dataset.index.isin(top_regions_overall)]
            
            # Spalten in gewünschter Reihenfolge sortieren
            column_order = []
            for diagnosis in main_diagnoses:
                if diagnosis == norm_diagnosis:
                    continue
                for category in ['whiteCAT', 'NSS', 'others']:
                    col_name = f"{diagnosis}-{category}"
                    if col_name in heatmap_dataset_top.columns:
                        column_order.append(col_name)
            
            heatmap_dataset_ordered = heatmap_dataset_top[column_order]
            
            # Heatmap erstellen
            if not heatmap_dataset_ordered.empty and len(heatmap_dataset_ordered.columns) > 0:
                fig_width = max(16, len(heatmap_dataset_ordered.columns) * 2.5)
                fig_height = max(14, len(heatmap_dataset_ordered) * 0.4)
                
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                
                # NaN Maske
                mask = heatmap_dataset_ordered.isna()
                
                # Heatmap plotten
                sns.heatmap(heatmap_dataset_ordered, 
                        cmap="RdBu_r", 
                        center=0, 
                        annot=True, 
                        fmt=".2f", 
                        mask=mask,
                        cbar_kws={"label": "Cliff's Delta"}, 
                        linewidths=0.5,
                        ax=ax)
                
                # Titel und Labels
                ax.set_title(f"Regional Effect Sizes vs {norm_diagnosis}\n(Split by Dataset: whiteCAT, NSS, others)", 
                            fontsize=16, pad=20)
                ax.set_xlabel("Diagnosis-Dataset", fontsize=12)
                ax.set_ylabel("Brain Region", fontsize=12)
                
                # X-Achse Labels rotieren
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                
                # Vertikale Linien zwischen Diagnosen hinzufügen
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
                
                # Trennlinien zeichnen (außer nach der letzten Gruppe)
                for pos in col_positions[:-1]:
                    ax.axvline(x=pos, color='black', linewidth=2, alpha=0.8)
                
                plt.tight_layout()
                
                # Speichern
                plt.savefig(f"{save_dir}/figures/region_effect_heatmap_dataset_split_vs_{norm_diagnosis}.png",
                        dpi=300, bbox_inches='tight')
                plt.close()
                
                # Daten speichern
                heatmap_dataset_ordered.to_csv(f"{save_dir}/top_regions_heatmap_dataset_split_vs_{norm_diagnosis}.csv")
                
                # Zusammenfassung ausgeben
                print(f"\nDataset-split heatmap created successfully!")
                print(f"Shape: {heatmap_dataset_ordered.shape}")
                print(f"Columns: {list(heatmap_dataset_ordered.columns)}")
                
                # Datenübersicht pro Spalte
                print("\nData availability per column:")
                for col in heatmap_dataset_ordered.columns:
                    non_nan = heatmap_dataset_ordered[col].notna().sum()
                    total = len(heatmap_dataset_ordered)
                    pct = (non_nan/total*100) if total > 0 else 0
                    print(f"  {col}: {non_nan}/{total} regions ({pct:.1f}%)")
                    
                # Sample size Übersicht
                print("\nSample sizes per group:")
                sample_sizes = dataset_effects_df.groupby('Diagnosis_Dataset')['N_Subjects'].first()
                for group, n in sample_sizes.items():
                    print(f"  {group}: n={n}")
                    
            else:
                print("No data available for dataset-split heatmap")
                
        else:
            print("No effect sizes calculated for dataset splits")

        # Daten speichern
        effect_sizes_df.to_csv(f"{save_dir}/effect_sizes_{norm_diagnosis}.csv")
            
        print(f"\nRegional analysis completed. Results saved to {save_dir}")
        print(f"Total effect sizes calculated: {len(effect_sizes_df)}")
        print(f"Average absolute Cliff's Delta: {effect_sizes_df['Abs_Cliffs_Delta'].mean():.3f}")
        print(f"Max absolute Cliff's Delta: {effect_sizes_df['Abs_Cliffs_Delta'].max():.3f}")
        
        if catatonia_subgroups:
            print(f"\nCatatonia subgroups created: {list(catatonia_subgroups.keys())}")
        
    return effect_sizes_df

######################################################## CORRELATION ANALYSIS ################################################################

from statsmodels.stats.multitest import multipletests


def analyze_score_correlations_with_corrections(results_df, metadata_df, save_dir, 
                                               diagnoses_to_include=None,
                                               correction_method='fdr_bh',
                                               alpha=0.1):
    """
    Analysiert Korrelationen mit verschiedenen Korrekturoptionen
    
    Parameters:
    -----------
    results_df : DataFrame
        DataFrame mit Deviation Scores und Diagnosen
    metadata_df : DataFrame  
        DataFrame mit klinischen Scores
    save_dir : str
        Pfad zum Speichern der Ergebnisse
    diagnoses_to_include : list, optional
        Liste der zu inkludierenden Diagnosen
    correction_method : str
        Methode für multiple testing correction:
        - 'bonferroni': Bonferroni correction
        - 'fdr_bh': Benjamini-Hochberg FDR
        - 'fdr_by': Benjamini-Yekutieli FDR
        - 'holm': Holm-Sidak method
        - None: keine Korrektur
    alpha : float
        Signifikanzlevel (default: 0.05)
    """
    
    # Merge der DataFrames
    merged_data = pd.merge(results_df, metadata_df, on='Filename', how='inner')
    print(f"Merged data shape: {merged_data.shape}")
    
    # Bereinigung der Spaltennamen nach dem Merge
    merged_data = merged_data.rename(columns={'Age_x': 'Age', 'Sex_x': 'Sex', 'Dataset_x': 'Dataset'})
    
    # Filtere nach gewünschten Diagnosen
    if diagnoses_to_include:
        merged_data = merged_data[merged_data['Diagnosis_x'].isin(diagnoses_to_include)]
        print(f"Filtered to diagnoses {diagnoses_to_include}. New shape: {merged_data.shape}")
    
    # Definiere die Score-Spalten
    score_columns = ['GAF_Score', 'PANSS_Positive', 'PANSS_Negative', 
                     'PANSS_General', 'PANSS_Total', 'BPRS_Total', 
                     'NCRS_Motor', 'NCRS_Affective', 'NCRS_Behavioral', 
                     'NCRS_Total', 'NSS_Motor', 'NSS_Total']
    
    # Filtere nur vorhandene Score-Spalten
    available_scores = [col for col in score_columns if col in merged_data.columns]
    print(f"Available score columns: {available_scores}")
    
    deviation_metric = 'deviation_score'
    
    # Erstelle Ausgabeverzeichnis
    import os
    os.makedirs(f"{save_dir}/figures/correlations", exist_ok=True)
    
    # 1. BERECHNE ALLE KORRELATIONEN
    correlation_results = calculate_correlations_with_corrections(
        merged_data, available_scores, deviation_metric, correction_method, alpha
    )
    
    # 2. VISUALISIERUNGEN
    plot_correlation_heatmap_with_corrections(
        correlation_results, available_scores, deviation_metric, save_dir, correction_method
    )
    
    # 3. DIAGNOSE-SPEZIFISCHE KORRELATIONEN
    if 'Diagnosis_x' in merged_data.columns:
        diagnosis_results = analyze_diagnosis_correlations_with_corrections(
            merged_data, available_scores, deviation_metric, save_dir, correction_method, alpha
        )
    
    # 4. ZUSAMMENFASSUNGSTABELLEN
    create_corrected_summary_tables(correlation_results, save_dir, correction_method)
    
    return correlation_results, merged_data

def calculate_correlations_with_corrections(merged_data, score_columns, deviation_metric, 
                                          correction_method, alpha):
    """Berechnet Korrelationen mit verschiedenen Korrekturen"""
    
    correlations = {}
    all_p_values = []
    all_tests = []
    
    print(f"\n=== Analyzing correlations for {deviation_metric} ===")
    
    # Sammle alle p-Werte für multiple testing correction
    for score in score_columns:
        valid_data = merged_data[[deviation_metric, score]].dropna()
        
        if len(valid_data) < 10:
            print(f"Insufficient data for {score} (n={len(valid_data)})")
            continue
            
        # Berechne Korrelationen
        pearson_r, pearson_p = pearsonr(valid_data[deviation_metric], valid_data[score])
        spearman_r, spearman_p = spearmanr(valid_data[deviation_metric], valid_data[score])
        
        correlations[score] = {
            'n': len(valid_data),
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'valid_data': valid_data
        }
        
        # Sammle p-Werte für Korrektur
        all_p_values.extend([pearson_p, spearman_p])
        all_tests.extend([(score, 'pearson'), (score, 'spearman')])
    
    # Multiple testing correction
    if correction_method and len(all_p_values) > 0:
        print(f"\nApplying {correction_method} correction to {len(all_p_values)} tests...")
        
        rejected, corrected_p_values, alpha_sidak, alpha_bonf = multipletests(
            all_p_values, alpha=alpha, method=correction_method
        )
        
        # Füge korrigierte p-Werte zu den Ergebnissen hinzu
        for i, (score, test_type) in enumerate(all_tests):
            if score in correlations:
                correlations[score][f'{test_type}_p_corrected'] = corrected_p_values[i]
                correlations[score][f'{test_type}_significant_corrected'] = rejected[i]
        
        print(f"Correction applied. Effective alpha: {alpha_bonf:.6f}")
    
    # Ausgabe der Ergebnisse
    for score, results in correlations.items():
        print(f"\n{score}:")
        print(f"  n={results['n']}")
        print(f"  Pearson: r={results['pearson_r']:.3f}, p={results['pearson_p']:.3f}", end="")
        if correction_method:
            print(f", p_corr={results['pearson_p_corrected']:.3f} {'*' if results['pearson_significant_corrected'] else ''}")
        else:
            print("")
        
        print(f"  Spearman: r={results['spearman_r']:.3f}, p={results['spearman_p']:.3f}", end="")
        if correction_method:
            print(f", p_corr={results['spearman_p_corrected']:.3f} {'*' if results['spearman_significant_corrected'] else ''}")
        else:
            print("")
    
    return {deviation_metric: correlations}

def plot_correlation_heatmap_with_corrections(correlation_results, score_columns, 
                                            deviation_metric, save_dir, correction_method):
    """Erstellt Heatmaps mit und ohne Korrektur"""
    
    def create_correlation_matrices(use_corrected=False):
        pearson_values = []
        spearman_values = []
        pearson_p_values = []
        spearman_p_values = []
        
        p_suffix = '_corrected' if use_corrected else ''
        
        for score in score_columns:
            if score in correlation_results[deviation_metric]:
                pearson_values.append(correlation_results[deviation_metric][score]['pearson_r'])
                spearman_values.append(correlation_results[deviation_metric][score]['spearman_r'])
                
                pearson_p_key = f'pearson_p{p_suffix}'
                spearman_p_key = f'spearman_p{p_suffix}'
                
                pearson_p_values.append(correlation_results[deviation_metric][score].get(pearson_p_key, 
                                       correlation_results[deviation_metric][score]['pearson_p']))
                spearman_p_values.append(correlation_results[deviation_metric][score].get(spearman_p_key,
                                        correlation_results[deviation_metric][score]['spearman_p']))
            else:
                pearson_values.append(np.nan)
                spearman_values.append(np.nan)
                pearson_p_values.append(np.nan)
                spearman_p_values.append(np.nan)
        
        return (np.array(pearson_values).reshape(1, -1),
                np.array(spearman_values).reshape(1, -1),
                np.array(pearson_p_values).reshape(1, -1),
                np.array(spearman_p_values).reshape(1, -1))
    
    def create_annotations(corr_matrix, p_matrix):
        annotations = []
        for i in range(corr_matrix.shape[0]):
            row_annotations = []
            for j in range(corr_matrix.shape[1]):
                if np.isnan(corr_matrix[i, j]):
                    row_annotations.append('')
                else:
                    corr_val = corr_matrix[i, j]
                    p_val = p_matrix[i, j]
                    
                    if p_val < 0.001:
                        stars = '***'
                    elif p_val < 0.01:
                        stars = '**'
                    elif p_val < 0.05:
                        stars = '*'
                    else:
                        stars = ''
                    
                    annotation = f'{corr_val:.3f}{stars}'
                    row_annotations.append(annotation)
            annotations.append(row_annotations)
        return annotations
    
    # Plot ohne Korrektur
    pearson_matrix, spearman_matrix, pearson_p_matrix, spearman_p_matrix = create_correlation_matrices(False)
    
    # Pearson ohne Korrektur
    plt.figure(figsize=(14, 4))
    mask = np.isnan(pearson_matrix)
    pearson_annotations = create_annotations(pearson_matrix, pearson_p_matrix)
    
    sns.heatmap(pearson_matrix, 
                xticklabels=score_columns,
                yticklabels=[deviation_metric.replace('_', ' ').title()],
                annot=pearson_annotations,
                fmt='',
                cmap='RdBu_r', 
                center=0,
                mask=mask,
                square=False,
                cbar_kws={'label': 'Pearson Correlation Coefficient'})
    
    plt.title('Pearson Correlations (Uncorrected)\n(* p<0.05, ** p<0.01, *** p<0.001)', fontsize=14)
    plt.xlabel('Clinical Scores', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/correlations/pearson_correlation_uncorrected.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot mit Korrektur (falls verfügbar)
    if correction_method:
        pearson_matrix_corr, spearman_matrix_corr, pearson_p_matrix_corr, spearman_p_matrix_corr = create_correlation_matrices(True)
        
        # Pearson mit Korrektur
        plt.figure(figsize=(14, 4))
        mask = np.isnan(pearson_matrix_corr)
        pearson_annotations_corr = create_annotations(pearson_matrix_corr, pearson_p_matrix_corr)
        
        sns.heatmap(pearson_matrix_corr, 
                    xticklabels=score_columns,
                    yticklabels=[deviation_metric.replace('_', ' ').title()],
                    annot=pearson_annotations_corr,
                    fmt='',
                    cmap='RdBu_r', 
                    center=0,
                    mask=mask,
                    square=False,
                    cbar_kws={'label': 'Pearson Correlation Coefficient'})
        
        plt.title(f'Pearson Correlations ({correction_method.upper()} Corrected)\n(* p<0.05, ** p<0.01, *** p<0.001)', fontsize=14)
        plt.xlabel('Clinical Scores', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/figures/correlations/pearson_correlation_{correction_method}_corrected.png", dpi=300, bbox_inches='tight')
        plt.close()

def analyze_diagnosis_correlations_with_corrections(merged_data, score_columns, deviation_metric, 
                                                  save_dir, correction_method, alpha):
    """Analysiert Korrelationen nach Diagnosen mit Korrekturen"""
    
    # Filtere HC aus
    diagnoses = [d for d in merged_data['Diagnosis_x'].unique() if d != 'HC']
    
    diagnosis_results = {}
    
    for diagnosis in diagnoses:
        diag_data = merged_data[merged_data['Diagnosis_x'] == diagnosis]
        
        if len(diag_data) < 10:
            continue
        
        print(f"\n=== Analyzing correlations for {diagnosis} (n={len(diag_data)}) ===")
        
        # Berechne Korrelationen für diese Diagnose
        correlations = {}
        all_p_values = []
        all_tests = []
        
        for score in score_columns:
            if score not in diag_data.columns:
                continue
                
            valid_data = diag_data[[deviation_metric, score]].dropna()
            
            if len(valid_data) < 5:
                continue
                
            pearson_r, pearson_p = pearsonr(valid_data[deviation_metric], valid_data[score])
            spearman_r, spearman_p = spearmanr(valid_data[deviation_metric], valid_data[score])
            
            correlations[score] = {
                'n': len(valid_data),
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p
            }
            
            all_p_values.extend([pearson_p, spearman_p])
            all_tests.extend([(score, 'pearson'), (score, 'spearman')])
        
        # Multiple testing correction für diese Diagnose
        if correction_method and len(all_p_values) > 0:
            rejected, corrected_p_values, _, _ = multipletests(
                all_p_values, alpha=alpha, method=correction_method
            )
            
            for i, (score, test_type) in enumerate(all_tests):
                if score in correlations:
                    correlations[score][f'{test_type}_p_corrected'] = corrected_p_values[i]
                    correlations[score][f'{test_type}_significant_corrected'] = rejected[i]
        
        diagnosis_results[diagnosis] = correlations
    
    # Erstelle kombinierte Plots mit Korrekturen
    if diagnosis_results:
        create_combined_diagnosis_heatmap_with_corrections(
            diagnosis_results, list(diagnosis_results.keys()), score_columns, 
            deviation_metric, save_dir, correction_method
        )
    
    return diagnosis_results

def create_combined_diagnosis_heatmap_with_corrections(diagnosis_correlations, diagnoses, 
                                                     score_columns, deviation_metric, 
                                                     save_dir, correction_method):
    """Erstellt kombinierte Heatmaps mit und ohne Korrektur"""
    
    def create_diagnosis_matrices(use_corrected=False):
        correlation_matrix = np.full((len(diagnoses), len(score_columns)), np.nan)
        p_value_matrix = np.full((len(diagnoses), len(score_columns)), np.nan)
        
        p_suffix = '_corrected' if use_corrected else ''
        
        for i, diagnosis in enumerate(diagnoses):
            for j, score in enumerate(score_columns):
                if score in diagnosis_correlations[diagnosis]:
                    correlation_matrix[i, j] = diagnosis_correlations[diagnosis][score]['pearson_r']
                    
                    p_key = f'pearson_p{p_suffix}'
                    p_value_matrix[i, j] = diagnosis_correlations[diagnosis][score].get(
                        p_key, diagnosis_correlations[diagnosis][score]['pearson_p']
                    )
        
        return correlation_matrix, p_value_matrix
    
    def create_annotations(corr_matrix, p_matrix):
        annotations = []
        for i in range(len(diagnoses)):
            row_annotations = []
            for j in range(len(score_columns)):
                if np.isnan(corr_matrix[i, j]):
                    row_annotations.append('')
                else:
                    corr_val = corr_matrix[i, j]
                    p_val = p_matrix[i, j]
                    
                    if p_val < 0.001:
                        stars = '***'
                    elif p_val < 0.01:
                        stars = '**'
                    elif p_val < 0.05:
                        stars = '*'
                    else:
                        stars = ''
                    
                    annotation = f'{corr_val:.2f}{stars}'
                    row_annotations.append(annotation)
            annotations.append(row_annotations)
        return annotations
    
    # Plot ohne Korrektur
    correlation_matrix, p_value_matrix = create_diagnosis_matrices(False)
    
    plt.figure(figsize=(16, max(6, len(diagnoses) * 0.8)))
    mask = np.isnan(correlation_matrix)
    annotations = create_annotations(correlation_matrix, p_value_matrix)
    
    sns.heatmap(correlation_matrix,
                xticklabels=score_columns,
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
    
    plt.title(f'Correlations by Patient Groups (Uncorrected)\n(* p<0.05, ** p<0.01, *** p<0.001)', 
              fontsize=14, pad=20)
    plt.xlabel('Clinical Scores', fontsize=12)
    plt.ylabel('Patient Groups', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/correlations/diagnosis_correlations_uncorrected.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot mit Korrektur
    if correction_method:
        correlation_matrix_corr, p_value_matrix_corr = create_diagnosis_matrices(True)
        
        plt.figure(figsize=(16, max(6, len(diagnoses) * 0.8)))
        mask = np.isnan(correlation_matrix_corr)
        annotations_corr = create_annotations(correlation_matrix_corr, p_value_matrix_corr)
        
        sns.heatmap(correlation_matrix_corr,
                    xticklabels=score_columns,
                    yticklabels=diagnoses,
                    annot=annotations_corr,
                    fmt='',
                    cmap='RdBu_r',
                    center=0,
                    mask=mask,
                    square=False,
                    cbar_kws={'label': 'Pearson Correlation Coefficient'},
                    linewidths=0.5,
                    linecolor='white')
        
        plt.title(f'Correlations by Patient Groups ({correction_method.upper()} Corrected)\n(* p<0.05, ** p<0.01, *** p<0.001)', 
                  fontsize=14, pad=20)
        plt.xlabel('Clinical Scores', fontsize=12)
        plt.ylabel('Patient Groups', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/figures/correlations/diagnosis_correlations_{correction_method}_corrected.png", 
                    dpi=300, bbox_inches='tight')
        plt.close()

def create_corrected_summary_tables(correlation_results, save_dir, correction_method):
    """Erstellt Zusammenfassungstabellen mit Korrekturen"""
    
    summary_data = []
    
    for dev_metric, scores in correlation_results.items():
        for score, results in scores.items():
            row = {
                'Deviation_Metric': dev_metric,
                'Clinical_Score': score,
                'N': results['n'],
                'Pearson_r': results['pearson_r'],
                'Pearson_p': results['pearson_p'],
                'Spearman_r': results['spearman_r'],
                'Spearman_p': results['spearman_p']
            }
            
            # Füge korrigierte Werte hinzu falls verfügbar
            if correction_method:
                row.update({
                    'Pearson_p_corrected': results.get('pearson_p_corrected', np.nan),
                    'Spearman_p_corrected': results.get('spearman_p_corrected', np.nan),
                    'Pearson_significant_corrected': results.get('pearson_significant_corrected', False),
                    'Spearman_significant_corrected': results.get('spearman_significant_corrected', False)
                })
            
            summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Pearson_p')
    
    # Speichere Tabelle
    filename = f"correlation_summary{'_' + correction_method if correction_method else ''}.csv"
    summary_df.to_csv(f"{save_dir}/figures/correlations/{filename}", index=False)
    
    # Zeige Ergebnisse
    print(f"\n=== CORRELATION SUMMARY ===")
    if correction_method:
        print(f"Multiple testing correction: {correction_method}")
        sig_uncorrected = sum(summary_df['Pearson_p'] < 0.05)
        sig_corrected = sum(summary_df['Pearson_significant_corrected'])
        print(f"Significant correlations: {sig_uncorrected} uncorrected, {sig_corrected} corrected")
    else:
        sig_uncorrected = sum(summary_df['Pearson_p'] < 0.05)
        print(f"Significant correlations: {sig_uncorrected}")
    
    return summary_df

# HAUPTFUNKTION MIT KORREKTUROPTIONEN
def run_correlation_analysis(results_df, save_dir, 
                                            correction_method='fdr_bh',
                                            metadata_path=None, 
                                            diagnoses_to_include=None,
                                            alpha=0.1):
    """
    Hauptfunktion für Korrelationsanalyse mit Korrekturen
    
    Parameters:
    -----------
    results_df : DataFrame
        Deine results DataFrame mit Deviation Scores
    save_dir : str
        Pfad zum Speichern
    correction_method : str
        'bonferroni', 'fdr_bh', 'fdr_by', 'holm', oder None
    metadata_path : str, optional
        Pfad zur Metadaten-CSV
    diagnoses_to_include : list, optional
        Liste der Diagnosen zum Einschließen
    alpha : float
        Signifikanzlevel (default: 0.05)
    """
    
    # Lade Metadaten
    if metadata_path:
        metadata_df = pd.read_csv(metadata_path)
    else:
        metadata_df = pd.read_csv('/workspace/project/catatonia_VAE-main_bq/metadata_20250110/full_data_with_codiagnosis_and_scores.csv')
    
    # Führe Analyse durch
    correlation_results, merged_data = analyze_score_correlations_with_corrections(
        results_df=results_df,
        metadata_df=metadata_df, 
        save_dir=save_dir,
        diagnoses_to_include=diagnoses_to_include,
        correction_method=correction_method,
        alpha=alpha
    )
    
    print(f"\nAnalysis completed with correction method: {correction_method}")
    print(f"Results saved to: {save_dir}/figures/correlations/")
    
    return correlation_results, merged_data