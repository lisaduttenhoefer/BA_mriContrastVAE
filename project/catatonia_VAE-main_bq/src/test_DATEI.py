#helper function 1 for dev_score plotting
# Creates a summary table showing statistics for each diagnosis group colored by the color column
#for colored jitter plots
def create_color_summary_table(data, metric, color_col, diagnoses, save_dir):
    # ... your existing implementation
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
        data_processed.loc[data_processed['Diagnosis_x'].isin(['CTT-SCHZ', 'CTT-MDD']), 'Diagnosis_x'] = 'CTT'
    
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
        
        ctt_suffix = "split" if split_ctt else "combined"
        filename = f"{metric}_jitterplot_colored_by_{color_col}_ctt_{ctt_suffix}.png"
        plt.savefig(f"{save_dir}/figures/distributions/colored_by_columns/{filename}",
                   dpi=300, bbox_inches='tight')
        plt.close()
        create_color_summary_table(filtered_data, metric, color_col, current_plot_order, save_dir)

def calculate_deviations(normative_models, data_tensor, norm_diagnosis, annotations_df, device="cuda"):
    # ... your existing implementation remains the same
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

def plot_deviation_distributions(results_df, save_dir, col_jitter, norm_diagnosis, 
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
    plt.title("Reconstruction Error Distribution by Diagnosis", fontsize=16)
    plt.xlabel("Mean Reconstruction Error", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(title="Diagnosis", fontsize=12)
    sns.despine()
    plt.tight_layout()
    ctt_suffix = "split" if split_ctt else "combined"
    plt.savefig(f"{save_dir}/figures/distributions/recon_error_dist_ctt_{ctt_suffix}.png", dpi=300)
    plt.close()
    
    # Plot KL divergence distributions
    plt.figure(figsize=(12, 8))
    sns.kdeplot(data=results_processed[results_processed['Diagnosis'].isin(available_diagnoses)], 
                x="kl_divergence", hue="Diagnosis", palette=diagnosis_palette, common_norm=False)
    plt.title("KL Divergence Distribution by Diagnosis", fontsize=16)
    plt.xlabel("Mean KL Divergence", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(title="Diagnosis", fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/distributions/kl_div_dist_ctt_{ctt_suffix}.png", dpi=300)
    plt.close()
    
    # Plot combined deviation score distributions
    plt.figure(figsize=(12, 8))
    sns.kdeplot(data=results_processed[results_processed['Diagnosis'].isin(available_diagnoses)], 
                x="deviation_score", hue="Diagnosis", palette=diagnosis_palette, common_norm=False)
    plt.title("Combined Deviation Score Distribution by Diagnosis", fontsize=16)
    plt.xlabel("Deviation Score", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(title="Diagnosis", fontsize=12)
    sns.despine()
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
        
        plt.title(f"{metric.replace('_', ' ').title()}", fontsize=14)
        plt.xlabel("Deviation Metric", fontsize=12)
        plt.ylabel("Diagnosis", fontsize=12)    
        sns.despine()
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
        
        plt.title(f"{metric.replace('_', ' ').title()} by Diagnosis (vs {norm_diagnosis})", fontsize=14)
        plt.xlabel("Deviation Metric", fontsize=12)
        plt.ylabel("Diagnosis", fontsize=12)
        plt.subplots_adjust(left=0.25)
        sns.despine()
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

# Add convenience function to set up consistent plotting parameters
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

# Example usage function
def run_analysis_with_options(results_df, save_dir, col_jitter, norm_diagnosis, 
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
        custom_colors=custom_colors
    )
    
    return summary_dict