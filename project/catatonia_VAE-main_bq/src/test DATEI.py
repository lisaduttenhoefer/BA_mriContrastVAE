def calculate_deviations_improved_pvalues(normative_models, data_tensor, norm_diagnosis, annotations_df, device="cuda"):
    """
    Calculate deviation scores with improved p-value calculation specifically for 
    deviations from the normative diagnosis group.
    
    Args:
        normative_models: List of trained normative VAE models
        data_tensor: Tensor of input data to evaluate
        annotations_df: DataFrame with subject metadata
        device: Computing device
        
    Returns:
        DataFrame with deviation scores for each subject
    """

    # Verify data alignment (same as before)
    total_models = len(normative_models)
    total_subjects = data_tensor.shape[0]
    
    if total_subjects != len(annotations_df):
        print(f"WARNING: Size mismatch detected: {total_subjects} samples in data tensor vs {len(annotations_df)} rows in annotations")
        print("Creating properly aligned dataset by extracting common subjects...")
        valid_indices = list(range(min(total_subjects, len(annotations_df))))
        aligned_annotations = annotations_df.iloc[valid_indices].reset_index(drop=True)
        annotations_df = aligned_annotations
        print(f"Aligned datasets - working with {len(annotations_df)} subjects")
    
    # Calculate deviation metrics (same as before)
    all_recon_errors = np.zeros((total_subjects, total_models))
    all_kl_divs = np.zeros((total_subjects, total_models))
    all_z_scores = np.zeros((total_subjects, data_tensor.shape[1], total_models))
    
    for i, model in enumerate(normative_models):
        model.eval()
        model.to(device)
        with torch.no_grad():
            batch_data = data_tensor.to(device)
            recon, mu, log_var = model(batch_data)
            
            recon_error = torch.mean((batch_data - recon) ** 2, dim=1).cpu().numpy()
            all_recon_errors[:, i] = recon_error
            
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).cpu().numpy()
            all_kl_divs[:, i] = kl_div
            
            z_scores = ((batch_data - recon) ** 2).cpu().numpy()
            all_z_scores[:, :, i] = z_scores
        
        torch.cuda.empty_cache()
    
    # Average across bootstrap models
    mean_recon_error = np.mean(all_recon_errors, axis=1)
    std_recon_error = np.std(all_recon_errors, axis=1)
    mean_kl_div = np.mean(all_kl_divs, axis=1)
    std_kl_div = np.std(all_kl_divs, axis=1)
    mean_region_z_scores = np.mean(all_z_scores, axis=2)
    
    # Create result DataFrame
    results_df = annotations_df[["Filename", "Diagnosis", "Age", "Sex", "Dataset"]].copy()
    results_df["reconstruction_error"] = mean_recon_error
    results_df["reconstruction_error_std"] = std_recon_error
    results_df["kl_divergence"] = mean_kl_div
    results_df["kl_divergence_std"] = std_kl_div
    
    # Add region-wise z-scores
    new_columns = pd.DataFrame(
        mean_region_z_scores, 
        columns=[f"region_{i}_z_score" for i in range(mean_region_z_scores.shape[1])]
    )
    results_df = pd.concat([results_df, new_columns], axis=1)
    
    # Calculate combined deviation scores (same as before)
    scaler_recon = StandardScaler()
    scaler_kl = StandardScaler()
    
    z_norm_recon = scaler_recon.fit_transform(mean_recon_error.reshape(-1, 1)).flatten()
    z_norm_kl = scaler_kl.fit_transform(mean_kl_div.reshape(-1, 1)).flatten()
    
    results_df["deviation_score_zscore"] = (z_norm_recon + z_norm_kl) / 2
    
    recon_percentiles = stats.rankdata(mean_recon_error) / len(mean_recon_error)
    kl_percentiles = stats.rankdata(mean_kl_div) / len(mean_kl_div)
    results_df["deviation_score_percentile"] = (recon_percentiles + kl_percentiles) / 2
    
    min_recon = results_df["reconstruction_error"].min()
    max_recon = results_df["reconstruction_error"].max()
    norm_recon = (results_df["reconstruction_error"] - min_recon) / (max_recon - min_recon)
    
    min_kl = results_df["kl_divergence"].min()
    max_kl = results_df["kl_divergence"].max()
    norm_kl = (results_df["kl_divergence"] - min_kl) / (max_kl - min_kl)
    
    results_df["deviation_score"] = (norm_recon + norm_kl) / 2
    
    #=== IMPROVED P-VALUE CALCULATION ===
    
    # Get control group data
    control_mask = results_df["Diagnosis"] == norm_diagnosis
    if not control_mask.any():
        print(f"WARNING: No control group '{norm_diagnosis}' found in data. Available diagnoses: {results_df['Diagnosis'].unique()}")
        control_indices = np.argsort(results_df["deviation_score_zscore"])[:len(results_df)//4]
        control_mask = np.zeros(len(results_df), dtype=bool)
        control_mask[control_indices] = True
        print(f"Using bottom 25% ({control_mask.sum()} subjects) as reference group")
    
    # Extract normative group data
    norm_recon = results_df.loc[control_mask, "reconstruction_error"].values
    norm_kl = results_df.loc[control_mask, "kl_divergence"].values
    norm_combined = results_df.loc[control_mask, "deviation_score_zscore"].values
    
    print(f"Using {len(norm_recon)} subjects from '{norm_diagnosis}' as normative reference")
    
    # Method 1: Empirical p-values (current approach, but cleaner)
    p_values_empirical_recon = []
    p_values_empirical_kl = []
    p_values_empirical_combined = []
    
    # Method 2: Parametric p-values (assuming normal distribution)
    p_values_parametric_recon = []
    p_values_parametric_kl = []
    p_values_parametric_combined = []
    
    # Method 3: Z-scores relative to normative group
    z_scores_recon = []
    z_scores_kl = []
    z_scores_combined = []
    
    # Method 4: Percentile-based deviation scores
    percentile_scores_recon = []
    percentile_scores_kl = []
    percentile_scores_combined = []
    
    # Calculate normative group statistics
    norm_recon_mean = np.mean(norm_recon)
    norm_recon_std = np.std(norm_recon)
    norm_kl_mean = np.mean(norm_kl)
    norm_kl_std = np.std(norm_kl)
    norm_combined_mean = np.mean(norm_combined)
    norm_combined_std = np.std(norm_combined)
    
    for idx, row in results_df.iterrows():
        subject_recon = row["reconstruction_error"]
        subject_kl = row["kl_divergence"]
        subject_combined = row["deviation_score_zscore"]
        
        # === Method 1: Empirical p-values ===
        # Probability of seeing this value or higher in the normative group
        p_emp_recon = np.mean(norm_recon >= subject_recon)
        p_emp_kl = np.mean(norm_kl >= subject_kl)
        p_emp_combined = np.mean(norm_combined >= subject_combined)
        
        # For very extreme values, use (count + 1) / (n + 1) to avoid p=0
        p_emp_recon = max(p_emp_recon, 1/(len(norm_recon) + 1))
        p_emp_kl = max(p_emp_kl, 1/(len(norm_kl) + 1))
        p_emp_combined = max(p_emp_combined, 1/(len(norm_combined) + 1))
        
        p_values_empirical_recon.append(p_emp_recon)
        p_values_empirical_kl.append(p_emp_kl)
        p_values_empirical_combined.append(p_emp_combined)
        
        # === Method 2: Parametric p-values (assuming normal distribution) ===
        # Calculate z-score and then p-value
        if norm_recon_std > 0:
            z_recon = (subject_recon - norm_recon_mean) / norm_recon_std
            p_param_recon = 1 - stats.norm.cdf(z_recon)  # One-tailed test (higher is worse)
        else:
            z_recon = 0
            p_param_recon = 0.5
            
        if norm_kl_std > 0:
            z_kl = (subject_kl - norm_kl_mean) / norm_kl_std
            p_param_kl = 1 - stats.norm.cdf(z_kl)
        else:
            z_kl = 0
            p_param_kl = 0.5
            
        if norm_combined_std > 0:
            z_combined = (subject_combined - norm_combined_mean) / norm_combined_std
            p_param_combined = 1 - stats.norm.cdf(z_combined)
        else:
            z_combined = 0
            p_param_combined = 0.5
        
        p_values_parametric_recon.append(p_param_recon)
        p_values_parametric_kl.append(p_param_kl)
        p_values_parametric_combined.append(p_param_combined)
        
        # === Method 3: Store z-scores ===
        z_scores_recon.append(z_recon)
        z_scores_kl.append(z_kl)
        z_scores_combined.append(z_combined)
        
        # === Method 4: Percentile scores ===
        # What percentile is this subject in the normative distribution?
        percentile_recon = stats.percentileofscore(norm_recon, subject_recon, kind='rank')
        percentile_kl = stats.percentileofscore(norm_kl, subject_kl, kind='rank')
        percentile_combined = stats.percentileofscore(norm_combined, subject_combined, kind='rank')
        
        percentile_scores_recon.append(percentile_recon)
        percentile_scores_kl.append(percentile_kl)
        percentile_scores_combined.append(percentile_combined)
    
    # Add all p-value methods to DataFrame
    results_df["p_value_empirical_recon"] = p_values_empirical_recon
    results_df["p_value_empirical_kl"] = p_values_empirical_kl
    results_df["p_value_empirical_combined"] = p_values_empirical_combined
    
    results_df["p_value_parametric_recon"] = p_values_parametric_recon
    results_df["p_value_parametric_kl"] = p_values_parametric_kl
    results_df["p_value_parametric_combined"] = p_values_parametric_combined
    
    results_df["z_score_vs_norm_recon"] = z_scores_recon
    results_df["z_score_vs_norm_kl"] = z_scores_kl
    results_df["z_score_vs_norm_combined"] = z_scores_combined
    
    results_df["percentile_vs_norm_recon"] = percentile_scores_recon
    results_df["percentile_vs_norm_kl"] = percentile_scores_kl
    results_df["percentile_vs_norm_combined"] = percentile_scores_combined
    
    # === Method 5: Robust deviation scores using MAD (Median Absolute Deviation) ===
    # More robust to outliers than standard deviation
    norm_recon_median = np.median(norm_recon)
    norm_kl_median = np.median(norm_kl)
    norm_combined_median = np.median(norm_combined)
    
    norm_recon_mad = stats.median_abs_deviation(norm_recon)
    norm_kl_mad = stats.median_abs_deviation(norm_kl)
    norm_combined_mad = stats.median_abs_deviation(norm_combined)
    
    # Calculate robust z-scores (using MAD instead of std)
    if norm_recon_mad > 0:
        results_df["robust_z_score_recon"] = (results_df["reconstruction_error"] - norm_recon_median) / norm_recon_mad
    else:
        results_df["robust_z_score_recon"] = 0
        
    if norm_kl_mad > 0:
        results_df["robust_z_score_kl"] = (results_df["kl_divergence"] - norm_kl_median) / norm_kl_mad
    else:
        results_df["robust_z_score_kl"] = 0
        
    if norm_combined_mad > 0:
        results_df["robust_z_score_combined"] = (results_df["deviation_score_zscore"] - norm_combined_median) / norm_combined_mad
    else:
        results_df["robust_z_score_combined"] = 0
    
    # === Add summary statistics ===
    results_df["is_outlier_recon_2std"] = np.abs(results_df["z_score_vs_norm_recon"]) > 2
    results_df["is_outlier_recon_3std"] = np.abs(results_df["z_score_vs_norm_recon"]) > 3
    results_df["is_outlier_combined_2std"] = np.abs(results_df["z_score_vs_norm_combined"]) > 2
    results_df["is_outlier_combined_3std"] = np.abs(results_df["z_score_vs_norm_combined"]) > 3
    
    # Significance flags
    results_df["significant_empirical_p05"] = results_df["p_value_empirical_combined"] < 0.05
    results_df["significant_parametric_p05"] = results_df["p_value_parametric_combined"] < 0.05
    results_df["significant_empirical_p01"] = results_df["p_value_empirical_combined"] < 0.01
    results_df["significant_parametric_p01"] = results_df["p_value_parametric_combined"] < 0.01
    
    # Print summary statistics
    print(f"\n=== DEVIATION ANALYSIS SUMMARY ===")
    print(f"Normative group ('{norm_diagnosis}'): n={len(norm_recon)}")
    print(f"Total subjects analyzed: n={len(results_df)}")
    
    print(f"\nNormative group statistics:")
    print(f"  Reconstruction Error: {norm_recon_mean:.4f} ± {norm_recon_std:.4f}")
    print(f"  KL Divergence: {norm_kl_mean:.4f} ± {norm_kl_std:.4f}")
    print(f"  Combined Score: {norm_combined_mean:.4f} ± {norm_combined_std:.4f}")
    
    print(f"\nOutlier detection (>2 SD from norm):")
    for diagnosis in results_df["Diagnosis"].unique():
        if diagnosis == norm_diagnosis:
            continue
        dx_data = results_df[results_df["Diagnosis"] == diagnosis]
        n_outliers_2std = dx_data["is_outlier_combined_2std"].sum()
        n_outliers_3std = dx_data["is_outlier_combined_3std"].sum()
        n_sig_emp = dx_data["significant_empirical_p05"].sum()
        n_sig_param = dx_data["significant_parametric_p05"].sum()
        
        print(f"  {diagnosis} (n={len(dx_data)}): {n_outliers_2std} outliers >2SD, {n_outliers_3std} outliers >3SD")
        print(f"    Significant (p<0.05): {n_sig_emp} empirical, {n_sig_param} parametric")
    
    return results_df


def create_comprehensive_deviation_plots(results_df, save_dir, norm_diagnosis):
    """
    Create comprehensive plots showing different p-value calculation methods.
    """
    os.makedirs(f"{save_dir}/figures/pvalue_analysis", exist_ok=True)
    
    # Color palette
    palette = sns.light_palette("blue", n_colors=4, reverse=True)
    diagnosis_order = ["HC", "SCHZ", "CTT", "MDD"]
    diagnosis_palette = dict(zip(diagnosis_order, palette))
    
    # 1. Compare empirical vs parametric p-values
    plt.figure(figsize=(15, 10))
    
    diagnoses = [d for d in diagnosis_order if d in results_df["Diagnosis"].unique() and d != norm_diagnosis]
    
    for i, diagnosis in enumerate(diagnoses):
        dx_data = results_df[results_df["Diagnosis"] == diagnosis]
        if len(dx_data) == 0:
            continue
            
        plt.subplot(2, len(diagnoses), i+1)
        plt.scatter(dx_data["p_value_empirical_combined"], dx_data["p_value_parametric_combined"], 
                   alpha=0.6, color=diagnosis_palette.get(diagnosis, 'gray'))
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.7)
        plt.axhline(y=0.05, color='red', linestyle=':', alpha=0.7)
        plt.axvline(x=0.05, color='red', linestyle=':', alpha=0.7)
        plt.xlabel("Empirical p-value")
        plt.ylabel("Parametric p-value")
        plt.title(f"{diagnosis} (n={len(dx_data)})")
        
        plt.subplot(2, len(diagnoses), i+1+len(diagnoses))
        plt.hist2d(dx_data["p_value_empirical_combined"], dx_data["p_value_parametric_combined"], 
                  bins=20, cmap='Blues')
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.7)
        plt.axhline(y=0.05, color='red', linestyle=':', alpha=0.7)
        plt.axvline(x=0.05, color='red', linestyle=':', alpha=0.7)
        plt.xlabel("Empirical p-value")
        plt.ylabel("Parametric p-value")
        plt.colorbar(label='Count')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/pvalue_analysis/empirical_vs_parametric_pvalues.png", dpi=300)
    plt.close()
    
    # 2. Z-score distributions by diagnosis
    plt.figure(figsize=(15, 5))
    
    metrics = ["z_score_vs_norm_recon", "z_score_vs_norm_kl", "z_score_vs_norm_combined"]
    titles = ["Reconstruction Error Z-score", "KL Divergence Z-score", "Combined Z-score"]
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.subplot(1, 3, i+1)
        
        for diagnosis in diagnosis_order:
            if diagnosis not in results_df["Diagnosis"].unique():
                continue
            dx_data = results_df[results_df["Diagnosis"] == diagnosis]
            if len(dx_data) == 0:
                continue
                
            if diagnosis == norm_diagnosis:
                # Plot normative group as reference
                sns.kdeplot(dx_data[metric], label=f"{diagnosis} (norm)", 
                           color='black', linestyle='--', alpha=0.8)
            else:
                sns.kdeplot(dx_data[metric], label=f"{diagnosis} (n={len(dx_data)})", 
                           color=diagnosis_palette.get(diagnosis, 'gray'))
        
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.axvline(x=2, color='red', linestyle=':', alpha=0.7, label='2SD')
        plt.axvline(x=-2, color='red', linestyle=':', alpha=0.7)
        plt.axvline(x=3, color='red', linestyle='--', alpha=0.7, label='3SD')
        plt.axvline(x=-3, color='red', linestyle='--', alpha=0.7)
        
        plt.xlabel("Z-score")
        plt.ylabel("Density")
        plt.title(title)
        plt.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/pvalue_analysis/zscore_distributions.png", dpi=300)
    plt.close()
    
    # 3. Percentile distributions
    plt.figure(figsize=(15, 5))
    
    percentile_metrics = ["percentile_vs_norm_recon", "percentile_vs_norm_kl", "percentile_vs_norm_combined"]
    
    for i, (metric, title) in enumerate(zip(percentile_metrics, titles)):
        plt.subplot(1, 3, i+1)
        
        for diagnosis in diagnosis_order:
            if diagnosis not in results_df["Diagnosis"].unique() or diagnosis == norm_diagnosis:
                continue
            dx_data = results_df[results_df["Diagnosis"] == diagnosis]
            if len(dx_data) == 0:
                continue
                
            sns.kdeplot(dx_data[metric], label=f"{diagnosis} (n={len(dx_data)})", 
                       color=diagnosis_palette.get(diagnosis, 'gray'))
        
        plt.axvline(x=50, color='black', linestyle='-', alpha=0.3, label='Median')
        plt.axvline(x=95, color='red', linestyle=':', alpha=0.7, label='95th percentile')
        plt.axvline(x=99, color='red', linestyle='--', alpha=0.7, label='99th percentile')
        
        plt.xlabel("Percentile vs. Normative Group")
        plt.ylabel("Density")
        plt.title(f"{title} - Percentiles")
        plt.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/pvalue_analysis/percentile_distributions.png", dpi=300)
    plt.close()
    
    # 4. Summary table of different methods
    summary_data = []
    
    methods = [
        ("p_value_empirical_combined", "Empirical p-value"),
        ("p_value_parametric_combined", "Parametric p-value"),
        ("z_score_vs_norm_combined", "Z-score (>2SD)"),
        ("percentile_vs_norm_combined", "Percentile (>95th)")
    ]
    
    for diagnosis in diagnosis_order:
        if diagnosis not in results_df["Diagnosis"].unique() or diagnosis == norm_diagnosis:
            continue
        dx_data = results_df[results_df["Diagnosis"] == diagnosis]
        if len(dx_data) == 0:
            continue
            
        row = {"Diagnosis": diagnosis, "N": len(dx_data)}
        
        # Count significant subjects by different methods
        row["Empirical_p<0.05"] = (dx_data["p_value_empirical_combined"] < 0.05).sum()
        row["Empirical_%"] = (dx_data["p_value_empirical_combined"] < 0.05).mean() * 100
        
        row["Parametric_p<0.05"] = (dx_data["p_value_parametric_combined"] < 0.05).sum()
        row["Parametric_%"] = (dx_data["p_value_parametric_combined"] < 0.05).mean() * 100
        
        row["Z>2SD"] = (np.abs(dx_data["z_score_vs_norm_combined"]) > 2).sum()
        row["Z>2SD_%"] = (np.abs(dx_data["z_score_vs_norm_combined"]) > 2).mean() * 100
        
        row["Z>3SD"] = (np.abs(dx_data["z_score_vs_norm_combined"]) > 3).sum()
        row["Z>3SD_%"] = (np.abs(dx_data["z_score_vs_norm_combined"]) > 3).mean() * 100
        
        row["Percentile>95"] = (dx_data["percentile_vs_norm_combined"] > 95).sum()
        row["Percentile>95_%"] = (dx_data["percentile_vs_norm_combined"] > 95).mean() * 100
        
        row["Mean_Z_score"] = dx_data["z_score_vs_norm_combined"].mean()
        row["Mean_Percentile"] = dx_data["percentile_vs_norm_combined"].mean()
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{save_dir}/figures/pvalue_analysis/method_comparison_summary.csv", index=False)
    
    print(f"\nMethod comparison summary:")
    print(summary_df.to_string(index=False))
    
    return summary_df