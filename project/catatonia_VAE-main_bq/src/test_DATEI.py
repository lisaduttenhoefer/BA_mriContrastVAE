# Create visualization of top affected regions for each diagnosis
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
    ax.set_title(f"{diagnosis} vs. {norm_diagnosis}", fontsize=11, fontweight='bold', pad=10)
    
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


# ALTERNATIVES TO BOOTSTRAP PERCENTILES for your statistical testing:

# 1. PERMUTATION TESTS (Recommended for non-parametric testing)
def permutation_test_effect_size(group1_data, group2_data, n_permutations=10000, effect_type="cliffs_delta"):
    """
    Permutation test for effect sizes
    More robust than parametric tests, doesn't assume normality
    """
    # Observed effect size
    if effect_type == "cliffs_delta":
        observed_effect = calculate_cliffs_delta(group1_data, group2_data)
    elif effect_type == "cohens_d":
        observed_effect = calculate_cohens_d(group1_data, group2_data)
    
    # Combine data for permutation
    combined_data = np.concatenate([group1_data, group2_data])
    n1, n2 = len(group1_data), len(group2_data)
    
    # Permutation distribution
    permuted_effects = []
    for _ in range(n_permutations):
        np.random.shuffle(combined_data)
        perm_group1 = combined_data[:n1]
        perm_group2 = combined_data[n1:]
        
        if effect_type == "cliffs_delta":
            perm_effect = calculate_cliffs_delta(perm_group1, perm_group2)
        elif effect_type == "cohens_d":
            perm_effect = calculate_cohens_d(perm_group1, perm_group2)
        
        permuted_effects.append(perm_effect)
    
    # Calculate p-value
    p_value = np.mean(np.abs(permuted_effects) >= np.abs(observed_effect))
    
    # Calculate confidence interval from permutation distribution
    ci_lower = np.percentile(permuted_effects, 2.5)
    ci_upper = np.percentile(permuted_effects, 97.5)
    
    return {
        'observed_effect': observed_effect,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'permutation_distribution': permuted_effects
    }


# 2. ANALYTICAL CONFIDENCE INTERVALS (for parametric approaches)
def analytical_confidence_intervals(group1_data, group2_data, confidence_level=0.95):
    """
    Calculate analytical confidence intervals for effect sizes
    """
    n1, n2 = len(group1_data), len(group2_data)
    alpha = 1 - confidence_level
    
    # For Cohen's d
    pooled_std = np.sqrt(((n1 - 1) * np.var(group1_data, ddof=1) + 
                         (n2 - 1) * np.var(group2_data, ddof=1)) / (n1 + n2 - 2))
    cohens_d = (np.mean(group1_data) - np.mean(group2_data)) / pooled_std
    
    # Standard error for Cohen's d
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + cohens_d**2 / (2 * (n1 + n2)))
    
    # Confidence interval for Cohen's d
    t_crit = scipy_stats.t.ppf(1 - alpha/2, n1 + n2 - 2)
    d_ci_lower = cohens_d - t_crit * se_d
    d_ci_upper = cohens_d + t_crit * se_d
    
    # For Cliff's Delta (approximate)
    cliffs_delta = calculate_cliffs_delta(group1_data, group2_data)
    se_cliff = np.sqrt((n1 + n2 + 1) / (3 * n1 * n2))  # Approximate SE
    
    cliff_ci_lower = cliffs_delta - 1.96 * se_cliff
    cliff_ci_upper = cliffs_delta + 1.96 * se_cliff
    
    return {
        'cohens_d': cohens_d,
        'cohens_d_ci': (d_ci_lower, d_ci_upper),
        'cliffs_delta': cliffs_delta,
        'cliffs_delta_ci': (cliff_ci_lower, cliff_ci_upper)
    }


# 3. JACKKNIFE RESAMPLING (alternative to bootstrap)
def jackknife_confidence_intervals(group1_data, group2_data, confidence_level=0.95):
    """
    Jackknife confidence intervals for effect sizes
    Leave-one-out resampling approach
    """
    n1, n2 = len(group1_data), len(group2_data)
    
    # Jackknife for group 1
    jackknife_effects_g1 = []
    for i in range(n1):
        g1_jack = np.delete(group1_data, i)
        effect = calculate_cliffs_delta(g1_jack, group2_data)
        jackknife_effects_g1.append(effect)
    
    # Jackknife for group 2
    jackknife_effects_g2 = []
    for i in range(n2):
        g2_jack = np.delete(group2_data, i)
        effect = calculate_cliffs_delta(group1_data, g2_jack)
        jackknife_effects_g2.append(effect)
    
    # Combine jackknife estimates
    all_jackknife = jackknife_effects_g1 + jackknife_effects_g2
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(all_jackknife, (alpha/2) * 100)
    ci_upper = np.percentile(all_jackknife, (1 - alpha/2) * 100)
    
    return {
        'jackknife_mean': np.mean(all_jackknife),
        'jackknife_std': np.std(all_jackknife, ddof=1),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


# 4. BIAS-CORRECTED PERCENTILE METHOD (without bootstrap)
def bias_corrected_percentile_ci(sample_data, statistic_func, confidence_level=0.95, n_samples=1000):
    """
    Bias-corrected confidence intervals using subsampling
    """
    n = len(sample_data)
    subsample_size = int(n * 0.8)  # Use 80% subsamples
    
    # Generate subsamples
    subsample_statistics = []
    for _ in range(n_samples):
        subsample = np.random.choice(sample_data, size=subsample_size, replace=False)
        stat = statistic_func(subsample)
        subsample_statistics.append(stat)
    
    # Original statistic
    original_stat = statistic_func(sample_data)
    
    # Bias correction
    bias = np.mean(subsample_statistics) - original_stat
    bias_corrected_stats = [stat - bias for stat in subsample_statistics]
    
    # Confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bias_corrected_stats, (alpha/2) * 100)
    ci_upper = np.percentile(bias_corrected_stats, (1 - alpha/2) * 100)
    
    return {
        'original_statistic': original_stat,
        'bias': bias,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }