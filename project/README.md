# Normative VAE : clinical indication analysis 
## Introduction
## File explanation
## How to use
### Step-by-step
### Adaptability
**VOLUME TYPE SELECTION:**
- gray matter
- white matter
- cerebrospinal fluid

**ATLAS SELECTION:**
- cobra
- neuromorphometrics
- suit
- lpba40
- thalamus
- thalamic nuclei

**NORM DIAGNOSIS:**

**NORMALIZATION METHOD:**
- (A) Pinaya Normalization approach *(normalize_and_scale_pinaya)*
        -> Incorporates TICV normalization (if available) to account for total brain volume differences.
        -> Uses a robust (median-IQR) scaling approach, making it resistant to outliers while maintaining relative volume structure.
        -> when adjusting for overall brain size differences is crucial.

- (B) Log + MinMax Normalization + Z-Score *(normalize_log_minmax_column_zscore_row)*
        
        -> Spaltenweise Normalisierung: Logarithmische Transformation zur Stabilisierung der Varianz, gefolgt von MinMax-Skalierung.
        -> Zeilenweise Normalisierung: Z-Score-Normalisierung für jede Gehirnregion je Proband.
        -> Log transformation handles skewed distributions, MinMax scaling brings values into a bounded range, and Z-score ensures comparability within subjects.
        -> Especially useful for datasets with highly varying magnitude values.
- (C) Quantile Normalization + Robust Scaling *(normalize_quantile_column_robust_row)*
        
        -> Spaltenweise Normalisierung: Quantile-Normalisierung, um Werte innerhalb einer Verteilung zu bringen.
        -> Zeilenweise Skalierung: Robust Scaling über die Gehirnregionen für jeden Probanden.
        -> Quantile normalization forces similar distributions across subjects, making comparisons easier.
        -> Robust Scaling further prevents outliers from distorting values within subjects.
        -> Good for handling data that needs uniform distributions for better statistical inference.

- (D) Z-Score Normalization + Robust Scaling *(normalize_zscore_column_robust_row)*
        
        -> Spaltenweise Normalisierung: Z-Score-Normalisierung über die Probanden für jede Gehirnregion (Standardisierung).
        -> Zeilenweise Skalierung: Robust Scaling über die Gehirnregionen für jeden Probanden (Median-IQR-Scaling).
        -> Ensures standardized distributions across subjects while mitigating outlier effects using IQR scaling within subjects.
        -> Good for cases where preserving the relative variability of brain regions is important.