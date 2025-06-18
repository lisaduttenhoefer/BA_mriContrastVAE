import pandas as pd

# Lade die Daten
csv_path = "/workspace/project/catatonia_VAE-main_bq/analysis/TESTING/deviation_results_norm_results_HC_V_g_m_neuromorphometrics_20250605_1037_20250606_130527/deviation_scores_with_roi_names.csv"

df = pd.read_csv(csv_path)

# Wähle nur die Z-Scores der Hirnregionen
metadata_columns = ["Filename", "Diagnosis", "Age", "Sex", "Dataset","reconstruction_error","reconstruction_error_std","kl_divergence","kl_divergence_std", "deviation_score","deviation_score_zscore","deviation_score_zscore","deviation_score_percentile"]
                                          
region_columns = [col for col in df.columns if col not in metadata_columns]

# Berechne die Korrelationsmatrix
corr_matrix = df[region_columns].corr()

# Finde stark korrelierende Paare (Threshold anpassen)
threshold = 0.75  
strong_corrs = (
    corr_matrix.abs()
    .unstack()
    .sort_values(ascending=False)
    .drop_duplicates()
)

# Filtere die stärksten Korrelationen
strong_corrs = strong_corrs[strong_corrs >= threshold]

# Speichern der Ergebnisse in eine Datei
with open("/workspace/project/catatonia_VAE-main_bq/src/output.txt", "w") as f:
    f.write("Stark korrelierende Hirnregionen:\n\n")
    f.write(str(strong_corrs.head(20)))

print("Die stärksten Korrelationen wurden in 'output.txt' gespeichert.")
