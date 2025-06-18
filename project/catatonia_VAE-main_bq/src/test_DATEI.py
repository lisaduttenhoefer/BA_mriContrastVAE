import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

def analyze_score_correlations(results_df, metadata_df, save_dir, diagnoses_to_include=None):
    """
    Analysiert Korrelationen zwischen verschiedenen klinischen Scores und Deviation Score
    
    Parameters:
    -----------
    results_df : DataFrame
        DataFrame mit Deviation Scores und Diagnosen
    metadata_df : DataFrame  
        DataFrame mit klinischen Scores
    save_dir : str
        Pfad zum Speichern der Ergebnisse
    diagnoses_to_include : list, optional
        Liste der zu inkludierenden Diagnosen (default: alle)
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
    
    # Definiere die Score-Spalten (klinische Bewertungen)
    score_columns = ['GAF_Score', 'PANSS_Positive', 'PANSS_Negative', 
                     'PANSS_General', 'PANSS_Total', 'BPRS_Total', 
                     'NCRS_Motor', 'NCRS_Affective', 'NCRS_Behavioral', 
                     'NCRS_Total', 'NSS_Motor', 'NSS_Total']
    
    # Filtere nur vorhandene Score-Spalten
    available_scores = [col for col in score_columns if col in merged_data.columns]
    print(f"Available score columns: {available_scores}")
    
    # Nur deviation_score analysieren
    deviation_metric = 'deviation_score'
    
    # Erstelle Ausgabeverzeichnis
    import os
    os.makedirs(f"{save_dir}/figures/correlations", exist_ok=True)
    
    # 1. KORRELATIONSMATRIX ERSTELLEN
    correlation_results = {}
    
    if deviation_metric not in merged_data.columns:
        print(f"ERROR: {deviation_metric} not found in data!")
        return None, None
        
    print(f"\n=== Analyzing correlations for {deviation_metric} ===")
    
    correlations = {}
    for score in available_scores:
        # Entferne NaN-Werte für diese Kombination
        valid_data = merged_data[[deviation_metric, score]].dropna()
        
        if len(valid_data) < 10:  # Mindestens 10 Datenpunkte
            print(f"Insufficient data for {score} (n={len(valid_data)})")
            continue
            
        # Berechne Pearson und Spearman Korrelationen
        pearson_r, pearson_p = pearsonr(valid_data[deviation_metric], valid_data[score])
        spearman_r, spearman_p = spearmanr(valid_data[deviation_metric], valid_data[score])
        
        correlations[score] = {
            'n': len(valid_data),
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p
        }
        
        print(f"{score}: n={len(valid_data)}, Pearson r={pearson_r:.3f} (p={pearson_p:.3f}), "
              f"Spearman r={spearman_r:.3f} (p={spearman_p:.3f})")
    
    correlation_results[deviation_metric] = correlations
    
    # 2. VISUALISIERUNG DER KORRELATIONEN
    
    # Heatmap der Korrelationen (nur für deviation_score)
    plot_correlation_heatmap(correlation_results, available_scores, deviation_metric, save_dir)
    
    # Scatterplots für signifikante Korrelationen
    plot_significant_correlations(merged_data, correlation_results, save_dir, p_threshold=0.05)
    
    # Korrelationen nach Diagnosegruppen
    if 'Diagnosis_x' in merged_data.columns:
        analyze_correlations_by_diagnosis(merged_data, available_scores, deviation_metric, save_dir)
    
    # 3. ERGEBNISTABELLE ERSTELLEN
    create_correlation_summary_table(correlation_results, save_dir)
    
    return correlation_results, merged_data

def plot_correlation_heatmap(correlation_results, score_columns, deviation_metric, save_dir):
    """Erstellt Heatmaps der Korrelationskoeffizienten mit Signifikanz-Markierungen"""
    
    # Erstelle Arrays für Korrelationswerte und p-Werte
    pearson_values = []
    spearman_values = []
    pearson_p_values = []
    spearman_p_values = []
    
    for score in score_columns:
        if score in correlation_results[deviation_metric]:
            pearson_values.append(correlation_results[deviation_metric][score]['pearson_r'])
            spearman_values.append(correlation_results[deviation_metric][score]['spearman_r'])
            pearson_p_values.append(correlation_results[deviation_metric][score]['pearson_p'])
            spearman_p_values.append(correlation_results[deviation_metric][score]['spearman_p'])
        else:
            pearson_values.append(np.nan)
            spearman_values.append(np.nan)
            pearson_p_values.append(np.nan)
            spearman_p_values.append(np.nan)
    
    # Konvertiere zu numpy arrays und reshape für heatmap
    pearson_matrix = np.array(pearson_values).reshape(1, -1)
    spearman_matrix = np.array(spearman_values).reshape(1, -1)
    pearson_p_matrix = np.array(pearson_p_values).reshape(1, -1)
    spearman_p_matrix = np.array(spearman_p_values).reshape(1, -1)
    
    # Erstelle Annotations mit Signifikanz-Sternen
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
                    
                    # Bestimme Signifikanz-Level
                    if p_val < 0.001:
                        stars = '***'
                    elif p_val < 0.01:
                        stars = '**'
                    elif p_val < 0.05:
                        stars = '*'
                    else:
                        stars = ''
                    
                    # Formatiere Annotation
                    annotation = f'{corr_val:.3f}{stars}'
                    row_annotations.append(annotation)
            annotations.append(row_annotations)
        return annotations
    
    # Plot Pearson Korrelationen
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
    
    plt.title('Pearson Correlations: Clinical Scores vs Deviation Score\n(* p<0.05, ** p<0.01, *** p<0.001)', fontsize=14)
    plt.xlabel('Clinical Scores', fontsize=12)
    plt.ylabel('', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/correlations/pearson_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Spearman Korrelationen
    plt.figure(figsize=(14, 4))
    mask = np.isnan(spearman_matrix)
    spearman_annotations = create_annotations(spearman_matrix, spearman_p_matrix)
    
    sns.heatmap(spearman_matrix,
                xticklabels=score_columns,
                yticklabels=[deviation_metric.replace('_', ' ').title()],
                annot=spearman_annotations,
                fmt='',
                cmap='RdBu_r',
                center=0,
                mask=mask,
                square=False,
                cbar_kws={'label': 'Spearman Correlation Coefficient'})
    
    plt.title('Spearman Correlations: Clinical Scores vs Deviation Score\n(* p<0.05, ** p<0.01, *** p<0.001)', fontsize=14)
    plt.xlabel('Clinical Scores', fontsize=12)
    plt.ylabel('', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/correlations/spearman_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_significant_correlations(merged_data, correlation_results, save_dir, p_threshold=0.05):
    """Plot Scatterplots für signifikante Korrelationen"""
    
    significant_pairs = []
    
    # Finde signifikante Korrelationen
    for dev_metric, scores in correlation_results.items():
        for score, results in scores.items():
            if results['pearson_p'] < p_threshold:
                significant_pairs.append((dev_metric, score, results))
    
    if not significant_pairs:
        print("No significant correlations found!")
        return
    
    # Erstelle Scatterplots
    n_plots = len(significant_pairs)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    for i, (dev_metric, score, results) in enumerate(significant_pairs):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Bereinige Daten
        plot_data = merged_data[[dev_metric, score]].dropna()
        
        # Scatterplot
        plt.scatter(plot_data[score], plot_data[dev_metric], alpha=0.6, s=30)
        
        # Regressionslinie
        z = np.polyfit(plot_data[score], plot_data[dev_metric], 1)
        p = np.poly1d(z)
        plt.plot(plot_data[score], p(plot_data[score]), "r--", alpha=0.8)
        
        # Beschriftung
        plt.xlabel(score.replace('_', ' '), fontsize=10)
        plt.ylabel(dev_metric.replace('_', ' '), fontsize=10)
        plt.title(f'r={results["pearson_r"]:.3f}, p={results["pearson_p"]:.3f}\nn={results["n"]}', 
                 fontsize=11)
        
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/correlations/significant_correlations_scatterplots.png", 
                dpi=300, bbox_inches='tight')
    plt.close()

def analyze_correlations_by_diagnosis(merged_data, score_columns, deviation_metric, save_dir):
    """Analysiert Korrelationen getrennt nach Diagnosegruppen"""
    
    diagnoses = merged_data['Diagnosis_x'].unique()
    
    for diagnosis in diagnoses:
        diag_data = merged_data[merged_data['Diagnosis_x'] == diagnosis]
        
        if len(diag_data) < 10:  # Mindestens 10 Probanden pro Gruppe
            continue
            
        print(f"\n=== Correlations for {diagnosis} (n={len(diag_data)}) ===")
        
        # Erstelle separate Korrelationsmatrix für diese Diagnose
        correlations_diag = {}
        
        if deviation_metric not in diag_data.columns:
            continue
            
        correlations_diag[deviation_metric] = {}
        
        for score in score_columns:
            if score not in diag_data.columns:
                continue
                
            valid_data = diag_data[[deviation_metric, score]].dropna()
            
            if len(valid_data) < 5:
                continue
                
            pearson_r, pearson_p = pearsonr(valid_data[deviation_metric], valid_data[score])
            correlations_diag[deviation_metric][score] = {
                'r': pearson_r,
                'p': pearson_p,
                'n': len(valid_data)
            }
        
        # Speichere Ergebnisse für diese Diagnose
        save_diagnosis_correlations(correlations_diag, diagnosis, score_columns, deviation_metric, save_dir)

def save_diagnosis_correlations(correlations_diag, diagnosis, score_columns, deviation_metric, save_dir):
    """Speichert Korrelationen für eine spezifische Diagnose"""
    
    # Erstelle Arrays für Werte
    correlation_values = []
    p_values = []
    
    for score in score_columns:
        if score in correlations_diag[deviation_metric]:
            correlation_values.append(correlations_diag[deviation_metric][score]['r'])
            p_values.append(correlations_diag[deviation_metric][score]['p'])
        else:
            correlation_values.append(np.nan)
            p_values.append(np.nan)
    
    # Konvertiere zu Matrix für Heatmap
    matrix = np.array(correlation_values).reshape(1, -1)
    p_matrix = np.array(p_values).reshape(1, -1)
    
    # Erstelle Annotations mit Signifikanz
    annotations = []
    for j in range(len(score_columns)):
        if np.isnan(matrix[0, j]):
            annotations.append('')
        else:
            corr_val = matrix[0, j]
            p_val = p_matrix[0, j]
            
            # Bestimme Signifikanz-Level
            if p_val < 0.001:
                stars = '***'
            elif p_val < 0.01:
                stars = '**'
            elif p_val < 0.05:
                stars = '*'
            else:
                stars = ''
            
            annotations.append(f'{corr_val:.3f}{stars}')
    
    # Plot
    plt.figure(figsize=(14, 3))
    mask = np.isnan(matrix)
    sns.heatmap(matrix,
                xticklabels=score_columns,
                yticklabels=[deviation_metric.replace('_', ' ').title()],
                annot=[annotations],
                fmt='',
                cmap='RdBu_r',
                center=0,
                mask=mask,
                square=False,
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title(f'Correlations for {diagnosis}\n(* p<0.05, ** p<0.01, *** p<0.001)', fontsize=14)
    plt.xlabel('Clinical Scores', fontsize=12)
    plt.ylabel('', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/correlations/correlations_{diagnosis.replace('-', '_')}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_correlation_summary_table(correlation_results, save_dir):
    """Erstellt eine Zusammenfassungstabelle aller Korrelationen"""
    
    summary_data = []
    
    for dev_metric, scores in correlation_results.items():
        for score, results in scores.items():
            summary_data.append({
                'Deviation_Metric': dev_metric,
                'Clinical_Score': score,
                'N': results['n'],
                'Pearson_r': results['pearson_r'],
                'Pearson_p': results['pearson_p'],
                'Spearman_r': results['spearman_r'],
                'Spearman_p': results['spearman_p'],
                'Significant_Pearson': results['pearson_p'] < 0.05,
                'Significant_Spearman': results['spearman_p'] < 0.05
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Sortiere nach p-Wert
    summary_df = summary_df.sort_values('Pearson_p')
    
    # Speichere als CSV
    summary_df.to_csv(f"{save_dir}/figures/correlations/correlation_summary_table.csv", index=False)
    
    # Zeige die stärksten Korrelationen
    print("\n=== TOP 10 STRONGEST CORRELATIONS ===")
    print(summary_df.head(10)[['Deviation_Metric', 'Clinical_Score', 'N', 'Pearson_r', 'Pearson_p']])
    
    return summary_df

# HAUPTFUNKTION ZUM AUFRUFEN
def run_correlation_analysis(results_df, save_dir, metadata_path=None, diagnoses_to_include=None):
    """
    Hauptfunktion für die Korrelationsanalyse
    
    Parameters:
    -----------
    results_df : DataFrame
        Deine results DataFrame mit Deviation Scores
    save_dir : str
        Pfad zum Speichern
    metadata_path : str, optional
        Pfad zur Metadaten-CSV (falls nicht schon geladen)
    diagnoses_to_include : list, optional
        Liste der Diagnosen zum Einschließen
    """
    
    # Lade Metadaten wenn Pfad gegeben
    if metadata_path:
        metadata_df = pd.read_csv(metadata_path)
    else:
        # Standardpfad aus deinem Code
        metadata_df = pd.read_csv('/workspace/project/catatonia_VAE-main_bq/metadata_20250110/full_data_with_codiagnosis_and_scores.csv')
    
    # Führe Analyse durch
    correlation_results, merged_data = analyze_score_correlations(
        results_df=results_df,
        metadata_df=metadata_df, 
        save_dir=save_dir,
        diagnoses_to_include=diagnoses_to_include
    )
    
    return correlation_results, merged_data

# BEISPIEL-AUFRUF:
# correlation_results, merged_data = run_correlation_analysis(
#     results_df=your_results_df,
#     save_dir="/path/to/save/directory",
#     diagnoses_to_include=['CTT-SCHZ', 'CTT-MDD']  # Optional: nur diese Diagnosen
# )