# BLR-Anpassung für mriContrastVAE
# Diese Datei zeigt, wie Sie das PCNtoolkit BLR-Protocol mit Ihrem mriContrastVAE-Modell verwenden können

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# PCNtoolkit Importe
import pcntoolkit.normative as norm
from pcntoolkit.util.utils import compute_MSLL, create_design_matrix
import pcntoolkit.dataio as pio

# Importieren Sie Ihre eigenen Module aus dem mriContrastVAE Repository
sys.path.append('/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq')  # Passen Sie den Pfad an
from src.models import ContrastVAE_2D_f  # Ihr VAE-Modell
from src.module.data_processing_hc import load_mri_data_2D, preprocess_data  # Ihre Datenlade-Funktionen

# 1. Daten vorbereiten - Integration der Feature Maps aus Ihrem VAE Modell
def prepare_data_for_blr():
    """
    Bereitet die Feature Maps aus dem VAE-Modell für die Bayesianische Lineare Regression vor
    """
    # Daten laden (angepasst aus Ihrem data_loaders.py)
    data = load_mri_data_2D()  # Verwenden Sie Ihre eigene Ladefunktion
    
    # Extrahieren Sie die relevanten ROI-Volumenwerte
    roi_features = data['roi_volumes']  # Oder eine ähnliche Struktur aus Ihren Daten
    
    # Demographische/klinische Informationen, die als Kovariaten dienen
    covariates = data['demographics']  # Alter, Geschlecht, etc.
    
    # Teilen Sie die Daten in Trainings- und Testsets auf
    X_train, X_test, y_train, y_test = train_test_split(
        covariates, roi_features, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

# 2. Feature Extraktion aus Ihrem VAE
def extract_vae_features(data, model_path):
    """
    Extrahiert Features aus Ihrem trainierten VAE-Modell
    """
    # Laden Sie Ihr vortrainiertes VAE-Modell
    vae_model = ContrastVAE_2D_f()  # Passen Sie die Parameter entsprechend an
    vae_model.load_state_dict(torch.load(model_path))
    vae_model.eval()
    
    # Daten durch das VAE laufen lassen und Latent Space Features extrahieren
    with torch.no_grad():
        # Diese Zeile muss an Ihre spezifische Implementierung angepasst werden
        z_mean, z_log_var, z = vae_model.encode(data)
        
    return z  # Latent Features

# 3. Bayesianische Lineare Regression mit PCNtoolkit
def run_blr_analysis(X_train, X_test, y_train, y_test, output_dir='./results'):
    """
    Führt die Bayesianische Lineare Regression mit PCNtoolkit durch
    """
    # Erstellen Sie das Ausgabeverzeichnis, falls es nicht existiert
    os.makedirs(output_dir, exist_ok=True)
    
    # Design-Matrix erstellen (PCNtoolkit-Format)
    X_design_train = create_design_matrix(X_train, cols=['Age', 'Sex'])  # Anpassen an Ihre Kovariaten
    X_design_test = create_design_matrix(X_test, cols=['Age', 'Sex'])
    
    # Training des Bayesianischen linearen Modells
    respfile_train = os.path.join(output_dir, 'resp_train.txt')
    respfile_test = os.path.join(output_dir, 'resp_test.txt')
    covfile_train = os.path.join(output_dir, 'cov_train.txt')
    covfile_test = os.path.join(output_dir, 'cov_test.txt')
    
    # Speichern der Daten im PCNtoolkit-Format
    np.savetxt(respfile_train, y_train)
    np.savetxt(respfile_test, y_test)
    np.savetxt(covfile_train, X_design_train)
    np.savetxt(covfile_test, X_design_test)
    
    # BLR-Modell trainieren und testen
    outputsub = os.path.join(output_dir, 'blr_results')
    
    # Diese Zeile führt die eigentliche BLR mit PCNtoolkit durch
    wdir = os.path.join(output_dir, 'models')
    norm.estimate(covfile_train, respfile_train, 
                 testresp=respfile_test, testcov=covfile_test,
                 alg='blr', configparam=None, saveoutput=True, 
                 outputsub=outputsub, warp=None)
    
    # Laden der Ergebnisse
    yhat_test = pio.load_normative_prediction(outputsub)[0]
    
    # Berechnung der Modellgüte
    rmse = np.sqrt(np.mean((y_test - yhat_test) ** 2))
    msll = compute_MSLL(y_test, yhat_test, y_train, covfile_train, 
                        covfile_test, configparam=None)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MSLL: {msll:.4f}")
    
    return yhat_test

# 4. Visualisierung der Differenzwerte pro ROI
def visualize_roi_differences(y_true, y_pred, roi_names, output_dir='./results'):
    """
    Visualisiert die Differenzwerte zwischen vorhergesagten und tatsächlichen ROI-Werten
    """
    # Differenzwerte berechnen
    diff_values = y_true - y_pred
    
    # Dataframe für einfachere Visualisierung erstellen
    if len(roi_names) != diff_values.shape[1]:
        print("Warnung: Anzahl der ROI-Namen stimmt nicht mit den Daten überein")
        roi_names = [f"ROI_{i}" for i in range(diff_values.shape[1])]
    
    # Mittlere absolute Differenz pro ROI
    mean_abs_diff = np.mean(np.abs(diff_values), axis=0)
    diff_df = pd.DataFrame({
        'ROI': roi_names,
        'Mean_Absolute_Difference': mean_abs_diff
    })
    
    # Sortieren für eine bessere Visualisierung
    diff_df = diff_df.sort_values('Mean_Absolute_Difference', ascending=False)
    
    # 1. Balkendiagramm der mittleren absoluten Differenz pro ROI
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Mean_Absolute_Difference', y='ROI', data=diff_df)
    plt.title('Mittlere absolute Differenz pro ROI')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roi_mean_abs_diff.png'))
    
    # 2. Heatmap der Differenzwerte für alle Testfälle
    plt.figure(figsize=(14, 10))
    diff_matrix = pd.DataFrame(diff_values, columns=roi_names)
    sns.heatmap(diff_matrix.corr(), annot=False, cmap='coolwarm')
    plt.title('Korrelation der Differenzwerte zwischen ROIs')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roi_diff_correlation.png'))
    
    # 3. Boxplot der Differenzwerte pro ROI
    plt.figure(figsize=(14, 8))
    melted_df = pd.melt(diff_matrix, var_name='ROI', value_name='Differenz')
    sns.boxplot(x='ROI', y='Differenz', data=melted_df)
    plt.xticks(rotation=90)
    plt.title('Verteilung der Differenzwerte pro ROI')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roi_diff_boxplot.png'))
    
    return diff_df

# Hauptfunktion
def main():
    # Pfad zu Ihrem trainierten VAE-Modell
    vae_model_path = '/path/to/your/vae_model.pth'  # Passen Sie den Pfad an
    
    # 1. Daten vorbereiten
    X_train, X_test, y_train, y_test = prepare_data_for_blr()
    
    # 2. Optional: Feature Extraktion aus Ihrem VAE
    # Wenn Sie bereits Features haben, können Sie diesen Schritt überspringen
    # train_features = extract_vae_features(train_data, vae_model_path)
    # test_features = extract_vae_features(test_data, vae_model_path)
    
    # 3. BLR-Analyse durchführen
    predicted_values = run_blr_analysis(X_train, X_test, y_train, y_test)
    
    # 4. Differenzwerte visualisieren
    # Liste der ROI-Namen (passen Sie diese an Ihre eigenen Daten an)
    roi_names = [f"ROI_{i}" for i in range(y_test.shape[1])]  # Platzhalter
    
    diff_df = visualize_roi_differences(y_test, predicted_values, roi_names)
    
    print("Analyse abgeschlossen. Ergebnisse wurden im Verzeichnis './results' gespeichert.")
    
    return diff_df

if __name__ == "__main__":
    main()
