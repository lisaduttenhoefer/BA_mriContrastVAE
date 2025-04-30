# Integration von PCNtoolkit mit mriContrastVAE
# Dieses Skript zeigt, wie Sie die PCNtoolkit-Funktionalität in Ihre bestehende Pipeline integrieren können

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split

# PCNtoolkit und verwandte Bibliotheken installieren (falls noch nicht vorhanden)
# !pip install pcntoolkit seaborn scikit-learn

# PCNtoolkit-Import
import pcntoolkit.normative as norm
from pcntoolkit.util.utils import compute_MSLL, create_design_matrix
import pcntoolkit.dataio as pio

# Pfad zu Ihrem Repository anpassen
sys.path.append('/path/to/BA_mriContrastVAE')

# Import Ihrer eigenen Module
from src.model import ContrastVAE
# Passen Sie die folgenden Importe an Ihre aktuelle Implementierung an
from src.utils import load_config, setup_logger
from src.visualization import plot_results
from src.data_loaders import load_nifti_data, prepare_dataset

class PCNIntegration:
    """
    Klasse zur Integration von PCNtoolkit-Funktionalität in mriContrastVAE
    """
    def __init__(self, config_path='config/default.json', output_dir='results/blr_analysis'):
        """
        Initialisiert die Integration mit Konfigurationsparametern
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
            output_dir: Verzeichnis für die Ausgabe der Ergebnisse
        """
        self.config = load_config(config_path)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Logger einrichten
        self.logger = setup_logger('PCNIntegration', os.path.join(output_dir, 'pcn_integration.log'))
        self.logger.info("PCNtoolkit-Integration initialisiert")
        
        # Das VAE-Modell laden, wenn es existiert
        self.vae_model = None
        if os.path.exists(self.config.get('trained_model_path', '')):
            self.load_vae_model()
    
    def load_vae_model(self):
        """Lädt das vortrainierte VAE-Modell"""
        try:
            model_path = self.config.get('trained_model_path')
            self.logger.info(f"Lade VAE-Modell von {model_path}")
            
            # Modellparameter aus der Konfiguration
            model_config = self.config.get('model', {})
            
            # Instanziieren des Modells mit den Konfigurationsparametern
            self.vae_model = ContrastVAE(
                input_dim=model_config.get('input_dim', 128),
                hidden_dim=model_config.get('hidden_dim', 64),
                latent_dim=model_config.get('latent_dim', 16),
                contrast_dim=model_config.get('contrast_dim', 2)
            )
            
            # Laden der Modellgewichte
            self.vae_model.load_state_dict(torch.load(model_path))
            self.vae_model.eval()
            self.logger.info("VAE-Modell erfolgreich geladen")
        except Exception as e:
            self.logger.error(f"Fehler beim Laden des VAE-Modells: {str(e)}")
            raise
    
    def prepare_data(self, data_path=None):
        """
        Bereitet die Daten für die BLR-Analyse vor
        
        Args:
            data_path: Pfad zu den Daten oder None, um den Pfad aus der Konfiguration zu verwenden
        
        Returns:
            X_train, X_test, y_train, y_test: Aufgeteilte Daten für Training und Test
        """
        if data_path is None:
            data_path = self.config.get('data_path')
        
        self.logger.info(f"Lade Daten von {data_path}")
        
        # Ihre vorhandene Datenladefunktion verwenden
        # Hier müssen Sie möglicherweise Ihre eigene Funktion anpassen
        mri_data, metadata = load_nifti_data(data_path)
        
        # Extrahieren der Feature Maps (ROI-Volumina)
        # Anpassen an Ihre spezifische Datenstruktur
        roi_features = metadata[self.config.get('roi_columns', [])]
        
        # Kovariaten (demographische/klinische Daten)
        covariates = metadata[self.config.get('covariate_columns', ['age', 'sex'])]
        
        # Train-Test-Split
        X_train, X_test, y_train, y_test = train_test_split(
            covariates, roi_features, 
            test_size=self.config.get('test_size', 0.2), 
            random_state=self.config.get('random_seed', 42)
        )
        
        self.logger.info(f"Daten aufgeteilt: {X_train.shape[0]} Training, {X_test.shape[0]} Test Samples")
        
        return X_train, X_test, y_train, y_test
    
    def extract_vae_features(self, data):
        """
        Extrahiert Features aus dem vortrainierten VAE
        
        Args:
            data: Eingabedaten für das VAE
        
        Returns:
            Extrahierte Features aus dem Latent Space
        """
        if self.vae_model is None:
            self.logger.warning("VAE-Modell nicht geladen, lade es jetzt")
            self.load_vae_model()
        
        self.logger.info("Extrahiere Features aus VAE")
        
        # Konvertieren zu Tensor falls notwendig
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        
        # Feature-Extraktion aus dem VAE
        with torch.no_grad():
            # Diese Zeile müssen Sie an Ihre spezifische Implementierung anpassen
            z_mean, z_log_var, z = self.vae_model.encode(data)
            
        return z.numpy()  # Konvertieren zu NumPy für PCNtoolkit
    
    def run_blr_analysis(self, X_train, X_test, y_train, y_test):
        """
        Führt die Bayesianische Lineare Regression mit PCNtoolkit durch
        
        Args:
            X_train, X_test: Kovariaten für Training und Test
            y_train, y_test: ROI-Features für Training und Test
        
        Returns:
            yhat_test: Vorhergesagte Werte
            metrics: Evaluierungsmetriken
        """
        self.logger.info("Starte BLR-Analyse mit PCNtoolkit")
        
        # Dateipfade für PCNtoolkit
        respfile_train = os.path.join(self.output_dir, 'resp_train.txt')
        respfile_test = os.path.join(self.output_dir, 'resp_test.txt')
        covfile_train = os.path.join(self.output_dir, 'cov_train.txt')
        covfile_test = os.path.join(self.output_dir, 'cov_test.txt')
        
        # Erstellen der Design-Matrix (Anpassen an Ihre Kovariaten)
        X_design_train = create_design_matrix(X_train, 
                                             cols=self.config.get('covariate_columns', ['age', 'sex']))
        X_design_test = create_design_matrix(X_test, 
                                            cols=self.config.get('covariate_columns', ['age', 'sex']))
        
        # Speichern im PCNtoolkit-Format
        np.savetxt(respfile_train, y_train)
        np.savetxt(respfile_test, y_test)
        np.savetxt(covfile_train, X_design_train)
        np.savetxt(covfile_test, X_design_test)
        
        # BLR-Training und Vorhersage
        outputsub = os.path.join(self.output_dir, 'blr_results')
        wdir = os.path.join(self.output_dir, 'models')
        
        # BLR-Algorithmus von PCNtoolkit ausführen
        norm.estimate(covfile_train, respfile_train, 
                     testresp=respfile_test, testcov=covfile_test,
                     alg='blr', configparam=None, saveoutput=True, 
                     outputsub=outputsub, warp=None)
        
        # Laden der Ergebnisse
        yhat_test = pio.load_normative_prediction(outputsub)[0]
        
        # Berechnung der Metriken
        metrics = {}
        metrics['rmse'] = np.sqrt(np.mean((y_test - yhat_test) ** 2))
        metrics['msll'] = compute_MSLL(y_test, yhat_test, y_train, covfile_train, 
                                      covfile_test, configparam=None)
        
        self.logger.info(f"BLR-Analyse abgeschlossen: RMSE={metrics['rmse']:.4f}, MSLL={metrics['msll']:.4f}")
        
        return yhat_test, metrics
    
    def analyze_roi_differences(self, y_true, y_pred, roi_names=None):
        """
        Analysiert die Differenzwerte zwischen vorhergesagten und tatsächlichen ROI-Werten
        
        Args:
            y_true: Tatsächliche ROI-Werte
            y_pred: Vorhergesagte ROI-Werte
            roi_names: Namen der ROIs (optional)
        
        Returns:
            diff_df: DataFrame mit Differenzwerten
        """
        self.logger.info("Analysiere ROI-Differenzen")
        
        # Differenzwerte berechnen
        diff_values = y_true - y_pred
        
        # ROI-Namen überprüfen
        if roi_names is None or len(roi_names) != diff_values.shape[1]:
            self.logger.warning("ROI-Namen nicht spezifiziert oder stimmen nicht mit Daten überein")
            roi_names = [f"ROI_{i}" for i in range(diff_values.shape[1])]
        
        # DataFrame erstellen
        diff_matrix = pd.DataFrame(diff_values, columns=roi_names)
        
        # Mittlere absolute Differenz pro ROI
        mean_abs_diff = np.mean(np.abs(diff_values), axis=0)
        diff_df = pd.DataFrame({
            'ROI': roi_names,
            'Mean_Absolute_Difference': mean_abs_diff
        })
        
        # Sortieren für bessere Visualisierung
        diff_df = diff_df.sort_values('Mean_Absolute_Difference', ascending=False)
        
        # DataFrame mit Differenzwerten speichern
        diff_df.to_csv(os.path.join(self.output_dir, 'roi_differences.csv'), index=False)
        diff_matrix.to_csv(os.path.join(self.output_dir, 'all_differences.csv'), index=False)
        
        self.logger.info(f"ROI-Differenzen in {self.output_dir} gespeichert")
        
        return diff_df, diff_matrix
    
    def visualize_results(self, y_true, y_pred, diff_df, diff_matrix, roi_names=None):
        """
        Erstellt Visualisierungen für die BLR-Analyse und ROI-Differenzen
        
        Args:
            y_true, y_pred: Tatsächliche und vorhergesagte Werte
            diff_df: DataFrame mit mittleren Differenzen pro ROI
            diff_matrix: Matrix mit allen Differenzwerten
            roi_names: Namen der ROIs (optional)
        """
        self.logger.info("Erstelle Visualisierungen")
        
        # Verzeichnis für Visualisierungen
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. Balkendiagramm der mittleren absoluten Differenz pro ROI
        plt.figure(figsize=(12, 8))
        plt.barh(diff_df['ROI'][:15], diff_df['Mean_Absolute_Difference'][:15])
        plt.xlabel('Mittlere absolute Differenz')
        plt.title('Top 15 ROIs mit den größten Abweichungen')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'top_roi_differences.png'))
        
        # 2. Streudiagramm: Vorhergesagt vs. Tatsächlich für einige ausgewählte ROIs
        top_rois = diff_df['ROI'][:5].tolist()  # Top 5 ROIs mit größten Differenzen
        
        plt.figure(figsize=(15, 10))
        for i, roi in enumerate(top_rois):
            roi_idx = list(diff_matrix.columns).index(roi)
            plt.subplot(2, 3, i+1)
            plt.scatter(y_true[:, roi_idx], y_pred[:, roi_idx], alpha=0.7)
            plt.plot([y_true[:, roi_idx].min(), y_true[:, roi_idx].max()], 
                    [y_true[:, roi_idx].min(), y_true[:, roi_idx].max()], 'r--')
            plt.xlabel('Tatsächlicher Wert')
            plt.ylabel('Vorhergesagter Wert')
            plt.title(f'ROI: {roi}')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'prediction_scatter.png'))
        
        # 3. Heatmap der Korrelation zwischen ROI-Differenzen
        plt.figure(figsize=(14, 12))
        corr_matrix = diff_matrix.corr()
        
        # Erstellen Sie eine Maske für die obere Dreiecksmatrix
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Heatmap mit Seaborn erstellen
        import seaborn as sns
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', 
                   annot=False, vmin=-1, vmax=1, square=True)
        plt.title('Korrelation der ROI-Differenzwerte')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'roi_diff_correlation.png'))
        
        # 4. Verteilung der Differenzwerte für ausgewählte ROIs
        plt.figure(figsize=(14, 8))
        for i, roi in enumerate(top_rois):
            plt.subplot(1, 5, i+1)
            sns.histplot(diff_matrix[roi], kde=True)
            plt.title(f'ROI: {roi}')
            plt.xlabel('Differenz')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'roi_diff_distribution.png'))
        
        self.logger.info(f"Visualisierungen in {vis_dir} gespeichert")
    
    def run_full_analysis(self, data_path=None):
        """
        Führt die vollständige Analyse durch: Datenaufbereitung, BLR und Visualisierung
        
        Args:
            data_path: Pfad zu den Daten (optional)
        
        Returns:
            diff_df: DataFrame mit ROI-Differenzwerten
            metrics: Evaluierungsmetriken
        """
        # 1. Daten vorbereiten
        X_train, X_test, y_train, y_test = self.prepare_data(data_path)
        
        # 2. BLR-Analyse durchführen
        y_pred, metrics = self.run_blr_analysis(X_train, X_test, y_train, y_test)
        
        # 3. ROI-Differenzen analysieren
        # Verwenden Sie hier die tatsächlichen ROI-Namen aus Ihren Daten
        roi_names = self.config.get('roi_names', [f"ROI_{i}" for i in range(y_test.shape[1])])
        diff_df, diff_matrix = self.analyze_roi_differences(y_test, y_pred, roi_names)
        
        # 4. Ergebnisse visualisieren
        self.visualize_results(y_test, y_pred, diff_df, diff_matrix, roi_names)
        
        # Ausgabe der Ergebniszusammenfassung
        self.logger.info("Vollständige Analyse abgeschlossen")
        self.logger.info(f"Metriken: RMSE={metrics['rmse']:.4f}, MSLL={metrics['msll']:.4f}")
        
        # Top 5 ROIs mit den größten Abweichungen
        top5 = diff_df.head(5)
        self.logger.info(f"Top 5 ROIs mit den größten Abweichungen:")
        for i, row in top5.iterrows():
            self.logger.info(f"  {row['ROI']}: {row['Mean_Absolute_Difference']:.4f}")
        
        return diff_df, metrics


# Beispiel für die Verwendung der Integration
if __name__ == "__main__":
    # Initializing the integration
    pcn_integration = PCNIntegration(
        config_path='config/blr_config.json',  # Pfad zu Ihrer Konfigurationsdatei
        output_dir='results/blr_analysis'      # Ausgabeverzeichnis
    )
    
    # Ausführen der vollständigen Analyse
    diff_df, metrics = pcn_integration.run_full_analysis()
    
    print("Analyse abgeschlossen. Ergebnisse wurden gespeichert.")
    print(f"RMSE: {metrics['rmse']:.4f}, MSLL: {metrics['msll']:.4f}")