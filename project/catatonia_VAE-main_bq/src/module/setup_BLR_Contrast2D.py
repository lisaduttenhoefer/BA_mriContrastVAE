#!/bin/bash
# Setup-Skript für die PCNtoolkit-Integration mit mriContrastVAE

# Farben für die Terminalausgabe
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}PCNtoolkit-Integration Setup${NC}"
echo "Dieses Skript richtet die Integration von PCNtoolkit mit Ihrem mriContrastVAE-Projekt ein."
echo ""

# Verzeichnisstruktur erstellen
echo -e "${YELLOW}1. Erstelle Verzeichnisstruktur...${NC}"
mkdir -p config
mkdir -p results/blr_analysis
mkdir -p results/blr_analysis/visualizations
mkdir -p results/blr_analysis/models

echo "   ✓ Verzeichnisstruktur erstellt"
echo ""

# PCNtoolkit und Abhängigkeiten installieren
echo -e "${YELLOW}2. Installiere PCNtoolkit und Abhängigkeiten...${NC}"
pip install pcntoolkit seaborn scikit-learn matplotlib pandas

# Überprüfen, ob die Installation erfolgreich war
if [ $? -eq 0 ]; then
    echo "   ✓ Installation abgeschlossen"
else
    echo -e "${RED}   ✗ Fehler bei der Installation${NC}"
    echo "   Bitte installieren Sie die Pakete manuell:"
    echo "   pip install pcntoolkit seaborn scikit-learn matplotlib pandas"
fi
echo ""

# Dateien kopieren
echo -e "${YELLOW}3. Kopiere Integration-Dateien...${NC}"

# pcn_integration.py erstellen
echo "   Erstelle pcn_integration.py..."
cat > pcn_integration.py << 'EOL'
# PCNtoolkit-Integration für mriContrastVAE
# Diese Datei enthält die Hauptklasse für die Integration

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split

# PCNtoolkit-Import
import pcntoolkit.normative as norm
from pcntoolkit.util.utils import compute_MSLL, create_design_matrix
import pcntoolkit.dataio as pio

# Pfad zu Ihrem Repository anpassen (wenn nötig)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import Ihrer eigenen Module (anpassen nach Bedarf)
try:
    from src.model import ContrastVAE
    # Passen Sie die folgenden Importe an Ihre aktuelle Implementierung an
    from src.utils import load_config, setup_logger
    has_modules = True
except ImportError:
    print("Warnung: Konnte Module nicht importieren. Sie müssen die Implementierung anpassen.")
    has_modules = False

class PCNIntegration:
    """
    Klasse zur Integration von PCNtoolkit-Funktionalität in mriContrastVAE
    """
    def __init__(self, config_path='config/blr_config.json', output_dir='results/blr_analysis'):
        """
        Initialisiert die Integration mit Konfigurationsparametern
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
            output_dir: Verzeichnis für die Ausgabe der Ergebnisse
        """
        self.config_path = config_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Konfiguration laden
        self.config = self._load_config()
        
        # Logger einrichten
        self.logger = self._setup_logger()
        self.logger.info("PCNtoolkit-Integration initialisiert")
        
        # Das VAE-Modell laden, wenn es existiert
        self.vae_model = None
        if has_modules and os.path.exists(self.config.get('trained_model_path', '')):
            self.load_vae_model()
    
    def _load_config(self):
        """Lädt die Konfiguration aus der angegebenen Datei"""
        if has_modules and hasattr(__import__('src.utils', fromlist=['load_config']), 'load_config'):
            # Verwenden Sie Ihre eigene load_config-Funktion
            from src.utils import load_config
            return load_config(self.config_path)
        else:
            # Fallback zu einer einfachen JSON-Ladung
            import json
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Fehler beim Laden der Konfiguration: {str(e)}")
                return {}
    
    def _setup_logger(self):
        """Richtet einen Logger für die Integration ein"""
        if has_modules and hasattr(__import__('src.utils', fromlist=['setup_logger']), 'setup_logger'):
            # Verwenden Sie Ihre eigene setup_logger-Funktion
            from src.utils import setup_logger
            return setup_logger('PCNIntegration', os.path.join(self.output_dir, 'pcn_integration.log'))
        else:
            # Fallback zu einem einfachen Logger
            import logging
            logger = logging.getLogger('PCNIntegration')
            logger.setLevel(logging.INFO)
            
            # File Handler
            fh = logging.FileHandler(os.path.join(self.output_dir, 'pcn_integration.log'))
            fh.setLevel(logging.INFO)
            
            # Console Handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            # Logger konfigurieren
            logger.addHandler(fh)
            logger.addHandler(ch)
            
            return logger
    
    def load_vae_model(self):
        """Lädt das vortrainierte VAE-Modell"""
        try:
            model_path = self.config.get('trained_model_path')
            self.logger.info(f"Lade VAE-Modell von {model_path}")
            
            # Modellparameter aus der Konfiguration
            model_config = self.config.get('model', {})
            
            # Instanziieren des Modells mit den Konfigurationsparametern
            # ANPASSEN: Sie müssen diese Zeilen an Ihre spezifische Implementierung anpassen
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
        
        # ANPASSEN: Hier müssen Sie Ihre eigene Datenladefunktion verwenden
        try:
            if has_modules:
                # Ihre vorhandene Datenladefunktion verwenden
                from src.data_loaders import load_nifti_data
                mri_data, metadata = load_nifti_data(data_path)
            else:
                # Dummy-Implementierung für Tests - ERSETZEN Sie dies mit Ihrem eigenen Code
                self.logger.warning("Verwende Dummy-Daten. Ersetzen Sie dies mit Ihrem eigenen Datenladecode.")
                # Beispieldaten erzeugen
                import numpy as np
                n_samples = 100
                n_rois = len(self.config.get('roi_columns', []))
                if n_rois == 0:
                    n_rois = 45  # Standardwert, falls keine ROI-Spalten definiert sind
                
                mri_data = np.random.randn(n_samples, 128, 128, 128)  # Dummy MRI-Daten
                metadata = pd.DataFrame({
                    'age': np.random.uniform(20, 80, n_samples),
                    'sex': np.random.choice([0, 1], n_samples)
                })
                
                # Dummy ROI-Daten hinzufügen
                roi_names = self.config.get('roi_columns', [f"ROI_{i}" for i in range(n_rois)])
                for roi in roi_names:
                    metadata[roi] = np.random.uniform(800, 1200, n_samples)
            
            # Extrahieren der Feature Maps (ROI-Volumina)
            roi_columns = self.config.get('roi_columns', [])
            if not roi_columns:
                self.logger.warning("Keine ROI-Spalten in der Konfiguration definiert")
                # Versuche alle Spalten außer Kovariaten zu verwenden
                covariate_columns = self.config.get('covariate_columns', ['age', 'sex'])
                if isinstance(metadata, pd.DataFrame):
                    roi_columns = [col for col in metadata.columns if col not in covariate_columns]
                else:
                    self.logger.error("Metadaten sind kein DataFrame und keine ROI-Spalten definiert")
                    raise ValueError("ROI-Spalten müssen definiert sein")
            
            roi_features = metadata[roi_columns].values
            
            # Kovariaten (demographische/klinische Daten)
            covariate_columns = self.config.get('covariate_columns', ['age', 'sex'])
            covariates = metadata[covariate_columns].values
            
            # Train-Test-Split
            X_train, X_test, y_train, y_test = train_test_split(
                covariates, roi_features, 
                test_size=self.config.get('blr_analysis', {}).get('test_size', 0.2), 
                random_state=self.config.get('blr_analysis', {}).get('random_seed', 42)
            )
            
            self.logger.info(f"Daten aufgeteilt: {X_train.shape[0]} Training, {X_test.shape[0]} Test Samples")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Datenvorbereitung: {str(e)}")
            raise
    
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
        
        # ANPASSEN: Feature-Extraktion aus dem VAE
        # Diese Zeilen müssen an Ihre spezifische Implementierung angepasst werden
        try:
            with torch.no_grad():
                # Diese Zeile müssen Sie an Ihre spezifische Implementierung anpassen
                z_mean, z_log_var, z = self.vae_model.encode(data)
                
            return z.numpy()  # Konvertieren zu NumPy für PCNtoolkit
        except Exception as e:
            self.logger.error(f"Fehler bei der VAE-Feature-Extraktion: {str(e)}")
            raise
    
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
        
        # Erstellen der Design-Matrix
        covariate_columns = self.config.get('covariate_columns', ['age', 'sex'])
        # Wenn X_train ein numpy-Array ist, konvertieren wir es zu einem DataFrame mit den richtigen Spaltennamen
        if isinstance(X_train, np.ndarray):
            X_train_df = pd.DataFrame(X_train, columns=covariate_columns[:X_train.shape[1]])
            X_test_df = pd.DataFrame(X_test, columns=covariate_columns[:X_test.shape[1]])
        else:
            X_train_df = X_train
            X_test_df = X_test
            
        # Erstellen der Design-Matrix mit PCNtoolkit
        X_design_train = create_design_matrix(X_train_df, cols=covariate_columns[:X_train.shape[1]])
        X_design_test = create_design_matrix(X_test_df, cols=covariate_columns[:X_test.shape[1]])
        
        # Speichern im PCNtoolkit-Format
        np.savetxt(respfile_train, y_train)
        np.savetxt(respfile_test, y_test)
        np.savetxt(covfile_train, X_design_train)
        np.savetxt(covfile_test, X_design_test)
        
        # BLR-Training und Vorhersage
        outputsub = os.path.join(self.output_dir, 'blr_results')
        wdir = os.path.join(self.output_