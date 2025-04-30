# Beispiel für die Integration der PCNtoolkit-BLR-Analyse mit Ihrem mriContrastVAE-Projekt
# Speichern Sie diese Datei im Hauptverzeichnis Ihres Projekts

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

# Fügen Sie Ihr Projektverzeichnis zum Pythonpfad hinzu
sys.path.append('.')  # Passen Sie an, falls nötig

# Import Ihrer eigenen Module
from models.ContrastVAE_2D_f import ContrastVAE_2D
from utils import load_config, setup_logging #
from src.data_loaders import load_data #

# Import der PCNtoolkit-Integration
from pcn_integration import PCNIntegration  # Speichern Sie die integration_script.py als pcn_integration.py

def parse_args():
    """Kommandozeilenparameter parsen"""
    parser = argparse.ArgumentParser(description='BLR-Analyse mriContrastVAE')
    parser.add_argument('--config', type=str, default='config/blr_config.json',
                        help='Pfad zur Konfigurationsdatei')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Pfad zu den Daten (überschreibt Konfiguration)')
    parser.add_argument('--output_dir', type=str, default='results/blr_analysis',
                        help='Verzeichnis für die Ausgabe der Ergebnisse')
    parser.add_argument('--use_vae_features', action='store_true',
                        help='Verwende Features aus dem VAE statt direkt ROI-Volumen')
    return parser.parse_args()

def main():
    """Hauptfunktion für die BLR-Analyse"""
    # Argumente parsen
    args = parse_args()
    
    # Logging einrichten
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging('blr_analysis', os.path.join(args.output_dir, 'blr_analysis.log'))
    logger.info(f"BLR-Analyse gestartet mit Konfiguration: {args.config}")
    
    # PCNtoolkit-Integration initialisieren
    pcn = PCNIntegration(config_path=args.config, output_dir=args.output_dir)
    
    # Analyse durchführen
    if args.use_vae_features:
        logger.info("Verwende VAE-Features für die Analyse")
        # Hier müssten Sie zusätzliche Schritte implementieren, um die VAE-Features zu extrahieren
        # und anstelle der direkten ROI-Volumina zu verwenden
        
        # Beispiel für den Extraktionsprozess:
        # 1. Laden Sie die Daten
        data_raw = load_data(args.data_path)
        
        # 2. Extrahieren Sie Features mit dem VAE
        features = pcn.extract_vae_features(data_raw)
        
        # 3. Teilen Sie die Daten auf und führen Sie die BLR-Analyse durch
        # ... (Implementieren Sie hier den spezifischen Code für VAE-Features)
        
        logger.info("VAE-Feature-basierte Analyse noch nicht vollständig implementiert")
        print("Die VAE-Feature-basierte Analyse ist noch nicht vollständig implementiert.")
        print("Bitte implementieren Sie die benötigten Funktionen in der pcn_integration.py")
    else:
        logger.info("Führe Standard-BLR-Analyse mit ROI-Volumina durch")
        # Standard-Analyse mit ROI-Volumina
        diff_df, metrics = pcn.run_full_analysis(args.data_path)
        
        # Ausgabe der Ergebnisse
        print("\n--- BLR-Analyse Ergebnisse ---")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MSLL: {metrics['msll']:.4f}")
        print("\nTop 5 ROIs mit den größten Differenzen:")
        for i, row in diff_df.head(5).iterrows():
            print(f"  {row['ROI']}: {row['Mean_Absolute_Difference']:.4f}")
        
        print(f"\nAlle Ergebnisse wurden in {args.output_dir} gespeichert.")
        print(f"Visualisierungen finden Sie in {os.path.join(args.output_dir, 'visualizations')}")

if __name__ == "__main__":
    main()