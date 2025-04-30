# Integration von PCNtoolkit mit mriContrastVAE

Diese Anleitung beschreibt, wie Sie das PCNtoolkit (Predictive Clinical Neuroscience Toolkit) zur Durchführung von Bayesianischer Linearer Regression (BLR) in Ihr mriContrastVAE-Projekt integrieren können. Die Integration ermöglicht es Ihnen, Differenzwerte pro ROI zu analysieren und aussagekräftige Visualisierungen zu erstellen.

## Installation

Bevor Sie beginnen, installieren Sie PCNtoolkit und die erforderlichen Abhängigkeiten:

```bash
pip install pcntoolkit seaborn scikit-learn matplotlib pandas
```

## Dateien und Struktur

Die Integration besteht aus folgenden Hauptdateien:

1. **pcn_integration.py**: Enthält die Hauptklasse `PCNIntegration` zur Verbindung von PCNtoolkit mit Ihrem Projekt
2. **config/blr_config.json**: Konfigurationsdatei für die BLR-Analyse
3. **run_blr_analysis.py**: Beispielskript zur Ausführung der BLR-Analyse

## Konfiguration

Die Datei `blr_config.json` enthält alle erforderlichen Parameter für die Analyse:

- **data_path**: Pfad zu Ihren MRT-Daten
- **trained_model_path**: Pfad zu Ihrem trainierten VAE-Modell
- **model**: Konfiguration für das VAE-Modell (Dimensionen etc.)
- **blr_analysis**: Parameter für die BLR-Analyse
- **roi_columns**: Namen der zu analysierenden ROIs
- **covariate_columns**: Kovariaten für die Analyse (z.B. Alter, Geschlecht)
- **visualization**: Parameter für die Visualisierung der Ergebnisse

## Verwendung

### 1. Dateien in Ihr Projekt kopieren

Kopieren Sie die folgenden Dateien in Ihr Projektverzeichnis:

- `pcn_integration.py` (aus dem bereitgestellten `integration_script`)
- `blr_config.json` (aus dem bereitgestellten `config_file`)
- `run_blr_analysis.py` (aus dem bereitgestellten `usage_example`)

### 2. Konfiguration anpassen

Passen Sie die Konfigurationsdatei `blr_config.json` an Ihre spezifischen Anforderungen an:

- Aktualisieren Sie den `data_path` und `trained_model_path`
- Passen Sie die `model` Parameter an Ihre VAE-Implementierung an
- Überprüfen Sie die `roi_columns`, um sicherzustellen, dass sie mit Ihren Daten übereinstimmen

### 3. BLR-Analyse ausführen

Führen Sie die Analyse mit dem folgenden Befehl aus:

```bash
python run_blr_analysis.py --config config/blr_config.json --output_dir results/blr_analysis
```

Optionen:
- `--data_path`: Überschreibt den Datenpfad aus der Konfigurationsdatei
- `--use_vae_features`: Verwendet Features aus dem VAE statt direkter ROI-Volumen (erfordert zusätzliche Implementierung)

## Anpassung an Ihre Implementierung

Die Integration wurde so entworfen, dass sie mit minimalen Änderungen an Ihrem bestehenden Code funktioniert. Sie müssen jedoch einige Anpassungen vornehmen:

1. **Datenladung**: Passen Sie die `prepare_data` Methode in `PCNIntegration` an Ihre Datenlade-Funktionen an
2. **VAE-Modell**: Stellen Sie sicher, dass die `load_vae_model` und `extract_vae_features` Methoden mit Ihrer VAE-Implementierung kompatibel sind
3. **ROI-Namen**: Aktualisieren Sie die ROI-Namen in der Konfigurationsdatei, um sie an Ihre spezifischen Daten anzupassen

## Ergebnisse und Visualisierungen

Nach der Ausführung der Analyse werden folgende Ergebnisse erzeugt:

1. **ROI-Differenzwerte**: CSV-Dateien mit den Differenzen zwischen vorhergesagten und tatsächlichen ROI-Werten
2. **Visualisierungen**:
   - Balkendiagramm der Top-ROIs mit den größten Abweichungen
   - Streudiagramme für ausgewählte ROIs (Vorhersage vs. Tatsächlich)
   - Heatmap der Korrelationen zwischen ROI-Differenzen
   - Verteilungsplots der Differenzwerte für ausgewählte ROIs
3. **Metriken**: RMSE und MSLL zur Bewertung der Modellqualität

## Erweiterung mit VAE-Features

Um die Analyse mit Features aus Ihrem VAE-Modell statt direkter ROI-Volumen durchzuführen:

1. Implementieren Sie die entsprechenden Methoden in `PCNIntegration` für die Extraktion von VAE-Features
2. Führen Sie die Analyse mit der Option `--use_vae_features` aus
3. Vergleichen Sie die Ergebnisse der beiden Ansätze

## Fehlerbehebung

Wenn Probleme auftreten:

1. Überprüfen Sie die Logdateien im Ausgabeverzeichnis
2. Stellen Sie sicher, dass die Pfade und Konfigurationsparameter korrekt gesetzt sind
3. Überprüfen Sie die Datenformate und -dimensionen
4. Stellen Sie sicher, dass alle erforderlichen Abhängigkeiten installiert sind

## Weiterführende Anpassungen

Sie können die Integration erweitern, indem Sie:

1. Weitere Visualisierungen hinzufügen
2. Andere Algorithmen aus dem PCNtoolkit integrieren (z.B. Gaussian Process Models)
3. Die Analyse für verschiedene Untergruppen von Daten oder ROIs durchführen
4. Die Ergebnisse mit anderen statistischen Tests oder Analysen kombinieren