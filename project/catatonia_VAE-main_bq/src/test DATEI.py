import os
os.environ["SCIPY_ARRAY_API"] = "1"
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import re

def normalize_spaces(text):
    """
    Normalisiert Leerzeichen und bereinigt Strings für den Abgleich.
    Ersetzt alle nicht-alphanumerischen Zeichen (außer Leerzeichen) durch Leerzeichen,
    ersetzt alle Unicode-Leerzeichen durch ein Standardleerzeichen,
    entfernt führende/nachfolgende Leerzeichen und ersetzt multiple Leerzeichen durch ein einzelnes.
    """
    if isinstance(text, str):
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        normalized_text = text.replace('\xa0', ' ') 
        normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
        return normalized_text.strip()
    return text


def prepare_effect_sizes_dataframe(effect_sizes_file_path, file_type, raw_roi_name_col, effect_size_col, target_atlas_name="neuromorphometrics"):
    """
    Lädt Effektgrößen aus einer Datei, filtert nach Atlasname und bereinigt ROI-Namen.

    Args:
        effect_sizes_file_path (str): Pfad zur Datei (CSV oder Excel) mit den Effektgrößen.
        file_type (str): Der Typ der Datei ('csv' oder 'excel').
        raw_roi_name_col (str): Der ursprüngliche Name der Spalte mit den unbereinigten ROI-Namen.
        effect_size_col (str): Der Name der Spalte mit den Effektgrößen (z.B. 'Cliffs_Delta').
        target_atlas_name (str): Der Atlasname, nach dem gefiltert werden soll (z.B. 'neuromorphometrics').

    Returns:
        pd.DataFrame: Ein bereinigter DataFrame mit den Spalten 'ROI_Name_Cleaned' und dem Effektgrößen-Namen,
                      oder None, wenn ein Fehler auftritt.
    """
    try:
        if file_type == 'csv':
            df = pd.read_csv(effect_sizes_file_path)
        elif file_type == 'excel':
            df = pd.read_excel(effect_sizes_file_path)
        else:
            print(f"Fehler: Ungültiger Dateityp '{file_type}'. Bitte 'csv' oder 'excel' verwenden.")
            return None

        print(f"Roh-Effektgrößen-DataFrame erfolgreich geladen von: {effect_sizes_file_path}")

        if raw_roi_name_col not in df.columns:
            print(f"Fehler: Spalte '{raw_roi_name_col}' nicht im DataFrame gefunden.")
            print(f"Verfügbare Spalten: {df.columns.tolist()}")
            return None
        if effect_size_col not in df.columns:
            print(f"Fehler: Spalte '{effect_size_col}' nicht im DataFrame gefunden.")
            print(f"Verfügbare Spalten: {df.columns.tolist()}")
            return None

        df_filtered = df[df[raw_roi_name_col].astype(str).str.contains(target_atlas_name, na=False, case=False)] 
        print(f"DataFrame nach '{target_atlas_name}' gefiltert. Ursprüngliche Zeilen: {len(df)}, Gefilterte Zeilen: {len(df_filtered)}")

        if df_filtered.empty:
            print(f"Keine Zeilen für den Atlas '{target_atlas_name}' gefunden. Überprüfen Sie den 'raw_roi_name_col' und 'target_atlas_name'.")
            return None

        df_filtered['ROI_Name_Temp'] = df_filtered[raw_roi_name_col].astype(str).apply(
            lambda x: x.split(f"{target_atlas_name}_", 1)[1] if f"{target_atlas_name}_" in x else x
        )
        
        suffixes_to_remove = ['_Vgm', '_Vwm', '_csf'] 
        
        df_filtered['ROI_Name_Cleaned'] = df_filtered['ROI_Name_Temp'].astype(str).apply(
            lambda x: x
        )
        
        for suffix in suffixes_to_remove:
            df_filtered['ROI_Name_Cleaned'] = df_filtered['ROI_Name_Cleaned'].apply(
                lambda x: x.replace(suffix, '') if isinstance(x, str) and x.endswith(suffix) else x
            )
        
        df_filtered['ROI_Name_Cleaned'] = df_filtered['ROI_Name_Cleaned'].apply(normalize_spaces)
        
        cleaned_df = df_filtered[['ROI_Name_Cleaned', effect_size_col]].copy()
        cleaned_df.rename(columns={'ROI_Name_Cleaned': 'ROI_Name'}, inplace=True)

        print(f"ROI-Namen bereinigt und normalisiert. Beispielbereinigung:")
        if not cleaned_df.empty and raw_roi_name_col in df.columns:
            # Für eine konsistente Beispielausgabe: Holen Sie das Beispiel aus dem ursprünglichen DataFrame
            # basierend auf dem Index der ersten Zeile im bereinigten DataFrame
            example_idx = cleaned_df.index[0]
            original_raw_name_for_example = df.loc[example_idx, raw_roi_name_col]
            cleaned_name_for_example = cleaned_df.loc[example_idx, 'ROI_Name']
            print(f"Original: '{original_raw_name_for_example}' -> Bereinigt: '{cleaned_name_for_example}'")
        else:
            print("Keine bereinigten ROIs zum Anzeigen des Beispiels vorhanden (oder Original-DF leer).")

        return cleaned_df

    except FileNotFoundError:
        print(f"Fehler: Die Effektgrößen-Datei wurde nicht gefunden unter: {effect_sizes_file_path}")
        print("Bitte überprüfe den Pfad und stelle sicher, dass die Datei existiert.")
        return None
    except Exception as e:
        print(f"Fehler beim Laden oder Verarbeiten der Effektgrößen-Datei: {e}")
        return None


def plot_brain_with_effect_sizes_neuromorphometrics(
    atlas_nifti_path,
    label_map_xml_path,
    effect_sizes_df_cleaned,
    roi_name_col_cleaned,
    effect_size_col,
    output_filename_prefix="brain_effect_sizes",
    vmax=None,
    cmap='viridis',
    plot_interactive=False): # NEU: Option für interaktiven Plot
    """
    Erstellt 3D-Visualisierungen eines NeuroMorphometrics Gehirnatlas mit eingefärbten ROI-Effektgrößen.
    Erwartet einen bereits bereinigten DataFrame für die Effektgrößen.

    Args:
        atlas_nifti_path (str): Pfad zur NIfTI-Datei deines NeuroMorphometrics Gehirnatlas.
        label_map_xml_path (str): Pfad zur XML-Datei, die die Label-Map des Atlases enthält.
        effect_sizes_df_cleaned (pd.DataFrame): Der bereinigte DataFrame mit den Spalten 'ROI_Name' und Effektgrößen.
        roi_name_col_cleaned (str): Name der Spalte im bereinigten DataFrame, die die ROI-Namen enthält (z.B. 'ROI_Name').
        effect_size_col (str): Name der Spalte im DataFrame, die die Effektgrößen enthält.
        output_filename_prefix (str): Präfix für die Ausgabedatei-Namen (z.B. "brain_effect_sizes").
        vmax (float, optional): Maximalwert für die Farbskala. Wenn None, wird der maximale Effektgröße verwendet.
        cmap (str): Name der Colormap (z.B. 'viridis', 'hot', 'coolwarm').
        plot_interactive (bool): Wenn True, wird ein interaktiver Plot mit nilearn.plotting.view_img() erstellt.
    """

    # 1. NeuroMorphometrics Atlas laden
    try:
        atlas_img = nib.load(atlas_nifti_path)
        atlas_data = atlas_img.get_fdata()
        print(f"NeuroMorphometrics Atlas erfolgreich geladen von: {atlas_nifti_path}")
    except FileNotFoundError:
        print(f"Fehler: Die NIfTI-Datei des Atlases wurde nicht gefunden unter: {atlas_nifti_path}")
        print("Bitte überprüfe den Pfad und stelle sicher, dass die Datei existiert.")
        return
    except Exception as e:
        print(f"Fehler beim Laden des NeuroMorphometrics Atlases: {e}")
        return

    # 2. Label-Map XML-Datei parsen und Leerzeichen normalisieren
    try:
        tree = ET.parse(label_map_xml_path)
        root = tree.getroot()
        print(f"Label-Map XML-Datei erfolgreich geladen von: {label_map_xml_path}")
    except FileNotFoundError:
        print(f"Fehler: Die Label-Map XML-Datei wurde nicht gefunden unter: {label_map_xml_path}")
        print("Bitte überprüfe den Pfad und stelle sicher, dass die Datei existiert.")
        return
    except Exception as e:
        print(f"Fehler beim Parsen der Label-Map XML-Datei: {e}")
        return

    label_map = {}
    for label in root.findall('Label'):
        name = normalize_spaces(label.find('Name').text)
        number = int(label.find('Number').text)
        label_map[name] = number
    print("ROI-Namen aus Label-Map normalisiert.")

    # 3. Ein leeres Gehirnvolumen für die Effektgrößen erstellen
    effect_map_data = np.zeros_like(atlas_data, dtype=float)

    # 4. Effektgrößen den ROI-Nummern zuordnen und in das neue Volumen schreiben
    found_rois = []
    missing_rois = []
    
    if roi_name_col_cleaned not in effect_sizes_df_cleaned.columns or effect_size_col not in effect_sizes_df_cleaned.columns:
        print(f"Fehler: Die erforderlichen Spalten '{roi_name_col_cleaned}' oder '{effect_size_col}'")
        print("wurden im bereitgestellten DataFrame nicht gefunden. Überprüfen Sie die prepare_function.")
        return

    for index, row in effect_sizes_df_cleaned.iterrows():
        roi_name = str(row[roi_name_col_cleaned])
        effect_size = row[effect_size_col]

        if roi_name in label_map:
            roi_number = label_map[roi_name]
            if np.any(atlas_data == roi_number):
                effect_map_data[atlas_data == roi_number] = effect_size
                found_rois.append(roi_name)
            else:
                missing_rois.append(f"{roi_name} (ROI-Nummer {roi_number} nicht im Atlas gefunden)")
        else:
            missing_rois.append(f"{roi_name} (Nicht in Label-Map gefunden)")

    if missing_rois:
        print(f"\nWarnung: Folgende ROIs aus Ihrem bereinigten DataFrame wurden NICHT im Atlas / in der Label-Map gefunden und nicht geplottet:")
        for roi in missing_rois:
            print(f"- {roi}")
    if found_rois:
        print(f"\nInfo: {len(found_rois)} ROIs aus Ihrem bereinigten DataFrame wurden im Atlas gefunden und werden geplottet.")
    else:
        print("\nKeine der bereinigten ROIs aus Ihrem DataFrame wurde im Atlas gefunden. Bitte überprüfen Sie die ROI-Namen und die Atlas-Nummern.")
        return

    effect_map_img = nib.Nifti1Image(effect_map_data, atlas_img.affine, atlas_img.header)

    if vmax is None:
        max_val = np.max(np.abs(effect_map_data))
        if max_val == 0:
            vmax = 1.0
            print("Info: Alle Effektgrößen sind Null. vmax wurde auf 1.0 gesetzt.")
        else:
            vmax = max_val
            print(f"\nvmax wurde automatisch auf den Maximalwert der absoluten Effektgrößen gesetzt: {vmax:.2f}")

    # Standardmäßige statische Plots (Glass Brain und Orthogonal)
    print("\nErstelle statische Glass-Brain Visualisierung...")
    fig_glass = plotting.plot_glass_brain(
        effect_map_img,
        display_mode='lzr',
        colorbar=True,
        cmap=cmap,
        vmax=vmax,
        title="Gehirn-Effektgrößen (Glass Brain)",
        plot_abs=False
    )
    # glass_output_path = f"{output_filename_prefix}_glass.png"
    # fig_glass.save_img(glass_output_path)
    # print(f"Glass-Brain Visualisierung gespeichert unter: {glass_output_path}")

    print("Erstelle statische orthogonale Visualisierung...")
    fig_ortho = plotting.plot_stat_map(
        effect_map_img,
        bg_img=atlas_img,
        display_mode='ortho',
        colorbar=True,
        cmap=cmap,
        vmax=vmax,
        title="Gehirn-Effektgrößen (Orthogonal)",
        output_file=None
    )
    # ortho_output_path = f"{output_filename_prefix}_ortho.png"
    # fig_ortho.save_img(ortho_output_path)
    # print(f"Orthogonale Visualisierung gespeichert unter: {ortho_output_path}")

    plt.show() # Zeigt die statischen Plots in Jupyter

    # NEU: Interaktiver Plot mit view_img
    if plot_interactive:
        print("\nErstelle interaktive 3D-Visualisierung (öffnet im Browser oder Inline-Fenster)...")
        # bg_img=atlas_img überlagert den Effekt auf die Anatomie
        interactive_plot = plotting.view_img(
            effect_map_img,
            bg_img=atlas_img,
            colorbar=True,
            cmap=cmap,
            vmax=vmax,
            title="Interaktive Effektgrößen-Karte"
        )
        # interactive_plot.open_in_browser() # Optional: Öffnet den Plot in einem neuen Browser-Tab
        print("Interaktiver Plot generiert. Eventuell musst du das Ausgabefenster des Notebooks scrollen.")
        return interactive_plot # Gib das Plot-Objekt zurück, um es anzuzeigen


# Beispiel-Anwendung
if __name__ == '__main__':
    neuro_morphometrics_atlas_path = "/workspace/project/catatonia_VAE-main_bq/data/atlases_niis/neuro_neu.nii" # Dein aktueller Atlas-Pfad
    
    try:
        test_atlas_img = nib.load(neuro_morphometrics_atlas_path)
        print(f"Verwende vorhandene Atlas-Datei: {neuro_morphometrics_atlas_path}")
    except FileNotFoundError:
        print(f"Fehler: Atlas-Datei '{neuro_morphometrics_atlas_path}' nicht gefunden.")
        print("Bitte stelle sicher, dass der Pfad korrekt ist oder erstelle einen Dummy-Atlas für Tests.")

    my_actual_effect_sizes_csv_path = "/workspace/project/catatonia_VAE-main_bq/analysis/TESTING/deviation_results_norm_results_HC_0.7_all_20250521_0641_20250526_090143/regional_effect_sizes_vs_HC.csv"
    label_map_xml_path = '/workspace/project/catatonia_VAE-main_bq/data/atlases_niis/atlases_labels/1103_3_glm_LabelMap.xml'

    print("\n--- Vorverarbeitung des Effektgrößen-DataFrames ---")
    effect_sizes_cleaned_df = prepare_effect_sizes_dataframe(
        effect_sizes_file_path=my_actual_effect_sizes_csv_path,
        file_type='csv',
        raw_roi_name_col='ROI_Name',
        effect_size_col='Cliffs_Delta',
        target_atlas_name="neuromorphometrics"
    )

    if effect_sizes_cleaned_df is not None and not effect_sizes_cleaned_df.empty:
        print("\n--- Starte Plot-Funktion für NeuroMorphometrics Atlas ---")
        # NEU: plot_interactive=True hinzufügen, um den interaktiven Plot zu aktivieren
        interactive_viewer = plot_brain_with_effect_sizes_neuromorphometrics(
            atlas_nifti_path=neuro_morphometrics_atlas_path,
            label_map_xml_path=label_map_xml_path,
            effect_sizes_df_cleaned=effect_sizes_cleaned_df,
            roi_name_col_cleaned='ROI_Name',
            effect_size_col='Cliffs_Delta',
            output_filename_prefix="neuromorphometrics_effect_sizes_processed",
            cmap='coolwarm',
            vmax=0.3,
            plot_interactive=True # <--- Setze dies auf True für den interaktiven Plot
        )
        if interactive_viewer:
            interactive_viewer.save_as_html("/workspace/project/catatonia_VAE-main_bq/interactive_brain_plot.html") # Optional: Speichern als HTML
            print("YUP IT WORKED")
            # interactive_viewer # Dies würde den Plot im Jupyter Notebook anzeigen, wenn der letzte Ausdruck in einer Zelle ein Plot-Objekt ist
            
        print("\nPlot-Funktion abgeschlossen.")
    else:
        print("\nPlotting übersprungen, da es Probleme bei der Vorbereitung des Effektgrößen-DataFrames gab.")