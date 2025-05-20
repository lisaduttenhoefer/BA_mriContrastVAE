def extract_measurements(subjects):
    """
    Extrahiere die Messwerte aus einer Liste von Subjekten.
    Modifiziert, um das richtige Format für das Modell zu erzeugen.
    
    Args:
        subjects: Liste von subject-Dictionaries mit 'measurements'-Feld
        
    Returns:
        torch.Tensor: Tensor mit allen Messwerten in korrekter Form
    """
    print(f"[DEBUG] extract_measurements: Eingabe hat {len(subjects)} Subjekte")
    
    if len(subjects) == 0:
        print("[WARNUNG] extract_measurements: Leere Subjektliste!")
        return torch.tensor([])
    
    # Prüfe das erste Element zur Diagnose
    first_subj = subjects[0]
    if 'measurements' not in first_subj:
        print(f"[FEHLER] Subjekt hat kein 'measurements'-Feld! Verfügbare Schlüssel: {first_subj.keys()}")
        raise KeyError("'measurements' nicht in Subjekt vorhanden")
    
    first_measurements = first_subj['measurements']
    print(f"[DEBUG] Typ des ersten measurements: {type(first_measurements)}")
    
    if isinstance(first_measurements, list):
        print(f"[DEBUG] Länge der measurements-Liste im ersten Subjekt: {len(first_measurements)}")
        # Wenn list of lists, näheres untersuchen
        if len(first_measurements) > 0 and isinstance(first_measurements[0], list):
            print(f"[DEBUG] Innere Liste Länge: {len(first_measurements[0])}")
    
    # Sammle alle Messwerte
    all_measurements = []
    
    for i, subject in enumerate(subjects):
        try:
            measurements = subject['measurements']
            
            # Prüfe Typ und konvertiere wenn nötig
            if isinstance(measurements, list):
                # Prüfe auf verschachtelte Listen und ihre Dimensionen
                if len(measurements) > 0 and isinstance(measurements[0], list):
                    # Verschachtelte Liste: [ROIs][features] -> umformen zu flacher Liste
                    # Das Modell erwartet [samples, features], nicht [samples, ROIs, features]
                    # Reshape: von [ROIs, features] zu [features*ROIs]
                    print(f"[DEBUG] Subjekt {i}: Reshaping von [ROIs, features] zu flacher Form...")
                    
                    # Option 1: Direkt Flatten der 2D-Liste zu 1D
                    flat_measurements = []
                    for roi_features in measurements:
                        flat_measurements.extend(roi_features)
                    measurements = torch.tensor(flat_measurements, dtype=torch.float32)
                    
                    # Option 2: Behalte 2D-Struktur bei, wandle sie nur um
                    # measurements = torch.tensor(measurements, dtype=torch.float32)
                    # measurements = measurements.view(-1)  # Flatten zu 1D
                else:
                    measurements = torch.tensor(measurements, dtype=torch.float32)
            elif isinstance(measurements, np.ndarray):
                measurements = torch.from_numpy(measurements).float()
                # Flatten wenn mehrdimensional
                if len(measurements.shape) > 1:
                    measurements = measurements.reshape(-1)
            elif isinstance(measurements, torch.Tensor):
                # Flatten wenn mehrdimensional
                if len(measurements.shape) > 1:
                    measurements = measurements.reshape(-1)
            else:
                raise TypeError(f"Unerwarteter Typ für measurements: {type(measurements)}")
            
            all_measurements.append(measurements)
            
            # Diagnostik bei ersten paar Subjekten
            if i < 3:
                print(f"[DEBUG] Subjekt {i}: measurements shape nach Umformung: {measurements.shape}")
            
        except Exception as e:
            print(f"[FEHLER] Bei Subjekt {i}: {e}")
            raise
    
    # Prüfe ob alle Formen konsistent sind
    first_shape = all_measurements[0].shape
    inconsistent_shapes = [i for i, m in enumerate(all_measurements) if m.shape != first_shape]
    if inconsistent_shapes:
        print(f"[WARNUNG] Inkonsistente Formen gefunden bei Subjekten: {inconsistent_shapes}")
        for i in inconsistent_shapes[:5]:  # Zeige nur die ersten 5
            print(f"  Subjekt {i}: {all_measurements[i].shape} (erwartet: {first_shape})")
    
    # Stack zu einem Tensor
    try:
        result = torch.stack(all_measurements)
        print(f"[DEBUG] extract_measurements: Finale Form des gestackten Tensors: {result.shape}")
        
        # Stelle sicher, dass die Form 2D ist: [samples, features]
        if len(result.shape) > 2:
            original_shape = result.shape
            # Reshape zu [samples, features]
            result = result.reshape(result.shape[0], -1)
            print(f"[DEBUG] Tensor umgeformt von {original_shape} zu {result.shape}")
        
        return result
    except Exception as e:
        print(f"[FEHLER] Beim Stacking der Messungen: {e}")
        raise


def process_subjects(
    subjects: List[dict],
    batch_size: int,
    shuffle_data: bool,
) -> DataLoader:
    """
    Bereitet Subjekte für das Modell vor und erstellt einen DataLoader.
    Enthält Debug-Output, um Dimensionsprobleme zu identifizieren.
    
    Args:
        subjects: Liste von Subjekt-Dictionaries
        batch_size: Batchgröße für den DataLoader
        shuffle_data: Ob die Daten gemischt werden sollen
        
    Returns:
        DataLoader: DataLoader-Objekt mit korrekt formatierten Daten
    """
    print(f"[DEBUG] process_subjects: Verarbeite {len(subjects)} Subjekte")
    
    # Debug: Prüfe das erste Subjekt
    if len(subjects) > 0:
        first_subj = subjects[0]
        print(f"[DEBUG] process_subjects: Schlüssel im ersten Subjekt: {list(first_subj.keys())}")
        
        # Überprüfe measurements
        if 'measurements' in first_subj:
            measurements = first_subj['measurements']
            print(f"[DEBUG] process_subjects: Measurements Typ: {type(measurements)}")
            
            if isinstance(measurements, list):
                print(f"[DEBUG] process_subjects: Measurements ist Liste der Länge {len(measurements)}")
                # Wenn es eine Liste von Listen ist, prüfe die innere Struktur
                if len(measurements) > 0 and isinstance(measurements[0], list):
                    print(f"[DEBUG] process_subjects: Innere Liste hat Länge {len(measurements[0])}")
                    
                    # Analysiere die Struktur genauer
                    roi_count = len(measurements)
                    feature_count = len(measurements[0]) if len(measurements) > 0 else 0
                    print(f"[DEBUG] process_subjects: Struktur scheint zu sein: [ROIs({roi_count}), Features({feature_count})]")
                    print(f"[DEBUG] process_subjects: Das entspricht {roi_count * feature_count} Gesamtfeatures nach flattening")
                    
                    # Warnung wenn die Struktur ungewöhnlich erscheint
                    if feature_count == 2 and roi_count > 20:
                        print("[WARNUNG] Die Datenstruktur hat genau 2 Features pro ROI. " 
                              "Dies könnte das Problem mit (Nx2 vs 52x100) verursachen!")
            
            elif hasattr(measurements, 'shape'):
                print(f"[DEBUG] process_subjects: Measurements Shape: {measurements.shape}")

    # Erstelle Dataset
    dataset = CustomDataset_2D(subjects=subjects)
    
    # Überprüfe ein Element aus dem Dataset zur Diagnose
    try:
        sample_measurements, sample_labels, sample_name = dataset[0]
        print(f"[DEBUG] process_subjects: Beispiel aus Dataset:")
        print(f"  - Measurements Shape: {sample_measurements.shape}")
        print(f"  - Labels Shape: {sample_labels.shape}")
        print(f"  - Name: {sample_name}")
    except Exception as e:
        print(f"[FEHLER] Beim Zugriff auf Dataset-Element: {e}")
    
    # Erstelle DataLoader
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle_data, 
        num_workers=4,
        pin_memory=True
    )
    
    # Überprüfe einen Batch aus dem DataLoader
    try:
        sample_iter = iter(data_loader)
        batch_measurements, batch_labels, batch_names = next(sample_iter)
        print(f"[DEBUG] process_subjects: Beispiel-Batch aus DataLoader:")
        print(f"  - Batch Measurements Shape: {batch_measurements.shape}")
        print(f"  - Batch Labels Shape: {batch_labels.shape}")
        print(f"  - Batch Size: {len(batch_names)}")
        
        # Spezifische Prüfung auf das Problem "1664x2 and 52x100"
        if batch_measurements.shape[1] == 2:
            print("[KRITISCHER FEHLER] Die Batch-Dimension ist genau 2! " 
                  "Dies verursacht wahrscheinlich den '1664x2 and 52x100' Fehler.")
            print("[LÖSUNG] Die Messdaten müssen umgeformt werden - siehe korrigierte CustomDataset_2D Klasse.")
    except Exception as e:
        print(f"[FEHLER] Beim Zugriff auf Batch aus DataLoader: {e}")
    
    return data_loader

# Korrigierte Pipeline mit Dimensionsanpassung

# Prepare data loaders mit korrigierter Funktion
print("\n[DEBUG] === STARTING DATA LOADER PREPARATION ===")
train_loader_norm = process_subjects(
    subjects=train_subjects_norm,
    batch_size=config.BATCH_SIZE,
    shuffle_data=config.SHUFFLE_DATA,
)
valid_loader_norm = process_subjects(
    subjects=valid_subjects_norm,
    batch_size=config.BATCH_SIZE,
    shuffle_data=False,
)

# Log the used atlas and the number of ROIs
log_atlas_mode(atlas_name=config.ATLAS_NAME, num_rois=len_atlas)

# Log data setup
log_data_loading(
    datasets={
        "Training Data": len(train_subjects_norm),
        "Validation Data": len(valid_subjects_norm),
    }
)

## 2. Prepare and Run Normative Modeling Pipeline --------------------------------
print("\n[DEBUG] === PREPARE NORMATIVE MODELING PIPELINE ===")

# Initialize Model
log_model_setup()

# Extract features as torch tensors mit korrigierter Funktion
print("\n[DEBUG] Extrahiere measurements aus Subjekten...")
train_data = extract_measurements(train_subjects_norm)
valid_data = extract_measurements(valid_subjects_norm)

# Überprüfe resultierende Form
print(f"[DEBUG] Training data shape: {train_data.shape}")
print(f"[DEBUG] Validation data shape: {valid_data.shape}")

# Passe die Form an, falls nötig (falls extract_measurements nicht bereits umformt)
if len(train_data.shape) == 3:
    print(f"[DEBUG] Reshape training data von {train_data.shape} auf 2D")
    train_samples = train_data.shape[0]
    train_data = train_data.reshape(train_samples, -1)
    print(f"[DEBUG] Neue training data shape: {train_data.shape}")

if len(valid_data.shape) == 3:
    print(f"[DEBUG] Reshape validation data von {valid_data.shape} auf 2D")
    valid_samples = valid_data.shape[0]
    valid_data = valid_data.reshape(valid_samples, -1)
    print(f"[DEBUG] Neue validation data shape: {valid_data.shape}")

# Log endgültige Formen
log_and_print(f"Training data shape: {train_data.shape}")
log_and_print(f"Validation data shape: {valid_data.shape}")

# Save processed data tensors for future use
torch.save(train_data, f"{save_dir}/data/train_data_tensor.pt")
torch.save(valid_data, f"{save_dir}/data/valid_data_tensor.pt")