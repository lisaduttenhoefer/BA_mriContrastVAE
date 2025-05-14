

def main(args):
    # ---------------------- INITIAL SETUP (output dirs, device, seed) --------------------------------------------
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{args.output_dir}/clinical_deviations_{timestamp}" if args.output_dir else f"./clinical_deviations_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/figures", exist_ok=True)
    
    # Set up logging
    log_file = f"{save_dir}/deviation_analysis.log"
    logger = setup_logging_test(log_file=log_file)
    
    # Log start of analysis
    log_and_print_test("Starting deviation analysis for clinical groups")
    log_and_print_test(f"Model directory: {model_dir}")
    log_and_print_test(f"Output directory: {save_dir}")
    
    # Set device
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    log_and_print_test(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available() and not args.no_cuda:
        torch.cuda.manual_seed_all(args.seed)
    
    # ---------------------- LOAD MODEL CONFIG FROM IG TRAINING (cosistency)  --------------------------------------------
    try:
        config_path = os.path.join(model_dir, "config.csv")
        config_df = pd.read_csv(config_path)
        log_and_print_test(f"Loaded model configuration from {config_path}")
        
        # Extract relevant parameters
        atlas_name = config_df["ATLAS_NAME"].iloc[0]
        latent_dim = int(config_df["LATENT_DIM"].iloc[0])
        hidden_dim_1 = 100  # Default if not in config
        hidden_dim_2 = 100  # Default if not in config
        volume_type = config_df["VOLUME_TYPE"].iloc[0] if "VOLUME_TYPE" in config_df.columns else "Vgm"
        valid_volume_types = ["Vgm", "Vwm", "csf"]  # Default if not specified
        
    except (FileNotFoundError, KeyError) as e:
        log_and_print_test(f"Warning: Could not load config file. Using default parameters. Error: {e}")
        atlas_name = args.atlas_name
        latent_dim = args.latent_dim
        hidden_dim_1 = 100
        hidden_dim_2 = 100
        volume_type = "Vgm"
        valid_volume_types = ["Vgm", "Vwm", "csf"]
    
    # ------------------------------------------ LOADING CLINICAL DATA  --------------------------------------------
    log_and_print_test("Loading clinical data...")
    
    # Set paths for clinical data
    path_to_clinical_data = clinical_data_path
    
    # Load clinical data using the same function as for healthy controls
    if atlas_name != "all":
        subjects_clinical, annotations_clinical = load_mri_data_2D(
            csv_paths=[clinical_csv],
            data_path=path_to_clinical_data,
            atlas_name=atlas_name,
            diagnoses=["HC", "SCHZ", "CTT", "MDD"],  # Include all diagnoses
            hdf5=True,
            train_or_test="test",
            save=False,
            volume_type=volume_type,
            valid_volume_types=valid_volume_types,
        )
    else:
        all_data_paths = get_all_data(directory=path_to_clinical_data, ext="h5")
        subjects_clinical, annotations_clinical = load_mri_data_2D_all_atlases(
            csv_paths=[clinical_csv],
            data_paths=all_data_paths,
            diagnoses=["HC", "SCHZ", "CTT", "MDD"],
            hdf5=True,
            train_or_test="test",
            volume_type=volume_type,
            valid_volume_types=valid_volume_types,
        )
    
    # Extract measurements AND ROI names
    clinical_data, roi_names = extract_measurements(subjects_clinical)
    log_and_print_test(f"Clinical data shape: {clinical_data.shape}")
    log_and_print_test(f"Number of ROIs: {len(roi_names)}")
    
    # Get input dimension
    input_dim = clinical_data.shape[1]
    log_and_print_test(f"Input dimension: {input_dim}")
    
    # Count subjects by diagnosis
    diagnosis_counts = annotations_clinical["Diagnosis"].value_counts()
    log_and_print_test(f"Subject counts by diagnosis:\n{diagnosis_counts}")
    
    # Save ROI names for reference
    roi_df = pd.DataFrame({'ROI_Index': range(len(roi_names)), 'ROI_Name': roi_names})
    roi_df.to_csv(f"{save_dir}/roi_names.csv", index=False)
    log_and_print_test(f"Saved ROI names to {save_dir}/roi_names.csv")
    
    # ---------------------- LOAD BOOTSTRAP MODELS (increases robustness)  --------------------------------------------
    log_and_print_test("Loading normative bootstrap models...")
    bootstrap_models = []
    models_dir = os.path.join(model_dir, "models")
    model_files = [f for f in os.listdir(models_dir) if f.startswith("bootstrap_") and f.endswith(".pt")]
    
    if len(model_files) == 0:
        log_and_print_test("No bootstrap models found. Looking for baseline model...")
        if os.path.exists(os.path.join(models_dir, "baseline_model.pt")):
            model_files = ["baseline_model.pt"]
        else:
            raise FileNotFoundError("No models found in the specified directory.")
    
    # Load up to max_models if specified
    if args.max_models > 0:
        model_files = model_files[:args.max_models]
    
    for model_file in model_files:
        # Initialize model architecture
        model = NormativeVAE(
            input_dim=input_dim,
            hidden_dim_1=hidden_dim_1,
            hidden_dim_2=hidden_dim_2,
            latent_dim=latent_dim,
            learning_rate=1e-4,  # Not used for inference
            kldiv_loss_weight=1.0,  # Not used for inference
            dropout_prob=0.0,  # Set to 0 for inference
            device="cpu"  # Will move to appropriate device later
        )
        
        # Load model weights
        model_path = os.path.join(models_dir, model_file)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        bootstrap_models.append(model)
        log_and_print_test(f"Loaded model: {model_file}")
    
    log_and_print_test(f"Successfully loaded {len(bootstrap_models)} models")
    
    # ------------------------------- CALCULATION DEV_SCORES WITH ROI TRACKING --------------------------------------------
    log_and_print_test("Calculating deviation scores with ROI tracking...")
    
    # Use our modified function that maintains ROI names
    results_df = calculate_roi_deviation_scores(
        normative_models=bootstrap_models,
        data_tensor=clinical_data,
        annotations_df=