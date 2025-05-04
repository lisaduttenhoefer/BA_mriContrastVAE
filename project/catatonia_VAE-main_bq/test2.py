def compute_deviation_scores(bootstrap_models, data, batch_size=32):
    """
    Calculate deviation scores with a set of bootstrap models
    
    Args:
        bootstrap_models: List of trained bootstrap models
        data: Data to test (patients or HC)
        batch_size: Batch size
    
    Returns:
        z_scores: Z-scores for each dimension and data point
        global_z: Global Z-scores for each data point
    """
    device = bootstrap_models[0].device
    n_bootstraps = len(bootstrap_models)
    n_samples = len(data)
    n_features = data.shape[1]
    
    # Create DataLoader
    data_loader = DataLoader(
        TensorDataset(data),
        batch_size=batch_size,
        shuffle=False
    )
    
    # Array for reconstruction errors
    all_errors = np.zeros((n_bootstraps, n_samples, n_features))
    
    # Calculate reconstruction error for each bootstrap model
    for b, model in enumerate(tqdm(bootstrap_models, desc="Computing Deviation Scores")):
        model.eval()
        batch_start = 0
        
        with torch.no_grad():
            for batch_idx, (x,) in enumerate(data_loader):
                x = x.to(device)
                batch_size = x.shape[0]
                
                # Calculate reconstruction error
                recon_error = model.compute_reconstruction_error(x).cpu().numpy()
                
                # Store errors
                all_errors[b, batch_start:batch_start+batch_size] = recon_error
                batch_start += batch_size
    
    # Mean and standard deviation across bootstrap models
    mean_errors = np.mean(all_errors, axis=0)  # Mean across bootstraps
    std_errors = np.std(all_errors, axis=0)    # Std across bootstraps
    
    # Prevent division by zero
    std_errors = np.maximum(std_errors, 1e-8)
    
    # Calculate Z-scores (standardized errors)
    z_scores = mean_errors / std_errors
    
    # Global Z-scores (mean across dimensions)
    global_z = np.mean(z_scores, axis=1)
    
    return z_scores, global_z