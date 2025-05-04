def compute_regional_deviations(bootstrap_models, data, sample_ids=None, batch_size=32):
    """
    Calculate region-level deviation scores with a set of bootstrap models
    
    Args:
        bootstrap_models: List of trained bootstrap models
        data: Data to test (tensor)
        sample_ids: Optional list of sample IDs corresponding to data rows
        batch_size: Batch size
    
    Returns:
        region_z_scores: Z-scores for each region and sample [samples, regions]
        sample_ids: Original sample IDs in the same order as region_z_scores (if provided)
    """
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm
    import logging
    
    device = bootstrap_models[0].device
    n_bootstraps = len(bootstrap_models)
    n_samples = len(data)
    n_regions = data.shape[1]
    
    # Array for reconstruction errors per region
    all_errors = np.zeros((n_bootstraps, n_samples, n_regions))
    
    # Create DataLoader
    data_loader = DataLoader(
        TensorDataset(data),
        batch_size=batch_size,
        shuffle=False  # Important: keep original order for sample_ids matching
    )
    
    # Calculate reconstruction error for each bootstrap model
    for b, model in enumerate(tqdm(bootstrap_models, desc="Computing Regional Deviations")):
        model.eval()
        batch_start = 0
        
        with torch.no_grad():
            for batch_idx, (x,) in enumerate(data_loader):
                x = x.to(device)
                batch_size_actual = x.shape[0]
                
                # Calculate reconstruction error
                recon_error = model.compute_reconstruction_error(x).cpu().numpy()
                
                # Store errors
                all_errors[b, batch_start:batch_start+batch_size_actual] = recon_error
                batch_start += batch_size_actual
    
    # Mean and standard deviation across bootstrap models
    mean_errors = np.mean(all_errors, axis=0)  # Mean across bootstraps [samples, regions]
    std_errors = np.std(all_errors, axis=0)    # Std across bootstraps [samples, regions]
    
    # Prevent division by zero
    std_errors = np.maximum(std_errors, 1e-8)
    
    # Calculate Z-scores for each region
    region_z_scores = mean_errors / std_errors
    
    # Return with sample IDs if provided
    if sample_ids is not None:
        # Make sure sample_ids length matches the data
        if len(sample_ids) != len(region_z_scores):
            logging.warning(f"Length mismatch: {len(region_z_scores)} samples in results but {len(sample_ids)} in sample_ids")
            # Trim to match
            sample_ids = sample_ids[:len(region_z_scores)]
        return region_z_scores, sample_ids
    else:
        return region_z_scores