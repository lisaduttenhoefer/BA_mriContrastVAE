# Fix 1: Improve tensor handling in training loop
def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
    """Train for one epoch"""
    self.train()
    total_loss, contr_loss, recon_loss, kldiv_loss = 0.0, 0.0, 0.0, 0.0
    
    for batch_idx, (measurements, labels, names) in enumerate(train_loader):
        # More efficient tensor handling - avoid double stacking
        if isinstance(measurements, list):
            batch_measurements = torch.stack(measurements).to(self.device, non_blocking=True)
        else:
            batch_measurements = measurements.to(self.device, non_blocking=True)
            
        if isinstance(labels, list):
            batch_labels = torch.stack(labels).to(self.device, non_blocking=True)
        else:
            batch_labels = labels.to(self.device, non_blocking=True)
        
        # Clear gradients before forward pass
        self.optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=True):
            recon_data, mu, logvar = self(batch_measurements)
            b_total_loss, b_contr_loss, b_recon_loss, b_kldiv_loss = self.combined_loss_function(
                recon_measurements=recon_data,
                meas=batch_measurements,
                mu=mu,
                log_var=logvar,
                labels=batch_labels,
            )
        
        # Backward pass
        self.scaler.scale(b_total_loss).backward()
        
        # Handle NaN gradients
        for param in self.parameters():
            if param.grad is not None and torch.any(torch.isnan(param.grad)):
                param.grad.nan_to_num_(0.0)
        
        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
        
        # Update weights
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Accumulate losses
        total_loss += b_total_loss.item()
        contr_loss += b_contr_loss.item()
        recon_loss += b_recon_loss.item()
        kldiv_loss += b_kldiv_loss.item()
        
        # Clear cache periodically to prevent memory buildup
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        # Delete tensors to free memory
        del batch_measurements, batch_labels, recon_data, mu, logvar
        del b_total_loss, b_contr_loss, b_recon_loss, b_kldiv_loss
    
    # Calculate epoch metrics
    epoch_metrics = {
        "train_loss": total_loss / len(train_loader.dataset),
        "t_contr_loss": contr_loss / len(train_loader.dataset),
        "t_recon_loss": recon_loss / len(train_loader.dataset),
        "t_kldiv_loss": kldiv_loss / len(train_loader.dataset),
    }
    
    return epoch_metrics

# Fix 2: Improve validation method
@torch.no_grad()
def validate(self, valid_loader, epoch) -> Dict[str, float]:
    """Validate the model"""
    self.eval()
    
    total_loss, contr_loss, recon_loss, kldiv_loss = 0.0, 0.0, 0.0, 0.0
    
    for batch_idx, (measurements, labels, names) in enumerate(valid_loader):
        # More efficient tensor handling
        if isinstance(measurements, list):
            batch_measurements = torch.stack(measurements).to(self.device, non_blocking=True)
        else:
            batch_measurements = measurements.to(self.device, non_blocking=True)
            
        if isinstance(labels, list):
            batch_labels = torch.stack(labels).to(self.device, non_blocking=True)
        else:
            batch_labels = labels.to(self.device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=True):
            recon_data, mu, logvar = self(batch_measurements)
            b_total_loss, b_contr_loss, b_recon_loss, b_kldiv_loss = self.combined_loss_function(
                recon_measurements=recon_data,
                meas=batch_measurements,
                mu=mu,
                log_var=logvar,
                labels=batch_labels,
            )
        
        # Accumulate losses
        total_loss += b_total_loss.item()
        contr_loss += b_contr_loss.item()
        recon_loss += b_recon_loss.item()
        kldiv_loss += b_kldiv_loss.item()
        
        # Clear memory
        del batch_measurements, batch_labels, recon_data, mu, logvar
        del b_total_loss, b_contr_loss, b_recon_loss, b_kldiv_loss
        
        # Clear cache periodically
        if batch_idx % 5 == 0:
            torch.cuda.empty_cache()
    
    # Calculate metrics
    epoch_metrics = {
        "valid_loss": total_loss / len(valid_loader.dataset),
        "v_contr_loss": contr_loss / len(valid_loader.dataset),
        "v_recon_loss": recon_loss / len(valid_loader.dataset),
        "v_kldiv_loss": kldiv_loss / len(valid_loader.dataset),
        "accuracy": 0.0,
    }
    
    return epoch_metrics

# Fix 3: DataLoader configuration suggestions
def create_dataloader_with_proper_batch_size(dataset, device_memory_gb=None):
    """
    Create DataLoader with appropriate batch size based on available GPU memory
    """
    # Estimate appropriate batch size based on GPU memory
    if device_memory_gb is None:
        # Get available GPU memory in GB
        if torch.cuda.is_available():
            device_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            device_memory_gb = 4  # Conservative default
    
    # Conservative batch size estimation
    if device_memory_gb >= 24:      # RTX 4090, A6000, etc.
        batch_size = 64
    elif device_memory_gb >= 12:    # RTX 4070 Ti, RTX 3080, etc.
        batch_size = 32
    elif device_memory_gb >= 8:     # RTX 4060 Ti, RTX 3070, etc.
        batch_size = 16
    elif device_memory_gb >= 6:     # RTX 4060, RTX 3060, etc.
        batch_size = 8
    else:                           # Lower memory GPUs
        batch_size = 4
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # Reduce if causing issues
        pin_memory=True,
        drop_last=True,  # Avoid smaller last batch
        persistent_workers=True
    )
    
    return dataloader

# Fix 4: Memory monitoring function
def monitor_gpu_memory():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")
        
        # Clear cache if memory usage is high
        if reserved / total > 0.9:
            torch.cuda.empty_cache()
            print("Cleared CUDA cache due to high memory usage")

# Fix 5: Modified extract_latent_space with better memory management
@torch.no_grad()
def extract_latent_space(self, data_loader, data_type):
    """Extract latent space representations with better memory management"""
    log_extracting_latent_space(data_type)
    
    self.eval()
    latent_spaces = []
    sample_names = []
    
    # Process in smaller chunks to avoid memory issues
    for batch_idx, (measurements, labels, names) in enumerate(data_loader):
        if isinstance(measurements, list):
            batch_measurements = torch.stack(measurements).to(self.device, non_blocking=True)
        else:
            batch_measurements = measurements.to(self.device, non_blocking=True)
        
        # Get latent representations
        mu = self.to_latent(batch_measurements)
        
        # Move to CPU immediately to free GPU memory
        latent_spaces.append(mu.cpu().numpy())
        sample_names.extend(names)
        
        # Clean up GPU memory
        del batch_measurements, mu
        
        # Clear cache every few batches
        if batch_idx % 5 == 0:
            torch.cuda.empty_cache()
    
    # Create anndata object
    adata = ad.AnnData(np.concatenate(latent_spaces))
    adata.obs_names = sample_names        
    return adata

# Fix 6: Emergency memory cleanup function
def emergency_memory_cleanup():
    """Call this function if you get CUDA out of memory errors"""
    import gc
    
    # Clear Python garbage collection
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print("Emergency memory cleanup completed")

# Fix 7: Configuration suggestions for your Config_2D class
"""
Add these to your config:

BATCH_SIZE = 16  # Start small and increase if memory allows
NUM_WORKERS = 2  # Reduce if causing memory issues
PIN_MEMORY = True
DROP_LAST = True

# Memory management settings
CLEAR_CACHE_FREQUENCY = 10  # Clear cache every N batches
EMERGENCY_CLEANUP_ON_ERROR = True
"""