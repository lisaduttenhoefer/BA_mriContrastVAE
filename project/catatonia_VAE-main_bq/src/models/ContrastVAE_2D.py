import sys
import matplotlib.pyplot as plt
import logging
from typing import Dict, Tuple
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import os
from sklearn.utils import resample
try:
    import anndata as ad
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torch.nn.functional as F
import torchio as tio
from pytorch_msssim import ssim
from tqdm import tqdm

from models.base_model import (
    loss_proportions,
    supervised_contrastive_loss,
    update_performance_metrics,
)
from utils.config_utils_model import Config_2D

from module.data_processing_hc import (
    combine_latent_spaces,
    process_latent_space_2D,
    save_model,
    save_model_metrics,
)
from utils.logging_utils import (
    end_logging,
    log_and_print,
    log_checkpoint,
    log_early_stopping,
    log_extracting_latent_space,
    log_model_metrics,
    log_training_start,
)
from utils.plotting_utils import (
    latent_space_batch_plot,
    latent_space_details_plot,
    latent_space_plot,
    metrics_plot,
    recon_images_plot,
    plot_latent_space,
    plot_learning_curves,
    plot_bootstrap_metrics,
)
# ContrastVAE_2D Model Class
class NormativeVAE_2D(nn.Module):
    def __init__(
        self,
        #num_classes,
        recon_loss_weight,
        kldiv_loss_weight,
        contr_loss_weight,
        contr_temperature=0.1,
        input_dim:int = None,  # Need to specify input dimension
        hidden_dim_1=100,
        hidden_dim_2=100,
        latent_dim=20,
        learning_rate=1e-4,
        weight_decay=1e-5,
        device=None,
        dropout_prob=0.1,
        schedule_on_validation=True,
        scheduler_patience=10,
        scheduler_factor=0.5,
        
    ):
        super(NormativeVAE_2D, self).__init__()
        
        # Store important parameters for later
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.latent_dim = latent_dim
        #self.num_classes = num_classes
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.LeakyReLU(1e-2),
            nn.Dropout(dropout_prob),
            
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.LeakyReLU(1e-2),
            nn.Dropout(dropout_prob),
            
            nn.Linear(hidden_dim_2, latent_dim)
        )
        
        self.encoder_feature_dim = int(latent_dim)
        
        # Mu and logvar projections
        self.fc_mu = nn.Linear(self.encoder_feature_dim, latent_dim)
        self.fc_var = nn.Linear(self.encoder_feature_dim, latent_dim)
        
         # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_2),
            nn.LeakyReLU(1e-2),
            nn.Dropout(dropout_prob),
            
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.LeakyReLU(1e-2),
            nn.Dropout(dropout_prob),
            
            nn.Linear(hidden_dim_1, input_dim)
        )
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Scheduler for learning rate
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            verbose=True,
        )
        
        # Gradient Scaler for Mixed Precision
        self.scaler = GradScaler()
        
        # Set loss weights
        self.recon_loss_weight = recon_loss_weight
        self.kldiv_loss_weight = kldiv_loss_weight
        self.contr_loss_weight = contr_loss_weight
        
        # Additional loss metrics
        self.contr_temperature = contr_temperature
        
        # Record some values for logging
        self.schedule_on_validation = schedule_on_validation
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.latent_dim = latent_dim
        self.dropout_prob = dropout_prob
        
        # Send to Device
        if device is not None:
            self.device = device
            self.to(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(self.device)
        
        # Initialize weights
        self.apply(self._init_weights)
            
    def _init_weights(self, module):
        """Initialize weights for linear layers"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
    def reparameterize(self, mu, logvar):
        # self: The latent space produced by the encode
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """Forward pass through the model"""
        # Encoder
        x = self.encoder(x)
        
        # Get latent parameters
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        
        # Sample from latent space
        z = self.reparameterize(mu, logvar)
        
        # Decoder
        x = self.decoder(z)
        
        return x, mu, logvar
    
    def to_latent(self, x):
        """Extract latent space representation"""
        # Encoder
        self.eval()
        with torch.no_grad():
            x = self.encoder(x)
            # Latent space
            mu = self.fc_mu(x)
        
        return mu
    
    ###
    def reconstruct(self, x):
        """Rekonstruiere einen Input"""
        self.eval()
        with torch.no_grad():
            recon, _, _ = self(x)
        return recon
    
    def loss_function(self, recon_x, x, mu, logvar):
        """VAE loss function with reconstruction error and KL divergence"""
        # Rekonstruktionsloss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        
        # KL-Divergenz
        kldiv_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kldiv_loss = kldiv_loss * self.kldiv_loss_weight
        
        # total loss
        total_loss = recon_loss + kldiv_loss
        
        return total_loss, recon_loss, kldiv_loss

    
    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        # Set to train mode
        self.train()
        total_loss, contr_loss, recon_loss, kldiv_loss = 0.0, 0.0, 0.0, 0.0
        
        # Iterate through batches
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

            # Autocast for mixed precision
            with torch.cuda.amp.autocast(enabled=True):
                # Forward pass
                recon_data, mu, logvar = self(batch_measurements)
                
                # Calculate loss
                b_total_loss, b_contr_loss, b_recon_loss, b_kldiv_loss = self.combined_loss_function(
                    recon_measurements=recon_data,
                    meas=batch_measurements,
                    mu=mu,
                    log_var=logvar,
                    labels=batch_labels,
                )
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(b_total_loss).backward()
            
            # Remove NaNs from gradients
            for param in self.parameters():
                if param.grad is not None and torch.any(torch.isnan(param.grad)):
                    param.grad.nan_to_num_(0.0)
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
            
            # Update weights
            self.scaler.step(self.optimizer)
            self.scaler.update()
            #self.optimizer.zero_grad()
            # Update loss stats
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
        
        # Calculate loss proportions
        epoch_props = loss_proportions("train_loss", epoch_metrics)
        
        # Log metrics
        log_model_metrics(epoch, epoch_props, type="Training Metrics:")
        
        # Update learning rate if needed
        if not self.schedule_on_validation:
            self.scheduler.step(total_loss / len(train_loader))
            current_lr = self.optimizer.param_groups[0]["lr"]
            logging.info("Current Learning Rate: %f", current_lr)
            epoch_metrics["learning_rate"] = current_lr
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self, valid_loader, epoch) -> Dict[str, float]:
        """Validate the model"""
        # Set to evaluation mode
        self.eval()
        
        # Initialize loss values
        total_loss, contr_loss, recon_loss, kldiv_loss = 0.0, 0.0, 0.0, 0.0
        
        # Go through validation data
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
                
            # Update loss stats
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
            "accuracy": 0.0,  # Placeholder - in a real model would calculate accuracy
        }
        
        # Calculate loss proportions
        epoch_props = loss_proportions("valid_loss", epoch_metrics)
        
        # Log metrics
        log_model_metrics(epoch, epoch_props, type="Validation Metrics:")
        
        # Update learning rate if needed
        if self.schedule_on_validation:
            self.scheduler.step(total_loss / len(valid_loader))
            current_lr = self.optimizer.param_groups[0]["lr"]
            logging.info("Current Learning Rate: %f", current_lr)
            epoch_metrics["learning_rate"] = current_lr
        
        return epoch_metrics
    
    ###
    @torch.no_grad()
    def compute_reconstruction_error(self, x):
        """Calculate reconstruction error for a sample"""
        self.eval()
        x = x.to(self.device)
        recon_x, _, _ = self(x)
        # Berechne MSE für jedes Feature
        recon_error = (x - recon_x).pow(2)
        return recon_error
   

def train_normative_model_plots(train_data, valid_data, model, epochs, batch_size, save_best=True, return_history=True):
    """
    Train a single normative model with detailed tracking of metrics.
    
    Args:
        train_data: Training data tensor
        valid_data: Validation data tensor
        model: Model to train
        epochs: Number of training epochs
        batch_size: Batch size for training
        save_best: Whether to save best model based on validation loss
        return_history: Whether to return training history
        
    Returns:
        Trained model and history dictionary if return_history=True
    """
    device = model.device
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    
    # Initialize history dictionary
    history = {
        'train_loss': [],
        'val_loss': [],
        'recon_loss': [],
        'kl_loss': [],
        'best_epoch': 0,
        'best_val_loss': float('inf')
    }
    
    # Create data loaders
    log_and_print(f"Training data shape IM MODEL: {train_data.shape}")
    train_dataset = torch.utils.data.TensorDataset(train_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    valid_dataset = torch.utils.data.TensorDataset(valid_data)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, log_var = model(data)
            loss, recon_loss, kl_loss = model.loss_function(recon_batch, data, mu, log_var)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
        
        # Validation step
        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (data,) in enumerate(valid_loader):
                data = data.to(device)
                recon_batch, mu, log_var = model(data)
                loss, recon_loss, kl_loss = model.loss_function(recon_batch, data, mu, log_var)
                
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()
        
        # Normalize by batch count
        train_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_kl_loss /= len(train_loader)
        
        val_loss /= len(valid_loader)
        val_recon_loss /= len(valid_loader)
        val_kl_loss /= len(valid_loader)
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['recon_loss'].append(train_recon_loss)
        history['kl_loss'].append(train_kl_loss)
        
        # Save best model based on validation loss
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch
            if save_best:
                best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs-1:
            log_and_print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
                         f"Val Loss: {val_loss:.4f}, Recon: {train_recon_loss:.4f}, KL: {train_kl_loss:.4f}")
    
    # Load best model if available
    if save_best and best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    if return_history:
        return model, history
    return model


def bootstrap_train_normative_models_plots(
    train_data, valid_data, model, n_bootstraps, epochs, batch_size, save_dir, save_models=True):
    """
    Train multiple bootstrap models and save detailed metrics and visualizations.
    
    Args:
        train_data: Training data tensor
        valid_data: Validation data tensor
        model: Base model to bootstrap
        n_bootstraps: Number of bootstrap samples
        epochs: Number of training epochs
        batch_size: Batch size for training
        save_dir: Directory to save results
        save_models: Whether to save model weights
        
    Returns:
        Tuple of (List of trained models, List of metrics dictionaries)
    """
    n_samples = train_data.shape[0]
    device = model.device
    bootstrap_models = []
    bootstrap_metrics = []
    all_losses = []
    
    # Create figures directory structure
    figures_dir = os.path.join(save_dir, "figures")
    latent_dir = os.path.join(figures_dir, "latent_space")
    recon_dir = os.path.join(figures_dir, "reconstructions")
    loss_dir = os.path.join(figures_dir, "loss_curves")
    
    os.makedirs(latent_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)
    
    log_and_print(f"Starting bootstrap training with {n_bootstraps} iterations")
    
    # Combine training and validation data for visualization
    combined_data = torch.cat([train_data, valid_data], dim=0)
    combined_labels = torch.cat([torch.zeros(train_data.shape[0]), torch.ones(valid_data.shape[0])])
    
    for i in range(n_bootstraps):
        log_and_print(f"Training bootstrap model {i+1}/{n_bootstraps}")
        
        # Create bootstrap sample (with replacement)
        bootstrap_indices = torch.randint(0, n_samples, (n_samples,))
        bootstrap_train_data = train_data[bootstrap_indices]
        
        # Create a fresh model instance
        bootstrap_model = NormativeVAE(
            input_dim=model.input_dim,
            hidden_dim_1=model.hidden_dim_1,
            hidden_dim_2=model.hidden_dim_2,
            latent_dim=model.latent_dim,
            learning_rate=model.learning_rate,
            kldiv_loss_weight=model.kldiv_loss_weight,
            dropout_prob=model.dropout_prob,
            device=device
        )
        
        # Train the model and collect metrics
        trained_model, history = train_normative_model_plots(
            train_data=bootstrap_train_data,
            valid_data=valid_data,
            model=bootstrap_model,
            epochs=epochs,
            batch_size=batch_size,
            save_best=True,
            return_history=True
        )
        
        # Extract metrics
        metrics = {
            'bootstrap_id': i,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_recon_loss': history['recon_loss'][-1],
            'final_kl_loss': history['kl_loss'][-1],
            'best_epoch': history['best_epoch'],
            'best_val_loss': history['best_val_loss']
        }
        
        bootstrap_models.append(trained_model)
        bootstrap_metrics.append(metrics)
        all_losses.append(history)
        
        # Save model if requested
        if save_models:
            model_save_path = os.path.join(save_dir, "models", f"bootstrap_model_{i}.pt")
            torch.save(trained_model.state_dict(), model_save_path)
            log_and_print(f"Saved model {i+1} to {model_save_path}")
        
    
    # Save all metrics to CSV
    metrics_df = pd.DataFrame(bootstrap_metrics)
    metrics_df.to_csv(os.path.join(save_dir, "models", "bootstrap_metrics.csv"), index=False)
    
    # Create combined loss plots from all bootstraps
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for i, history in enumerate(all_losses):
        plt.plot(history['val_loss'], alpha=0.3, color='blue')
    plt.plot([np.mean([h['val_loss'][e] for h in all_losses]) for e in range(epochs)], 
             linewidth=2, color='red', label='Mean Validation Loss')
    plt.title('Validation Loss Across Bootstrap Models')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    for i, history in enumerate(all_losses):
        plt.plot(history['train_loss'], alpha=0.3, color='green')
    plt.plot([np.mean([h['train_loss'][e] for h in all_losses]) for e in range(epochs)], 
             linewidth=2, color='red', label='Mean Training Loss')
    plt.title('Training Loss Across Bootstrap Models')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    for i, history in enumerate(all_losses):
        plt.plot(history['recon_loss'], alpha=0.3, color='purple')
    plt.plot([np.mean([h['recon_loss'][e] for h in all_losses]) for e in range(epochs)], 
             linewidth=2, color='red', label='Mean Reconstruction Loss')
    plt.title('Reconstruction Loss Across Bootstrap Models')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    for i, history in enumerate(all_losses):
        plt.plot(history['kl_loss'], alpha=0.3, color='orange')
    plt.plot([np.mean([h['kl_loss'][e] for h in all_losses]) for e in range(epochs)], 
             linewidth=2, color='red', label='Mean KL Divergence Loss')
    plt.title('KL Divergence Loss Across Bootstrap Models')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "bootstrap_losses.png"))
    plt.close()
    
    # Plot distribution of metrics
    plot_bootstrap_metrics(bootstrap_metrics, os.path.join(figures_dir, "bootstrap_metrics_distribution.png"))
    
    return bootstrap_models, bootstrap_metrics

########
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

#######
 ###
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
    
    def weights_init(self, param):
        """Initialize weights"""
        if isinstance(param, (nn.Conv3d, nn.ConvTranspose3d)):       # if the parameter is a convolutional layer, use Kaiming Uniform initialization
            nn.init.kaiming_uniform_(param.weight, nonlinearity="relu")
            if param.bias is not None:
                nn.init.constant_(param.bias, 0)
        elif isinstance(param, nn.Linear):                 # if the parameter is a linear layer, use Xavier Uniform initialization
            nn.init.xavier_uniform_(param.weight)
            nn.init.constant_(param.bias, 0)
            
    def combined_loss_function(self, recon_measurements, meas, mu, log_var, labels):
        """Calculate combined loss"""
        # Contrastive loss
        contr_loss = supervised_contrastive_loss(mu, labels, self.contr_temperature)
        contr_loss = contr_loss * self.contr_loss_weight
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_measurements, meas, reduction="mean")
        recon_loss = recon_loss * self.recon_loss_weight
        
        # KL divergence loss
        kldiv_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        kldiv_loss = kldiv_loss * self.kldiv_loss_weight
        
        # Total loss
        total_loss = contr_loss + recon_loss + kldiv_loss
        
        return total_loss, contr_loss, recon_loss, kldiv_loss
