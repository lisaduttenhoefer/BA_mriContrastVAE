import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import os
from sklearn.utils import resample
try:
    import anndata as ad
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False

from utils.logging_utils import (
    log_and_print,
    log_data_loading,
    log_model_ready,
    log_model_setup, 
    setup_logging,
    log_atlas_mode
)
from utils.plotting_utils import (
    plot_latent_space,
    plot_learning_curves,
    plot_bootstrap_metrics,
)


# NormativeVAE class
class NormativeVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim_1=100,
        hidden_dim_2=100,
        latent_dim=20,
        learning_rate=1e-4,
        weight_decay=1e-5,
        kldiv_loss_weight=0.1,
        device=None,
        dropout_prob=0.1,
        schedule_on_validation=True,
        scheduler_patience=10,
        scheduler_factor=0.5,
    ):
        super(NormativeVAE, self).__init__()
        
        # Store important parameters for later
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.latent_dim = latent_dim
        
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
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            verbose=True,
        )
        
        # Gradient Scaler for Mixed Precision
        self.scaler = GradScaler()
        
        # Loss function weights
        self.kldiv_loss_weight = kldiv_loss_weight
        
        # Save parameters
        self.schedule_on_validation = schedule_on_validation
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_prob = dropout_prob
        
        # Set device
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
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """Forward pass through the model"""
        # Encoder
        x = self.encoder(x)
        
        # Latent parameters
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
    
    def reconstruct(self, x):
        """Reconstruct an input"""
        self.eval()
        with torch.no_grad():
            recon, _, _ = self(x)
        return recon
    
    def loss_function(self, recon_x, x, mu, logvar):
        """VAE loss function with reconstruction error and KL divergence"""
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        
        # KL divergence
        kldiv_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kldiv_loss = kldiv_loss * self.kldiv_loss_weight
        
        # Total loss
        total_loss = recon_loss + kldiv_loss
        
        return total_loss, recon_loss, kldiv_loss
    
    def train_one_epoch(self, train_loader):
        """One epoch of training"""
        self.train()
        total_loss, recon_loss, kldiv_loss = 0.0, 0.0, 0.0
        
        for batch_idx, (x,) in enumerate(train_loader):
            # Move to device
            x = x.to(self.device)
            
            # Autocast for mixed precision
            with autocast():
                # Forward pass
                recon_x, mu, logvar = self(x)
                
                # Calculate loss
                b_total_loss, b_recon_loss, b_kldiv_loss = self.loss_function(
                    recon_x=recon_x,
                    x=x,
                    mu=mu,
                    logvar=logvar
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
            
            # Update loss statistics
            total_loss += b_total_loss.item() * x.size(0)
            recon_loss += b_recon_loss.item() * x.size(0)
            kldiv_loss += b_kldiv_loss.item() * x.size(0)
        
        # Calculate epoch metrics
        epoch_metrics = {
            "train_loss": total_loss / len(train_loader.dataset),
            "train_recon_loss": recon_loss / len(train_loader.dataset),
            "train_kldiv_loss": kldiv_loss / len(train_loader.dataset),
        }
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self, valid_loader):
        """Validate the model"""
        self.eval()
        total_loss, recon_loss, kldiv_loss = 0.0, 0.0, 0.0
        
        for batch_idx, (x,) in enumerate(valid_loader):
            # Move to device
            x = x.to(self.device)
            
            # Forward pass
            recon_x, mu, logvar = self(x)
            
            # Calculate loss
            b_total_loss, b_recon_loss, b_kldiv_loss = self.loss_function(
                recon_x=recon_x,
                x=x,
                mu=mu,
                logvar=logvar
            )
            
            # Update loss statistics
            total_loss += b_total_loss.item() * x.size(0)
            recon_loss += b_recon_loss.item() * x.size(0)
            kldiv_loss += b_kldiv_loss.item() * x.size(0)
        
        # Calculate epoch metrics
        epoch_metrics = {
            "valid_loss": total_loss / len(valid_loader.dataset),
            "valid_recon_loss": recon_loss / len(valid_loader.dataset),
            "valid_kldiv_loss": kldiv_loss / len(valid_loader.dataset),
        }
        
        return epoch_metrics
    
    @torch.no_grad()
    def compute_reconstruction_error(self, x):
        """Calculate reconstruction error for a sample"""
        self.eval()
        x = x.to(self.device)
        recon_x, _, _ = self(x)
        # Calculate MSE for each feature
        recon_error = (x - recon_x).pow(2)
        return recon_error


# Function to train the normative model (only with HC data)
def train_normative_model(
        model, 
        healthy_train_loader, 
        healthy_val_loader, 
        epochs=100, 
        early_stopping_patience=20
    ):
    """
    Train the normative model with healthy control data (HC)
    
    Args:
        model: The NormativeVAE model
        healthy_train_loader: DataLoader with healthy control data for training
        healthy_val_loader: DataLoader with healthy control data for validation
        epochs: Number of training epochs
        early_stopping_patience: Number of epochs for early stopping
    
    Returns:
        model: The trained model
        history: Training history
    """
    history = {
        "train_loss": [], "train_recon_loss": [], "train_kldiv_loss": [],
        "valid_loss": [], "valid_recon_loss": [], "valid_kldiv_loss": [],
        "learning_rate": []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        train_metrics = model.train_one_epoch(healthy_train_loader)
        
        # Validation
        val_metrics = model.validate(healthy_val_loader)
        
        # Update learning rate
        if model.schedule_on_validation:
            model.scheduler.step(val_metrics["valid_loss"])
        else:
            model.scheduler.step(train_metrics["train_loss"])
        
        # Save current learning rate
        current_lr = model.optimizer.param_groups[0]["lr"]
        
        # Save metrics
        for k, v in train_metrics.items():
            history[k].append(v)
        for k, v in val_metrics.items():
            history[k].append(v)
        history["learning_rate"].append(current_lr)
        
        # Show progress
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_metrics['train_loss']:.6f} | "
              f"Valid Loss: {val_metrics['valid_loss']:.6f} | "
              f"LR: {current_lr:.6f}")
        
        # Early stopping
        if val_metrics["valid_loss"] < best_val_loss:
            best_val_loss = val_metrics["valid_loss"]
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping after {epoch+1} epochs!")
                # Restore best model
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
    
    # Convert to DataFrame for easier plotting
    history = {k: np.array(v) for k, v in history.items()}
    history_df = pd.DataFrame(history)
    
    return model, history_df


def bootstrap_train_normative_models(
    train_data,
    valid_data,
    model,
    n_bootstraps=50,
    epochs=20,
    batch_size=32,
    device=torch.device("cuda"),
    save_dir=None,
    save_models=False
):
    """
    Train multiple normative models with bootstrap resampling
    
    Args:
        train_data: Training data tensor with healthy control measurements
        valid_data: Validation data tensor with healthy control measurements
        model: Base model instance to use as template (NormativeVAE)
        n_bootstraps: Number of bootstrap samples
        epochs: Number of training epochs per bootstrap
        batch_size: Batch size for training
        device: Device for training (cuda/cpu)
        save_dir: Directory to save models and results
        save_models: Whether to save all bootstrap models
        
    Returns:
        bootstrap_models: List of trained models
    """
    import matplotlib.pyplot as plt
    
    # Use provided device
    log_and_print(f"Training {n_bootstraps} bootstrap models on {device}...")
    
    # List for trained models and their performance metrics
    bootstrap_models = []
    all_metrics = []
    
    for b in range(n_bootstraps):
        log_and_print(f"\nBootstrap {b+1}/{n_bootstraps}")
        
        # Create bootstrap sample with replacement from training data
        n_samples = len(train_data)
        bootstrap_indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
        bootstrap_data = train_data[bootstrap_indices]
        
        # Create DataLoaders
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(bootstrap_data),
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(valid_data),
            batch_size=batch_size,
            shuffle=False
        )
        
        # Initialize a new model instance with the same configuration
        try:
            bootstrap_model = NormativeVAE(
                input_dim=model.input_dim,
                hidden_dim_1=model.hidden_dim_1 if hasattr(model, 'hidden_dim_1') else 100,
                hidden_dim_2=model.hidden_dim_2 if hasattr(model, 'hidden_dim_2') else 100,
                latent_dim=model.latent_dim,
                learning_rate=model.learning_rate if hasattr(model, 'learning_rate') else 1e-4,
                kldiv_loss_weight=model.kldiv_loss_weight,
                dropout_prob=model.dropout_prob if hasattr(model, 'dropout_prob') else 0.1,
                device=device
            )
            
            # Train model
            log_and_print(f"Training bootstrap model {b+1}...")
            trained_model, history = train_normative_model(
                model=bootstrap_model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                device=device
            )
            
            # Save model checkpoint if requested
            if save_dir and save_models:
                model_path = os.path.join(save_dir, "models", f"bootstrap_model_{b+1}.pt")
                torch.save(trained_model.state_dict(), model_path)
                log_and_print(f"Saved bootstrap model {b+1} to {model_path}")
            
            # Save training history
            if history and isinstance(history, dict):
                all_metrics.append({
                    "bootstrap_id": b+1,
                    "final_train_loss": history["train_loss"][-1] if len(history["train_loss"]) > 0 else float('nan'),
                    "final_val_loss": history["val_loss"][-1] if len(history["val_loss"]) > 0 else float('nan'),
                    "best_val_loss": min(history["val_loss"]) if len(history["val_loss"]) > 0 else float('nan')
                })
            else:
                all_metrics.append({
                    "bootstrap_id": b+1,
                    "final_train_loss": float('nan'),
                    "final_val_loss": float('nan'),
                    "best_val_loss": float('nan')
                })
            
            # Add model to the list
            bootstrap_models.append(trained_model)
            log_and_print(f"Successfully trained bootstrap model {b+1}")
            
        except Exception as e:
            log_and_print(f"Error during training bootstrap model {b+1}: {str(e)}")
            # Continue with the next bootstrap model
            continue
    
    # Save metrics summary if we have any models
    if save_dir and all_metrics:
        try:
            # Create a DataFrame from the metrics and save it
            metrics_df = pd.DataFrame(all_metrics)
            metrics_path = os.path.join(save_dir, "bootstrap_metrics.csv")
            metrics_df.to_csv(metrics_path, index=False)
            log_and_print(f"Saved bootstrap metrics to {metrics_path}")
            
            # Create summary figure of validation losses if we have any valid data
            if any(not np.isnan(m["final_val_loss"]) for m in all_metrics):
                plt.figure(figsize=(10, 6))
                # Filter out NaN values
                val_losses = [m["final_val_loss"] for m in all_metrics if not np.isnan(m["final_val_loss"])]
                if val_losses:
                    plt.boxplot(val_losses)
                    plt.title("Distribution of Validation Losses Across Bootstrap Models")
                    plt.ylabel("Validation Loss")
                    fig_path = os.path.join(save_dir, "figures", "bootstrap_losses.png")
                    plt.savefig(fig_path)
                    log_and_print(f"Saved validation loss distribution plot to {fig_path}")
        except Exception as e:
            log_and_print(f"Error saving metrics or figures: {str(e)}")
    
    log_and_print(f"Successfully trained {len(bootstrap_models)} bootstrap models")
    return bootstrap_models

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

def train_normative_model(model, train_loader, val_loader, epochs=20, device=None, early_stopping_patience=10):
    """
    Train a normative VAE model
    
    Args:
        model: NormativeVAE model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        device: Device to use (cuda/cpu)
        early_stopping_patience: Number of epochs to wait for improvement before stopping
        
    Returns:
        model: Trained model
        history: Dictionary containing training history
    """
    # Ensure we have default return values
    history = {
        "train_loss": [],
        "val_loss": [],
        "recon_loss": [],
        "kl_loss": []
    }
    """
    Train a normative VAE model
    
    Args:
        model: NormativeVAE model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        device: Device to use (cuda/cpu)
        early_stopping_patience: Number of epochs to wait for improvement before stopping
        
    Returns:
        model: Trained model
        history: Dictionary containing training history
    """
    if device is None:
        device = model.device
    
    model = model.to(device)
    model.train()
    
    # Initialize history dictionary to track metrics
    history = {
        "train_loss": [],
        "val_loss": [],
        "recon_loss": [],
        "kl_loss": []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    log_and_print(f"Starting training for {epochs} epochs")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        recon_losses = []
        kl_losses = []
        
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device).float()
            
            # Forward pass
            model.optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            
            # Calculate loss
            loss, recon_loss, kl_loss = model.loss_function(recon_batch, data, mu, logvar)
            
            # Backward pass and optimize
            loss.backward()
            model.optimizer.step()
            
            # Track losses
            train_losses.append(loss.item())
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())
        
        # Calculate average training metrics for this epoch
        avg_train_loss = np.mean(train_losses)
        avg_recon_loss = np.mean(recon_losses)
        avg_kl_loss = np.mean(kl_losses)
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_idx, (data,) in enumerate(val_loader):
                data = data.to(device).float()
                
                # Forward pass
                recon_batch, mu, logvar = model(data)
                
                # Calculate loss
                loss, _, _ = model.loss_function(recon_batch, data, mu, logvar)
                
                # Track validation loss
                val_losses.append(loss.item())
        
        # Calculate average validation loss
        avg_val_loss = np.mean(val_losses)
        
        # Update learning rate scheduler if applicable
        if hasattr(model, 'update_learning_rate'):
            model.update_learning_rate(avg_val_loss)
        
        # Update history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["recon_loss"].append(avg_recon_loss)
        history["kl_loss"].append(avg_kl_loss)
        
        # Log progress
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            log_and_print(f"Epoch {epoch+1}/{epochs} - "
                         f"Train Loss: {avg_train_loss:.4f}, "
                         f"Val Loss: {avg_val_loss:.4f}, "
                         f"Recon Loss: {avg_recon_loss:.4f}, "
                         f"KL Loss: {avg_kl_loss:.4f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                log_and_print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model if early stopping was triggered
    if best_model_state is not None and patience_counter >= early_stopping_patience:
        model.load_state_dict(best_model_state)
        log_and_print("Loaded best model based on validation loss")
    
    return model, history

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


def get_atlas_regions(atlas_name):
    """
    Get region names for a given atlas.
    
    Parameters:
    -----------
    atlas_name : str
        Name of the atlas
        
    Returns:
    --------
    list: List of region names
    """
    # Define atlas region mappings
    atlas_regions = {
        "cobra": ['lStriatum', 'lGloPal', 'lTha', 'lAntCerebLI_II', 'lAntCerebLIII', 'lAntCerebLIV', 'lAntCerebLV', 'lSupPostCerebLVI', 'lSupPostCerebCI', 'lSupPostCerebCII', 'lSupPostCerebLVIIB', 'lInfPostCerebLVIIIA', 'lInfPostCerebLVIIIB', 'lInfPostCerebLIX', 'lInfPostCerebLX', 'lAntCerebWM', 'lAmy', 'lHCA1', 'lSub', 'lFor', 'lCA4', 'lCA2_3', 'lStratum', 'lFimbra', 'lMamBody', 'lAlveus', 'rStriatum', 'rGloPal', 'rTha', 'rAntCerebLI_II', 'rAntCerebLIII', 'rAntCerebLIV', 'rAntCerebLV', 'rSupPostCerebLVI', 'rSupPostCerebCI', 'rSupPostCerebCII', 'rSupPostCerebLVIIB', 'rInfPostCerebLVIIIA', 'rInfPostCerebLVIIIB', 'rInfPostCerebLIX', 'rInfPostCerebLX', 'rAntCerebWM', 'rAmy', 'rHCA1', 'rSub', 'rFor', 'rCA4', 'rCA2_3', 'rStratum', 'rFimbra', 'rMamBody', 'rAlveus'],

        # Add other atlases as needed
        #row_names = df.index.tolist() -> automatisch machen lassen
    }
    
    # If atlas is not in our mapping, return None and let the function generate generic names
    return atlas_regions.get(atlas_name, None)

      