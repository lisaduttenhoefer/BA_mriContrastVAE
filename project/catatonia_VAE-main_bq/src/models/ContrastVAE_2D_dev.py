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
def train_normative_model(model, healthy_train_loader, healthy_val_loader, epochs=100, early_stopping_patience=20):
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


# Bootstrap training for normative models
# def bootstrap_train_normative_models(model_class, healthy_data, input_dim, n_bootstraps=100, epochs=50, batch_size=32, device=None):
#     """
#     Train multiple normative models with bootstrap resampling
    
#     Args:
#         model_class: Model class (NormativeVAE)
#         healthy_data: Healthy control data as tensor
#         input_dim: Input dimension
#         n_bootstraps: Number of bootstrap samples
#         epochs: Number of training epochs per bootstrap
#         batch_size: Batch size
#         device: Device for training
    
#     Returns:
#         bootstrap_models: List of trained models
#     """
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     print(f"Training {n_bootstraps} bootstrap models on {device}...")
    
#     # List for trained models
#     bootstrap_models = []
    
#     for b in range(n_bootstraps):
#         print(f"\nBootstrap {b+1}/{n_bootstraps}")
        
#         # Create bootstrap sample with replacement
#         n_samples = len(healthy_data)
#         bootstrap_indices = resample(range(n_samples), replace=True, n_samples=n_samples)
#         bootstrap_data = healthy_data[bootstrap_indices]
        
#         # Train-val split
#         n_val = int(0.2 * len(bootstrap_data))
#         val_indices = np.random.choice(len(bootstrap_data), n_val, replace=False)
#         train_indices = np.array([i for i in range(len(bootstrap_data)) if i not in val_indices])
        
#         train_data = bootstrap_data[train_indices]
#         val_data = bootstrap_data[val_indices]
        
#         # Create DataLoaders
#         train_loader = DataLoader(
#             TensorDataset(train_data),
#             batch_size=batch_size,
#             shuffle=True
#         )
        
#         val_loader = DataLoader(
#             TensorDataset(val_data),
#             batch_size=batch_size,
#             shuffle=False
#         )
        
#         # Initialize new model
#         bootstrap_model = model_class(
#             input_dim=input_dim,
#             device=device
#         )
        
#         # Train model
#         bootstrap_model, _ = train_normative_model(
#             model=bootstrap_model,
#             healthy_train_loader=train_loader,
#             healthy_val_loader=val_loader,
#             epochs=epochs,
#             early_stopping_patience=10
#         )
        
#         # Save trained model
#         bootstrap_models.append(bootstrap_model)
    
#     return bootstrap_models
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
        
        # # Generate and save visualizations (once every 5 models to avoid excessive plotting)
        # if i % 5 == 0 or i == n_bootstraps - 1:
        #     # Plot loss curves
        #     plot_learning_curves(
        #         history['train_loss'], 
        #         history['val_loss'], 
        #         history['kl_loss'], 
        #         history['recon_loss'],
        #         os.path.join(loss_dir, f"bootstrap_{i}_losses.png")
        #     )
            
        #     # Get latent representations
        #     with torch.no_grad():
        #         trained_model.eval()
        #         combined_data_tensor = combined_data.to(device)
        #         features = trained_model.encoder(combined_data_tensor)
        #         mu = trained_model.fc_mu(features)
        #         log_var = trained_model.fc_var(features)
        #         latent_vectors = trained_model.reparameterize(mu, log_var)
        
        #         # Reconstruct validation data
        #         valid_data_tensor = valid_data.to(device)
        #         valid_recon, _, _ = trained_model(valid_data_tensor)
            
        #     # # Plot latent space
        #     # plot_latent_space(
        #     #     latent_vectors=latent_vectors,
        #     #     labels=combined_labels,
        #     #     save_path=os.path.join(latent_dir, f"bootstrap_{i}_latent_tsne.png"),
        #     #     method='tsne',
        #     #     title=f'Bootstrap {i} (Train=0, Valid=1)'
            # )
            
            # plot_latent_space(
            #     latent_vectors=latent_vectors,
            #     labels=combined_labels,
            #     save_path=os.path.join(latent_dir, f"bootstrap_{i}_latent_umap.png"),
            #     method='umap',
            #     title=f'Bootstrap {i} (Train=0, Valid=1)'
            # )
            
    
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

# Compute deviation scores with bootstrap models
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


# Visualize deviation scores
def visualize_deviation_scores(hc_scores, patient_scores, title="Deviation Scores Distribution", save_path=None):
    """
    Visualize the distribution of deviation scores
    
    Args:
        hc_scores: Scores of healthy controls
        patient_scores: Scores of patients
        title: Plot title
        save_path: Path to save the figure or None
    """
    plt.figure(figsize=(10, 6))
    
    # Convert tensor to numpy if needed
    if isinstance(hc_scores, torch.Tensor):
        hc_scores = hc_scores.cpu().numpy()
    if isinstance(patient_scores, torch.Tensor):
        patient_scores = patient_scores.cpu().numpy()
    
    # Histogram for HC
    plt.hist(hc_scores, bins=30, alpha=0.5, label='Healthy Controls')
    
    # Histogram for patients
    plt.hist(patient_scores, bins=30, alpha=0.5, label='Patients')
    
    # # Thresholds
    # for z in [2, 3]:
    #     plt.axvline(x=z, color='r', linestyle='--', alpha=0.7, 
    #                label=f'Threshold (Z={z})')
    
    plt.xlabel('Global Deviation Score (Z-Score)')
    plt.ylabel('Number of Samples')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # # Percentage of samples above thresholds
    # for z in [2, 3]:
    #     hc_above = 100 * np.mean(hc_scores > z)
    #     pat_above = 100 * np.mean(patient_scores > z)
    #     print(f"Z > {z}: {hc_above:.1f}% of HC and {pat_above:.1f}% of patients")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# Plot deviation maps for different diagnoses
def plot_deviation_maps(diagnosis_scores, hc_scores, save_path=None, title="Deviation Scores by Diagnosis"):
    """
    Plot deviation scores for different diagnoses
    
    Args:
        diagnosis_scores: Dictionary with diagnosis names as keys and scores as values
        hc_scores: Scores of healthy controls
        save_path: Path to save the figure or None
        title: Plot title
    """
    plt.figure(figsize=(12, 7))
    
    # Convert tensor to numpy if needed
    if isinstance(hc_scores, torch.Tensor):
        hc_scores = hc_scores.cpu().numpy()
    
    # Box plot data
    data = [hc_scores] + [scores for diag, scores in diagnosis_scores.items()]
    labels = ['Healthy Controls'] + list(diagnosis_scores.keys())
    
    # Create boxplot
    box = plt.boxplot(data, patch_artist=True, labels=labels, showfliers=False)
    
    # Colors for groups
    colors = ['lightblue'] + ['lightgreen', 'salmon', 'wheat', 'lightpink', 'lightcyan'][:len(diagnosis_scores)]
    
    # Add colors to boxplots
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    # Add jittered points
    for i, d in enumerate([hc_scores] + [scores for diag, scores in diagnosis_scores.items()]):
        # Add jitter
        x = np.random.normal(i+1, 0.08, size=len(d))
        plt.scatter(x, d, alpha=0.4, s=20, c='k', edgecolors='none')
    
    # Add threshold lines
    plt.axhline(y=2, color='r', linestyle='--', alpha=0.7, label='Z=2 Threshold')
    plt.axhline(y=3, color='r', linestyle='-', alpha=0.7, label='Z=3 Threshold')
    
    plt.xlabel('Diagnostic Group')
    plt.ylabel('Global Deviation Score (Z-Score)')
    plt.title(title)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Print summary statistics
    print("Summary Statistics:")
    print(f"Healthy Controls: Mean={np.mean(hc_scores):.2f}, Median={np.median(hc_scores):.2f}, Std={np.std(hc_scores):.2f}")
    for diag, scores in diagnosis_scores.items():
        print(f"{diag}: Mean={np.mean(scores):.2f}, Median={np.median(scores):.2f}, Std={np.std(scores):.2f}")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# Main function for normative modeling
def run_normative_modeling_pipeline(healthy_data, patient_data, contrastvae=None,
                                  n_bootstraps=100, train_epochs=50, batch_size=32,
                                  save_dir="./results", device=None, save_models=True):
    """
    Run the complete normative modeling pipeline
    
    Args:
        healthy_data: Tensor with healthy control data
        patient_data: Tensor with patient data
        contrastvae: Optional pre-trained ContrastVAE model
        n_bootstraps: Number of bootstrap samples
        train_epochs: Number of training epochs per bootstrap
        batch_size: Batch size
        save_dir: Directory to save results
        device: Device for training
        save_models: Whether to save all bootstrap models
    
    Returns:
        bootstrap_models: List of trained models
        hc_global_z: Global Z-scores for HC
        patient_global_z: Global Z-scores for patients
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/models", exist_ok=True)
    os.makedirs(f"{save_dir}/figures", exist_ok=True)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_dim = healthy_data.shape[1]
    logging.info(f"Input dimension: {input_dim}")
    
    # 1. Bootstrap training
    logging.info(f"Starting bootstrap training with {n_bootstraps} models...")
    bootstrap_models = bootstrap_train_normative_models(
        model_class=NormativeVAE,
        healthy_data=healthy_data,
        input_dim=input_dim,
        n_bootstraps=n_bootstraps,
        epochs=train_epochs,
        batch_size=batch_size,
        #device=device
    )
    
    # Save models
    if save_models:
        logging.info("Saving bootstrap models...")
        for i, model in enumerate(bootstrap_models):
            torch.save(model.state_dict(), f"{save_dir}/models/bootstrap_model_{i}.pt")
    
    # 2. Calculate deviation scores
    logging.info("Computing deviation scores for healthy controls...")
    hc_z_scores, hc_global_z = compute_deviation_scores(
        bootstrap_models=bootstrap_models,
        data=healthy_data,
        batch_size=batch_size
    )
    
    logging.info("Computing deviation scores for patients...")
    patient_z_scores, patient_global_z = compute_deviation_scores(
        bootstrap_models=bootstrap_models,
        data=patient_data,
        batch_size=batch_size
    )
    
    # 3. Visualize deviation scores
    logging.info("Visualizing deviation scores...")
    visualize_deviation_scores(
        hc_scores=hc_global_z,
        patient_scores=patient_global_z,
        save_path=f"{save_dir}/figures/deviation_scores_distribution.png"
    )
    logging.info("Identifying top deviant regions...")
    identify_top_deviant_regions(patient_z_scores, atlas_regions=None, n_top=10)
    
    # 4. Save results as CSV
    logging.info("Saving results...")
    
    # HC Scores
    hc_results = pd.DataFrame({
        'sample_id': [f"HC_{i}" for i in range(len(hc_global_z))],
        'global_z_score': hc_global_z,
        'group': 'Healthy Control'
    })
    
    # Patient Scores
    patient_results = pd.DataFrame({
        'sample_id': [f"Patient_{i}" for i in range(len(patient_global_z))],
        'global_z_score': patient_global_z,
        'group': 'Patient'
    })
    
    # Combine and save
    all_results = pd.concat([hc_results, patient_results], axis=0)
    all_results.to_csv(f"{save_dir}/deviation_scores.csv", index=False)
    
    # 5. Save results as AnnData for further analysis (if available)
    if HAS_ANNDATA:
        # Feature-level Z-scores as AnnData
        combined_z_scores = np.vstack([hc_z_scores, patient_z_scores])
        
        # Create AnnData object
        adata = ad.AnnData(X=combined_z_scores)
        
        # Add metadata
        adata.obs['sample_id'] = all_results['sample_id'].values
        adata.obs['group'] = all_results['group'].values
        adata.obs['global_z_score'] = all_results['global_z_score'].values
        
        # Add feature names (if available)
        try:
            if hasattr(healthy_data, 'columns'):
                adata.var_names = healthy_data.columns
            else:
                adata.var_names = [f'feature_{i}' for i in range(input_dim)]
        except:
            adata.var_names = [f'feature_{i}' for i in range(input_dim)]
        
        # Save AnnData
        adata.write(f"{save_dir}/deviation_scores.h5ad")
    
    # Save detailed deviation scores for each feature
    hc_z_df = pd.DataFrame(hc_z_scores)
    hc_z_df['sample_id'] = hc_results['sample_id'].values
    hc_z_df['group'] = 'Healthy Control'
    hc_z_df['global_z_score'] = hc_global_z
    
    patient_z_df = pd.DataFrame(patient_z_scores)
    patient_z_df['sample_id'] = patient_results['sample_id'].values
    patient_z_df['group'] = 'Patient'
    patient_z_df['global_z_score'] = patient_global_z
    
    all_z_df = pd.concat([hc_z_df, patient_z_df], axis=0)
    all_z_df.to_csv(f"{save_dir}/feature_z_scores.csv", index=False)
    
    logging.info(f"Normative modeling pipeline completed. Results saved in {save_dir}")
    
    return bootstrap_models, hc_global_z, patient_global_z

#####

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
    
# def compute_regional_deviations(bootstrap_models, data, sample_ids=None, batch_size=32):
#     """
#     Calculate region-level deviation scores with a set of bootstrap models
    
#     Args:
#         bootstrap_models: List of trained bootstrap models
#         data: Data to test (tensor)
#         sample_ids: Optional list of sample IDs corresponding to data rows
#         batch_size: Batch size
    
#     Returns:
#         region_z_scores: Z-scores for each region and sample [samples, regions]
#         sample_ids: Original sample IDs in the same order as region_z_scores (if provided)
#     """
#     import torch
#     import numpy as np
#     from torch.utils.data import DataLoader, TensorDataset
#     from tqdm import tqdm
#     import logging
    
#     device = bootstrap_models[0].device
#     n_bootstraps = len(bootstrap_models)
#     n_samples = len(data)
#     n_regions = data.shape[1]
    
#     # Array for reconstruction errors per region
#     all_errors = np.zeros((n_bootstraps, n_samples, n_regions))
    
#     # Create DataLoader
#     data_loader = DataLoader(
#         TensorDataset(data),
#         batch_size=batch_size,
#         shuffle=False  # Important: keep original order for sample_ids matching
#     )
    
#     # Calculate reconstruction error for each bootstrap model
#     for b, model in enumerate(tqdm(bootstrap_models, desc="Computing Regional Deviations")):
#         model.eval()
#         batch_start = 0
        
#         with torch.no_grad():
#             for batch_idx, (x,) in enumerate(data_loader):
#                 x = x.to(device)
#                 batch_size_actual = x.shape[0]
                
#                 # Calculate reconstruction error
#                 recon_error = model.compute_reconstruction_error(x).cpu().numpy()
                
#                 # Store errors
#                 all_errors[b, batch_start:batch_start+batch_size_actual] = recon_error
#                 batch_start += batch_size_actual
    
#     # Mean and standard deviation across bootstrap models
#     mean_errors = np.mean(all_errors, axis=0)  # Mean across bootstraps [samples, regions]
#     std_errors = np.std(all_errors, axis=0)    # Std across bootstraps [samples, regions]
    
#     # Prevent division by zero
#     std_errors = np.maximum(std_errors, 1e-8)
    
#     # Calculate Z-scores for each region
#     region_z_scores = mean_errors / std_errors
    
#     # Return with sample IDs if provided
#     if sample_ids is not None:
#         # Make sure sample_ids length matches the data
#         if len(sample_ids) != len(region_z_scores):
#             logging.warning(f"Length mismatch: {len(region_z_scores)} samples in results but {len(sample_ids)} in sample_ids")
#             # Trim to match
#             sample_ids = sample_ids[:len(region_z_scores)]
#         return region_z_scores, sample_ids
#     else:
#         return region_z_scores

def identify_top_deviant_regions(patient_z_scores, atlas_regions=None, n_top=10):
    """
    Identify top deviant regions across samples
    
    Args:
        region_z_scores: Z-scores for each region and sample [samples, regions]
        atlas_regions: List of region names (optional)
        n_top: Number of top regions to return
    
    Returns:
        top_regions: Dictionary with statistics for top deviant regions
    """
    # Mean Z-score across samples
    mean_region_z = np.mean(patient_z_scores, axis=0)
    
    # If no region names are provided, create generic names
    if atlas_regions is None or len(atlas_regions) != patient_z_scores.shape[1]:
        atlas_regions = [f"Region_{i+1}" for i in range(patient_z_scores.shape[1])]
    
    # Find top deviant regions
    top_indices = np.argsort(mean_region_z)[::-1][:n_top]
    
    top_regions = {
        "region_names": [atlas_regions[i] for i in top_indices],
        "mean_z_scores": mean_region_z[top_indices],
        "indices": top_indices
    }
    
    return top_regions


def region_distribution_by_diagnosis(patient_z_scores, patient_diagnoses, patient_ids=None, sample_ids=None, diagnosis_labels=None):
    """
    Group regional Z-scores by diagnosis, ensuring data alignment between scores and diagnoses
    
    Args:
        region_z_scores: Z-scores for each region and sample [samples, regions]
        patient_diagnoses: List of diagnoses for each patient
        patient_ids: List of patient IDs corresponding to the diagnoses
        sample_ids: List of sample IDs corresponding to the region_z_scores
                   (if None, assumes patient_ids and region_z_scores are already aligned)
        diagnosis_labels: Dictionary mapping diagnosis codes to readable labels
    
    Returns:
        diagnosis_grouped: Dictionary with region scores grouped by diagnosis
    """
    import numpy as np
    import logging
    
    # Default diagnosis labels if not provided
    if diagnosis_labels is None:
        diagnosis_labels = {}
    
    # Convert to numpy array if not already
    if not isinstance(patient_z_scores, np.ndarray):
        patient_z_scores = np.array(patient_z_scores)
    
    # Check for dimension mismatch and fix it
    if len(patient_diagnoses) != len(patient_z_scores):
        logging.warning(f"Dimension mismatch: {len(patient_z_scores)} samples in patient_z_scores but {len(patient_diagnoses)} diagnoses")
        
        # If patient_ids and sample_ids are provided, we can properly align the data
        if patient_ids is not None and sample_ids is not None:
            logging.info("Aligning data using patient IDs and sample IDs")
            # Create a mapping from patient_ids to diagnoses
            patient_to_diagnosis = {pid: diag for pid, diag in zip(patient_ids, patient_diagnoses)}
            
            # Get diagnoses for only the samples we have data for
            aligned_diagnoses = []
            for sid in sample_ids:
                if sid in patient_to_diagnosis:
                    aligned_diagnoses.append(patient_to_diagnosis[sid])
                else:
                    # Handle case where we have data but no diagnosis
                    aligned_diagnoses.append("Unknown")
                    logging.warning(f"No diagnosis found for sample ID: {sid}")
            
            # Use the aligned diagnoses for grouping
            patient_diagnoses = aligned_diagnoses
        else:
            # If we can't properly align, just truncate the longer list
            logging.warning("No ID information for alignment. Truncating lists to match shortest length.")
            min_len = min(len(patient_z_scores), len(patient_diagnoses))
            patient_z_scores = patient_z_scores[:min_len]
            patient_diagnoses = patient_diagnoses[:min_len]
    
    # Group by diagnosis
    unique_diagnoses = set(patient_diagnoses)
    diagnosis_grouped = {}
    
    for diagnosis in unique_diagnoses:
        # Find samples with this diagnosis
        diagnosis_mask = np.array([d == diagnosis for d in patient_diagnoses])
        
        # Double-check the dimensions
        if len(diagnosis_mask) != len(patient_z_scores):
            logging.error(f"Dimension mismatch after alignment: {len(patient_z_scores)} samples vs {len(diagnosis_mask)} mask elements")
            continue
        
        # Skip if no samples with this diagnosis
        if not np.any(diagnosis_mask):
            logging.info(f"No samples found for diagnosis: {diagnosis}")
            continue
        
        # Get scores for this diagnosis
        diagnosis_scores = patient_z_scores[diagnosis_mask]
        
        # Use label if available, otherwise use the code
        label = diagnosis_labels.get(diagnosis, diagnosis)
        diagnosis_grouped[label] = diagnosis_scores
        logging.info(f"Found {len(diagnosis_scores)} samples for diagnosis: {label}")
    
    return diagnosis_grouped

def create_region_heatmap(patient_z_scores, atlas_regions=None, save_path=None):
    """
    Create a heatmap of region deviations
    
    Args:
        region_z_scores: Z-scores for each region and sample [samples, regions]
        atlas_regions: List of region names
        save_path: Path to save the figure
    """
    # If no region names are provided, create generic names
    if atlas_regions is None or len(atlas_regions) != patient_z_scores.shape[1]:
        atlas_regions = [f"Region_{i+1}" for i in range(patient_z_scores.shape[1])]
    
    # Mean Z-score across samples
    mean_region_z = np.mean(patient_z_scores, axis=0)
    
    # Sort regions by mean Z-score
    sorted_indices = np.argsort(mean_region_z)[::-1]
    sorted_regions = [atlas_regions[i] for i in sorted_indices]
    sorted_scores = mean_region_z[sorted_indices]
    
    # Create DataFrame for heatmap
    df = pd.DataFrame({'Region': sorted_regions, 'Mean Z-Score': sorted_scores})
    
    # Create plot
    plt.figure(figsize=(10, max(8, len(sorted_regions) * 0.3)))
    
    # Create heatmap
    sns.heatmap(
        df[['Mean Z-Score']].T,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=df['Region'],
        yticklabels=['Mean Z-Score'],
        cbar_kws={'label': 'Z-Score'}
    )
    
    plt.title('Region Deviation Scores')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def create_diagnosis_region_heatmap(diagnosis_grouped, atlas_regions=None, top_n=20, save_path=None):
    """
    Create a heatmap of top regions by diagnosis
    
    Args:
        diagnosis_grouped: Dictionary with region scores grouped by diagnosis
        atlas_regions: List of region names
        top_n: Number of top regions to show
        save_path: Path to save the figure
    """
    if not diagnosis_grouped:
        logging.warning("No diagnosis data available for heatmap")
        return
    
    # If no region names are provided, create generic names based on first diagnosis
    first_diag = list(diagnosis_grouped.keys())[0]
    first_data = diagnosis_grouped[first_diag]
    
    if atlas_regions is None or len(atlas_regions) != first_data.shape[1]:
        atlas_regions = [f"Region_{i+1}" for i in range(first_data.shape[1])]
    
    # Calculate mean Z-scores for each diagnosis and region
    diagnosis_means = {}
    for diagnosis, scores in diagnosis_grouped.items():
        diagnosis_means[diagnosis] = np.mean(scores, axis=0)
    
    # Combine all means
    all_means = np.vstack(list(diagnosis_means.values()))
    
    # Calculate overall region importance (mean across diagnoses)
    region_importance = np.mean(all_means, axis=0)
    
    # Get top N important regions
    top_indices = np.argsort(region_importance)[::-1][:top_n]
    top_regions = [atlas_regions[i] for i in top_indices]
    
    # Create data for heatmap
    heatmap_data = []
    for diagnosis, means in diagnosis_means.items():
        for idx in top_indices:
            heatmap_data.append({
                'Diagnosis': diagnosis,
                'Region': atlas_regions[idx],
                'Z-Score': means[idx]
            })
    
    heatmap_df = pd.DataFrame(heatmap_data)
    
    # Pivot for heatmap format
    pivot_df = heatmap_df.pivot(index='Region', columns='Diagnosis', values='Z-Score')
    
    # Sort regions by mean Z-score
    region_means = pivot_df.mean(axis=1)
    pivot_df = pivot_df.loc[region_means.sort_values(ascending=False).index]
    
    # Create plot
    plt.figure(figsize=(12, max(8, len(top_regions) * 0.4)))
    
    # Create heatmap
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_kws={'label': 'Mean Z-Score'}
    )
    
    plt.title('Top Deviant Regions by Diagnosis')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def visualize_region_deviations(patient_z_scores, atlas_regions=None, top_n=15, save_path=None):
    """
    Visualize the top deviant regions as a bar plot
    
    Args:
        region_z_scores: Z-scores for each region and sample [samples, regions]
        atlas_regions: List of region names
        top_n: Number of top regions to show
        save_path: Path to save the figure
    """
    
    # If no region names are provided, create generic names
    if atlas_regions is None or len(atlas_regions) != patient_z_scores.shape[1]:
        atlas_regions = [f"Region_{i+1}" for i in range(patient_z_scores.shape[1])]
    
    # Mean Z-score across samples
    mean_region_z = np.mean(patient_z_scores, axis=0)
    
    # Find top deviant regions
    top_indices = np.argsort(mean_region_z)[::-1][:top_n]
    top_regions = [atlas_regions[i] for i in top_indices]
    top_scores = mean_region_z[top_indices]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot horizontal bars
    y_pos = np.arange(len(top_regions))
    plt.barh(y_pos, top_scores, align='center', alpha=0.8, color='skyblue')
    plt.yticks(y_pos, top_regions)
    
    # Add values to the bars
    for i, v in enumerate(top_scores):
        plt.text(v + 0.1, i, f"{v:.2f}", va='center')
    
    plt.xlabel('Mean Z-Score')
    plt.title('Top Deviant Brain Regions')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Add a vertical line at threshold 2.0
    plt.axvline(x=2.0, color='red', linestyle='--', alpha=0.7, label='Z=2 Threshold')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def run_regional_analysis_pipeline(bootstrap_models, healthy_data, patient_data, 
                                 patient_diagnoses=None, patient_ids=None, sample_ids=None,
                                 atlas_regions=None, diagnosis_labels=None, save_dir=None, batch_size=32,):
    """
    Run the regional deviation analysis pipeline with improved data alignment
    
    Args:
        bootstrap_models: List of trained bootstrap models
        healthy_data: Tensor of healthy control data
        patient_data: Tensor of patient data
        patient_diagnoses: List of diagnoses for patient samples (optional)
        patient_ids: List of patient IDs corresponding to diagnoses (optional)
        sample_ids: List of sample IDs corresponding to patient_data rows (optional)
        atlas_regions: List of region names (optional)
        diagnosis_labels: Dictionary mapping diagnosis codes to readable labels
        save_dir: Directory to save results
    
    Returns:
        analysis_results: Dictionary with analysis results
    """
    
    
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
    
    # 1. Compute regional deviations
    logging.info("Computing regional deviations for healthy controls...")
    # hc_region_z_scores = compute_regional_deviations(
    #     bootstrap_models=bootstrap_models,
    #     data=healthy_data
    # )
    hc_patient_z_scores, hc_region_global_z = compute_deviation_scores(
        bootstrap_models=bootstrap_models,
        data=healthy_data, 
        batch_size=batch_size,  
     )
    
    logging.info("Computing regional deviations for patients...")
    if sample_ids is None:
        # If no sample IDs provided, compute without them
        patient_patient_z_scores, patient_region_global_z = compute_deviation_scores(
        bootstrap_models=bootstrap_models,
        data=patient_data,
        batch_size=batch_size,
    )
    else:
        # Compute with sample IDs for tracking
        patient_patient_z_scores, aligned_sample_ids = compute_deviation_scores(
        bootstrap_models=bootstrap_models,
        data=patient_data,
        batch_size=batch_size,
        )
    
    # Save region z-scores with IDs if available
    if save_dir is not None:
        # Create DataFrame with region z-scores
        patient_z_df = pd.DataFrame(patient_patient_z_scores)
        
        # Add sample/patient IDs if available
        if sample_ids is not None:
            patient_z_df.insert(0, 'sample_id', sample_ids[:len(patient_patient_z_scores)])
        
        if patient_ids is not None and len(patient_ids) == len(patient_patient_z_scores):
            patient_z_df.insert(0, 'patient_id', patient_ids[:len(patient_patient_z_scores)])
        
        # Save to CSV
        patient_z_df.to_csv(save_dir / "patient_patient_z_scores_with_ids.csv", index=False)
    
    # 2. Identify top deviant regions
    logging.info("Identifying top deviant regions...")
    hc_top_regions = identify_top_deviant_regions(
        patient_z_scores=hc_patient_z_scores,
        atlas_regions=atlas_regions
    )
    
    patient_top_regions = identify_top_deviant_regions(
        patient_z_scores=patient_patient_z_scores,
        atlas_regions=atlas_regions
    )
    
    # 3. Analysis by diagnosis (if available)
    diagnosis_grouped = None
    if patient_diagnoses is not None:
        logging.info("Grouping regional deviations by diagnosis...")
        
        # Use our updated function with sample IDs for alignment
        diagnosis_grouped = region_distribution_by_diagnosis(
            patient_z_scores=patient_patient_z_scores,
            patient_diagnoses=patient_diagnoses,
            patient_ids=patient_ids,
            sample_ids=sample_ids if sample_ids is not None else None,
            diagnosis_labels=diagnosis_labels
        )
    
    # 4. Visualizations
    if save_dir is not None:
        # Create figures directory
        figures_dir = save_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        logging.info("Creating visualizations...")
        
        # Visualize top deviant regions
        try:
            visualize_region_deviations(
                patient_z_scores=patient_patient_z_scores,
                atlas_regions=atlas_regions,
                top_n=15,
                save_path=figures_dir / "top_deviant_regions.png"
            )
        except Exception as e:
            logging.error(f"Error creating region deviation visualization: {e}")
        
        # Create region heatmap
        try:
            create_region_heatmap(
                patient_z_scores=patient_patient_z_scores,
                atlas_regions=atlas_regions,
                save_path=figures_dir / "region_heatmap.png"
            )
        except Exception as e:
            logging.error(f"Error creating region heatmap: {e}")
        
        # Create diagnosis-specific heatmap
        if diagnosis_grouped:
            try:
                create_diagnosis_region_heatmap(
                    diagnosis_grouped=diagnosis_grouped,
                    atlas_regions=atlas_regions,
                    top_n=20,
                    save_path=figures_dir / "diagnosis_region_heatmap.png"
                )
            except Exception as e:
                logging.error(f"Error creating diagnosis heatmap: {e}")
    
    # 5. Save results
    if save_dir is not None:
        logging.info("Saving analysis results...")
        
        # Save region z-scores
        np.save(save_dir / "hc_patient_z_scores.npy", hc_patient_z_scores)
        np.save(save_dir / "patient_patient_z_scores.npy", patient_patient_z_scores)
        
        # Save top regions as CSV
        if atlas_regions is not None:
            hc_top_df = pd.DataFrame({
                'Region': hc_top_regions['region_names'],
                'Mean Z-Score': hc_top_regions['mean_z_scores']
            })
            hc_top_df.to_csv(save_dir / "hc_top_regions.csv", index=False)
            
            patient_top_df = pd.DataFrame({
                'Region': patient_top_regions['region_names'],
                'Mean Z-Score': patient_top_regions['mean_z_scores']
            })
            patient_top_df.to_csv(save_dir / "patient_top_regions.csv", index=False)
    
    # 6. Return results
    analysis_results = {
        "hc_patient_z_scores": hc_patient_z_scores,
        "patient_patient_z_scores": patient_patient_z_scores,
        "hc_top_regions": hc_top_regions,
        "patient_top_regions": patient_top_regions,
        "diagnosis_grouped": diagnosis_grouped
    }
    
    return analysis_results

def integrate_regional_analysis(bootstrap_models, train_data, patient_data, annotations_patients, 
                              atlas_name, save_dir):
    """
    Integrate the regional analysis into the main script with improved data alignment
    
    Parameters:
    -----------
    bootstrap_models : list
        List of trained normative VAE models from bootstrap samples
    train_data : torch.Tensor
        Tensor of healthy control ROI measurements
    patient_data : torch.Tensor
        Tensor of patient ROI measurements
    annotations_patients : pandas.DataFrame
        DataFrame containing patient metadata including diagnoses
    atlas_name : str
        Name of the brain atlas used
    save_dir : str or Path
        Directory to save results
        
    Returns:
    --------
    dict: Analysis results
    """
    import logging
    import numpy as np
    import pandas as pd
    from pathlib import Path
    
    # Create a dictionary to map diagnosis codes to readable labels
    diagnosis_labels = {
        "CTT": "Catatonia",
        "SCHZ": "Schizophrenia",
        "MDD": "Major Depression",
        "HC": "Healthy Control"
    }
    
    # Create a new directory for regional analysis
    regional_dir = Path(save_dir) / "regional_analysis"
    regional_dir.mkdir(exist_ok=True)
    
    # Extract patient IDs - check for common ID column names
    patient_ids = []
    id_columns = ["ID", "PatientID", "SubjectID", "Subject_ID", "Patient_ID", "sample_id"]
    id_column_found = None
    
    for col in id_columns:
        if col in annotations_patients.columns:
            patient_ids = annotations_patients[col].tolist()
            id_column_found = col
            print(f"Found patient IDs in column '{col}', total: {len(patient_ids)}")
            break
    
    if not patient_ids:
        logging.warning("No patient ID column found. Using generic IDs.")
        patient_ids = [f"Patient_{i}" for i in range(len(annotations_patients))]
    
    # Extract diagnoses and patient IDs from annotations
    patient_diagnoses = []
    
    if "Diagnosis" in annotations_patients.columns:
        patient_diagnoses = annotations_patients["Diagnosis"].tolist()
        print(f"Found {len(patient_diagnoses)} diagnoses in annotations")
    else:
        logging.warning("No 'Diagnosis' column found in patient annotations. Using generic labels.")
        patient_diagnoses = ["Unknown"] * len(annotations_patients)
    
    # Generate sample IDs for our actual data (these might not match the annotation IDs)
    sample_ids = [f"Sample_{i}" for i in range(len(patient_data))]
    
    # Check for dimension mismatch
    if len(patient_diagnoses) != len(sample_ids):
        logging.warning(f"Dimension mismatch: {len(sample_ids)} samples in data but {len(patient_diagnoses)} diagnoses in annotations")
        
        # If we have both patient IDs in annotations and an actual ID column in the data, we would align here
        # For now, we'll just create an ID mapping for consistency
        
        # Check if we have the same number of patients in annotations and data
        if len(annotations_patients) == len(patient_data):
            logging.info("Same number of records in annotations and data. Creating 1:1 mapping.")
            # Create a direct mapping between annotation rows and data samples
            sample_to_patient = {sample_id: patient_id for sample_id, patient_id in zip(sample_ids, patient_ids)}
        else:
            logging.warning("Different number of records. Cannot establish reliable 1:1 mapping.")
            # In a real scenario, you would have actual IDs in your data to match with annotations
            sample_to_patient = {}
    else:
        # Direct 1:1 mapping
        sample_to_patient = {sample_id: patient_id for sample_id, patient_id in zip(sample_ids, patient_ids)}
    
    # Get region names for the atlas
    atlas_regions = get_atlas_regions(atlas_name)
    
    # Run the regional analysis pipeline
    logging.info(f"Starting regional analysis for atlas: {atlas_name}")
    
    #from regional_deviation_analysis import run_regional_analysis_pipeline
    
    analysis_results = run_regional_analysis_pipeline(
        bootstrap_models=bootstrap_models,
        healthy_data=train_data,
        patient_data=patient_data,
        patient_diagnoses=patient_diagnoses,
        patient_ids=patient_ids,      # Pass patient IDs from annotations
        sample_ids=sample_ids,        # Pass sample IDs for data alignment
        atlas_regions=atlas_regions,
        diagnosis_labels=diagnosis_labels,
        save_dir=regional_dir
    )
    
    logging.info(f"Regional analysis completed. Results saved to {regional_dir}")
    
    return analysis_results
      