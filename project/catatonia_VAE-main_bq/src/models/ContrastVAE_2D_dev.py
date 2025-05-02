import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.cuda.amp import GradScaler, autocast

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
        self.hidden_dims = [hidden_dim_1, hidden_dim_2]
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
def bootstrap_train_normative_models(model_class, healthy_data, input_dim, n_bootstraps=100, epochs=50, batch_size=32, device=None):
    """
    Train multiple normative models with bootstrap resampling
    
    Args:
        model_class: Model class (NormativeVAE)
        healthy_data: Healthy control data as tensor
        input_dim: Input dimension
        n_bootstraps: Number of bootstrap samples
        epochs: Number of training epochs per bootstrap
        batch_size: Batch size
        device: Device for training
    
    Returns:
        bootstrap_models: List of trained models
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Training {n_bootstraps} bootstrap models on {device}...")
    
    # List for trained models
    bootstrap_models = []
    
    for b in range(n_bootstraps):
        print(f"\nBootstrap {b+1}/{n_bootstraps}")
        
        # Create bootstrap sample with replacement
        n_samples = len(healthy_data)
        bootstrap_indices = resample(range(n_samples), replace=True, n_samples=n_samples)
        bootstrap_data = healthy_data[bootstrap_indices]
        
        # Train-val split
        n_val = int(0.2 * len(bootstrap_data))
        val_indices = np.random.choice(len(bootstrap_data), n_val, replace=False)
        train_indices = np.array([i for i in range(len(bootstrap_data)) if i not in val_indices])
        
        train_data = bootstrap_data[train_indices]
        val_data = bootstrap_data[val_indices]
        
        # Create DataLoaders
        train_loader = DataLoader(
            TensorDataset(train_data),
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            TensorDataset(val_data),
            batch_size=batch_size,
            shuffle=False
        )
        
        # Initialize new model
        bootstrap_model = model_class(
            input_dim=input_dim,
            device=device
        )
        
        # Train model
        bootstrap_model, _ = train_normative_model(
            model=bootstrap_model,
            healthy_train_loader=train_loader,
            healthy_val_loader=val_loader,
            epochs=epochs,
            early_stopping_patience=10
        )
        
        # Save trained model
        bootstrap_models.append(bootstrap_model)
    
    return bootstrap_models


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
    
    # Thresholds
    for z in [2, 3]:
        plt.axvline(x=z, color='r', linestyle='--', alpha=0.7, 
                   label=f'Threshold (Z={z})')
    
    plt.xlabel('Global Deviation Score (Z-Score)')
    plt.ylabel('Number of Samples')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Percentage of samples above thresholds
    for z in [2, 3]:
        hc_above = 100 * np.mean(hc_scores > z)
        pat_above = 100 * np.mean(patient_scores > z)
        print(f"Z > {z}: {hc_above:.1f}% of HC and {pat_above:.1f}% of patients")
    
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
        device=device
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


# Add this to your main() function before the return statement
def integrate_regional_analysis(bootstrap_models, train_data, patient_data, annotations_patients, 
                              atlas_name, save_dir):
    """
    Integrate the regional analysis into the main script.
    
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
    None
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
    
    # Extract diagnoses from annotations
    if "Diagnosis" in annotations_patients.columns:
        patient_diagnoses = annotations_patients["Diagnosis"].tolist()
    else:
        logging.warning("No 'Diagnosis' column found in patient annotations. Using generic labels.")
        patient_diagnoses = ["Unknown"] * len(patient_data)
    
    # Get region names for the atlas
    atlas_regions = get_atlas_regions(atlas_name)
    
    # Create a new directory for regional analysis
    regional_dir = Path(save_dir) / "regional_analysis"
    regional_dir.mkdir(exist_ok=True)
    
    # Run the regional analysis pipeline
    logging.info(f"Starting regional analysis for atlas: {atlas_name}")
    
    from regional_deviation_analysis import run_regional_analysis_pipeline
    
    analysis_results = run_regional_analysis_pipeline(
        bootstrap_models=bootstrap_models,
        healthy_data=train_data,
        patient_data=patient_data,
        patient_diagnoses=patient_diagnoses,
        atlas_regions=atlas_regions,
        diagnosis_labels=diagnosis_labels,
        save_dir=regional_dir
    )
    
    logging.info(f"Regional analysis completed. Results saved to {regional_dir}")
    
    return analysis_results
