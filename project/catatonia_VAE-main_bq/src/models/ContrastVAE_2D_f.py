import sys
import matplotlib
import logging
from typing import Dict, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchio as tio
from pytorch_msssim import ssim
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
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
)
# ContrastVAE_2D Model Class
class ContrastVAE_2D(nn.Module):
    def __init__(
        self,
        num_classes,
        learning_rate,
        weight_decay,
        recon_loss_weight,
        kldiv_loss_weight,
        contr_loss_weight,
        scaler,
        contr_temperature=0.1,
        input_dim=100,  # Need to specify input dimension
        hidden_dim_1=100,
        hidden_dim_2=100,
        latent_dim=20,
        device=None,
        dropout_prob=0.1,
        schedule_on_validation=True,
        scheduler_patience=10,
        scheduler_factor=0.5,
    ):
        super(ContrastVAE_2D, self).__init__()
        
        self.num_classes = num_classes
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.LeakyReLU(1e-2),
            
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.LeakyReLU(1e-2),
            
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
            
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.LeakyReLU(1e-2),
            
            nn.Linear(hidden_dim_1, input_dim)
        )
        
        # Set training params
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Scheduler for learning rate
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            verbose=True,
        )
        
        self.scaler = scaler
        
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
        
        # Initialize weights
        self.apply(self.weights_init)
    
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
        x = self.encoder(x)
        
        # Get latent space
        mu = self.fc_mu(x)
        
        return mu
    
    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        # Set to train mode
        self.train()
        total_loss, contr_loss, recon_loss, kldiv_loss = 0.0, 0.0, 0.0, 0.0
        
        # Iterate through batches
        for batch_idx, (measurements, labels, names) in enumerate(train_loader):
            # Move to device
            batch_measurements = torch.stack([measurement for measurement in measurements]).to(self.device)
            batch_labels = torch.stack([label for label in labels]).to(self.device)
            
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
            # Move to device
            batch_measurements = torch.stack([measurement for measurement in measurements]).to(self.device)
            batch_labels = torch.stack([label for label in labels]).to(self.device)
            
            # Calculate loss
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
            
            # Update loss stats
            total_loss += b_total_loss.item()
            contr_loss += b_contr_loss.item()
            recon_loss += b_recon_loss.item()
            kldiv_loss += b_kldiv_loss.item()
        
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
    
    @torch.no_grad()
    def extract_latent_space(self, data_loader, data_type):
        """Extract latent space representations"""
        log_extracting_latent_space(data_type)
        
        self.eval()
        latent_spaces = []
        sample_names = []
        
        for batch_idx, (measurements, labels, names) in enumerate(data_loader):
            # Move to device
            batch_measurements = torch.stack([measurement.to(self.device) for measurement in measurements])
            
            # Get latent representations
            mu = self.to_latent(batch_measurements)
            
            # Collect outputs
            latent_spaces.append(mu.cpu().numpy())
            sample_names.extend(names)
        
        # Create anndata object (simulated here)
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

def train_ContrastVAE_2D(
    model,
    train_loader,
    valid_loader,
    annotations,
    model_metrics,
    config,
    accuracy_tune=False,
    no_plotting=True,
    no_val_plotting=False,
    no_saving=False
):
    """Train the ContrastVAE_2D model"""
    log_training_start()
    
    # Train and validate for each epoch
    for epoch in tqdm(
        range(config.START_EPOCH, config.FINAL_EPOCH),
        initial=config.START_EPOCH,
        ncols=100,
        desc="\nEpochs",
    ):
        # Separate log messages from different epochs
        print("\n")

        # determine if it is a checkpoint epoch
        is_checkpoint = (
            epoch % config.CHECKPOINT_INTERVAL == 0 or epoch == config.FINAL_EPOCH - 1
        )

       
       
       
        # Validate the model
        valid_metrics = model.validate(valid_loader=valid_loader, epoch=epoch)
        
        # Train for one epoch
        train_metrics = model.train_one_epoch(train_loader=train_loader, epoch=epoch)
        
        # Track best accuracy
        best_accuracy = 0
        if epoch != config.START_EPOCH:
            if valid_metrics["accuracy"] > best_accuracy:
                best_accuracy = valid_metrics["accuracy"]
                if not no_saving:
                    save_model(
                        model=model,
                        save_path=config.MODEL_DIR,
                        timestamp=config.TIMESTAMP,
                        descriptor=config.RUN_NAME,
                        epoch='best',
                    )
        
        # Update metrics
        model_metrics = update_performance_metrics(model_metrics, [train_metrics, valid_metrics])
        
        # Handle checkpoint operations
        if is_checkpoint:

            if not no_plotting:
            # extract latent space and process it
                valid_latent = model.extract_latent_space(valid_loader, "Validation Data")
                train_latent = model.extract_latent_space(train_loader, "Training Data")

               
                train_latent = process_latent_space(
                    adata=train_latent,
                    annotations=annotations,
                    umap_neighbors=config.UMAP_NEIGHBORS,
                    seed=config.SEED,
                    save_data=True,
                    save_path=config.DATA_DIR,
                    timestamp=config.TIMESTAMP,
                    epoch=epoch,
                    data_type="train",
                )
                valid_latent = process_latent_space(
                    adata=valid_latent,
                    annotations=annotations,
                    umap_neighbors=config.UMAP_NEIGHBORS,
                    seed=config.SEED,
                    save_data=True,
                    save_path=config.DATA_DIR,
                    timestamp=config.TIMESTAMP,
                    epoch=epoch,
                    data_type="valid",
                )
                # combination of validation & training -> similar?
               
                combi_latent = combine_latent_spaces(
                    tdata=train_latent,
                    vdata=valid_latent,
                    umap_neighbors=config.UMAP_NEIGHBORS,
                    seed=config.SEED,
                    save_data=True,
                    save_path=config.DATA_DIR,
                    timestamp=config.TIMESTAMP,
                    epoch=epoch,
                    data_type="combined",
                )
              
                # plot latent space umap
                latent_space_plot(
                    data=combi_latent,
                    epoch=epoch,
                    plot_types=["umap"],
                    plot_by=[
                        "Diagnosis", 
                        "Dataset",
                        "Sex",
                        "Age",
                    ],
                    data_type="combined",
                    save_path=config.FIGURES_DIR,
                    timestamp=config.TIMESTAMP,
                    save=True,
                    show=False,
                )

                # plot latent space umap with detailed annotations -> combined & alone
                # If there's only validation data, we use bigger dots as there's less data
                for data, data_type, descriptor, size in zip(
                    [combi_latent, combi_latent, valid_latent],
                    ["train", "valid", "valid"],
                    ["", "", "_solo"],
                    [
                        config.UMAP_DOT_SIZE,
                        config.UMAP_DOT_SIZE,
                        config.UMAP_DOT_SIZE * 3,
                    ],
                ):
                    latent_space_details_plot(
                        data=data,
                        epoch=epoch,
                        plot_by=[
                            "Diagnosis",
                            "Sex",
                            "Age",
                        ],  # Data Type is set, Dataset makes no sense
                        data_type=data_type,
                        save_path=config.FIGURES_DIR,
                        timestamp=config.TIMESTAMP,
                        save=True,
                        show=False,
                        descriptor=descriptor,
                        size=size,
                    )

                # plot latent space batch plot. This shows off batch effect in individual diagnoses well
                # since it has less data, we use bigger dots
                for data, data_type, size in zip(
                    [train_latent, valid_latent],
                    ["train", "valid"],
                    [config.UMAP_DOT_SIZE * 3, config.UMAP_DOT_SIZE * 9],
                ):
                    latent_space_batch_plot(
                        data=data,
                        epoch=epoch,
                        data_type=data_type,
                        save_path=config.FIGURES_DIR,
                        timestamp=config.TIMESTAMP,
                        save=True,
                        show=False,
                        size=size,
                    )

                # plot reconstructed images that the decoder produces
                recon_images_plot(
                    data_loader=valid_loader,
                    model=model,
                    save_path=config.FIGURES_DIR,
                    timestamp=config.TIMESTAMP,
                    epoch=epoch,
                    slice_index=64,
                    device=config.DEVICE,
                    n_model_outputs=4,
                )

            # log that we've done the checkpoint plots
            log_checkpoint(figure_path=config.FIGURES_DIR)

            
            if epoch != config.START_EPOCH and not no_saving:
                save_model(
                    model=model,
                    save_path=config.MODEL_DIR,
                    timestamp=config.TIMESTAMP,
                    descriptor=config.RUN_NAME,
                    epoch=epoch,
                )
            
            # Early stopping check
            if config.EARLY_STOPPING:
                if model_metrics["learning_rate"].iloc[-1] <= config.STOP_LEARNING_RATE:
                    log_early_stopping(
                        model_metrics["learning_rate"].iloc[-1],
                        config.STOP_LEARNING_RATE,
                        epoch,
                    )
                    break
    
    # Save final metrics
    if not no_val_plotting:
        metrics_plot(
            metrics=model_metrics,
            save_path=config.FIGURES_DIR,
            timestamp=config.TIMESTAMP,
            skip_first_n_epochs=config.DONT_PLOT_N_EPOCHS,
        )
        save_model_metrics(
            model_metrics=model_metrics,
            save_path=config.MODEL_DIR,
            timestamp=config.TIMESTAMP,
            descriptor=config.RUN_NAME,
            epoch=epoch,
        )
    
    # Log best accuracy
    log_and_print(f"Best Accuracy: {model_metrics['accuracy'].max()} at epoch {model_metrics['accuracy'].idxmax()}")
    
    end_logging(config)
